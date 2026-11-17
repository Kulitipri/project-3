"""Fixed-size training cohorts with different session-start coverage and a shared future test cohort."""
import argparse
import copy
import shutil
from pathlib import Path

import pandas as pd
import yaml

from src.config import load_config
from src.seed_study import run_step
from src.utils import ensure_dir, run_metadata, save_json


def make_cohorts(sessions, original, context, settings, sample_seed):
    n_train, n_val = len(original['train']), len(original['val'])
    origin = sessions.StartTime.min()
    stream_seconds = float(context.EventTimes.map(lambda t: t[-1]).max())
    windows = {'first_sessions': None, 'first_2h': settings['hours'] * 3600,
               'first_fraction': settings['time_fraction'] * stream_seconds}
    common_cutoff = origin + pd.to_timedelta(max(v for v in windows.values() if v is not None), unit='s')
    common_test = sessions.loc[sessions.StartTime > common_cutoff].sort_values('StartTime', kind='stable').reset_index(drop=True)
    cohorts, descriptions = {}, {}
    ends = context.set_index('BlockId').EventTimes.map(lambda t: float(t[-1]))
    for name, seconds in windows.items():
        if seconds is None:
            train, val = original['train'].copy(), original['val'].copy()
            cutoff = max(train.StartTime.max(), val.StartTime.max())
        else:
            cutoff = origin + pd.to_timedelta(seconds, unit='s')
            pool = sessions.loc[(sessions.Label == 0) & (sessions.StartTime <= cutoff)]
            if len(pool) < n_train + n_val:
                raise ValueError(f'{name}: insufficient Normal sessions: {len(pool)}')
            sample = pool.sample(n=n_train + n_val, random_state=sample_seed).sort_values('StartTime', kind='stable')
            train, val = sample.iloc[:n_train].copy(), sample.iloc[n_train:].copy()
        if len(train) != n_train or len(val) != n_val:
            raise ValueError('Training/validation size changed')
        if train.StartTime.max() > val.StartTime.min():
            raise ValueError('Validation must be the latest part of the selected cohort')
        used = pd.concat([train, val])
        if used.Label.any() or set(used.BlockId) & set(common_test.BlockId):
            raise ValueError('Label leakage or overlapping session IDs')
        if used.StartTime.max() > common_cutoff:
            raise ValueError('Training sessions start after common cutoff')
        cohorts[name] = {'train': train.reset_index(drop=True), 'val': val.reset_index(drop=True), 'test': common_test}
        descriptions[name] = {'train': n_train, 'val': n_val, 'test': len(common_test),
                              'nominal_start_cutoff': str(cutoff), 'common_test_after': str(common_cutoff),
                              'train_start_min': str(train.StartTime.min()), 'train_start_max': str(train.StartTime.max()),
                              'train_start_span_hours': (train.StartTime.max()-train.StartTime.min()).total_seconds()/3600,
                              'last_training_event_hours': float(ends.loc[train.BlockId].max()/3600),
                              'last_train_val_event_hours': float(ends.loc[used.BlockId].max()/3600),
                              'train_val_sessions_ending_after_test_cutoff': int((ends.loc[used.BlockId] > (common_cutoff-origin).total_seconds()).sum())}
    return cohorts, descriptions


def prepare(cfg):
    paths, settings = cfg['paths'], cfg['time_coverage']
    sessions = pd.read_pickle(paths['sessions'])
    context = pd.read_pickle(paths['block_context'])
    original = {name: pd.read_pickle(paths['splits_dir'] / f'{name}.pkl') for name in ('train', 'val')}
    cohorts, descriptions = make_cohorts(sessions, original, context, settings, cfg['select']['tune_seed'])
    configs = {}
    for name, splits in cohorts.items():
        split_dir = ensure_dir(paths['coverage_splits_dir'] / name)
        result_dir = ensure_dir(paths['coverage_results_dir'] / name)
        model_dir = ensure_dir(paths['coverage_models_dir'] / name)
        for split, frame in splits.items():
            frame.to_pickle(split_dir / f'{split}.pkl')
        run_cfg = copy.deepcopy(cfg)
        run_cfg.pop('config_file', None)
        run_cfg['paths'] = {k: str(v) for k, v in paths.items()}
        run_cfg['paths'].update(splits_dir=str(split_dir), results_dir=str(result_dir), model_dir=str(model_dir))
        if name != 'first_sessions':
            run_cfg['split'].update(coverage_window_hours=float(settings['hours'] if name == 'first_2h'
                                    else settings['time_fraction'] * context.EventTimes.map(lambda t: t[-1]).max() / 3600),
                                    sampling_seed=cfg['select']['tune_seed'], validation_strategy='latest')
        config_path = result_dir / 'config.yaml'
        config_path.write_text(yaml.safe_dump(run_cfg, allow_unicode=True, sort_keys=False))
        configs[name] = config_path
        save_json({'meta': run_metadata(load_config(config_path)), **descriptions[name],
                   'interpretation': 'Offline session-start coverage sensitivity; complete training sessions extend beyond start cutoffs.'},
                  split_dir / 'summary.json')
        if name == 'first_sessions':
            # Exact same train/val BlockIds as the original; reuse weights with original provenance intact.
            for seed in cfg['robustness']['seeds']:
                target = ensure_dir(model_dir / 'seeds' / f'seed{seed}') / 'deeplog_best.pt'
                shutil.copy2(paths['model_dir'] / 'seeds' / f'seed{seed}' / 'deeplog_best.pt', target)
    save_json({'meta': run_metadata(cfg), 'cohorts': descriptions,
               'sampling_seed': cfg['select']['tune_seed'], 'validation_strategy': 'latest', 'model_seeds': cfg['robustness']['seeds'],
               'warning': 'No strict online-training claim: session-start split permits later events in training sessions.'},
              paths['coverage_results_dir'] / 'design.json')
    return configs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/config.yaml')
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    cfg = load_config(args.config)
    configs = prepare(cfg)
    print('Prepared:', {k: str(v) for k, v in configs.items()}, flush=True)
    if args.run:
        import subprocess
        import sys
        for name, config in configs.items():
            command = [sys.executable, '-m', 'src.seed_study', '--config', str(config)]
            if name == 'first_sessions':
                command.append('--reuse-models')
            subprocess.run(command, check=True)
            run_step('src.ensemble', config)


if __name__ == '__main__':
    main()
