"""Audit result provenance; archive unverifiable artifacts without deleting them."""
import argparse
from datetime import datetime
from pathlib import Path
import re
import shutil

from src.config import PROJECT_ROOT, load_config
from src.utils import load_json, save_json, run_metadata


def metadata_for(path: Path):
    candidates = [path] if path.suffix == '.json' else [path.with_suffix('.meta.json')]
    stem = path.stem
    if stem.startswith('deeplog_predictions_'):
        candidates.append(path.with_name(stem.replace('predictions', 'metrics') + '.json'))
    if stem.startswith('model_selection_') and stem.endswith('_grid'):
        candidates.append(path.with_name(stem.removesuffix('_grid') + '.json'))
    if stem.startswith('hybrid_'):
        candidates.append(path.parent / 'hybrid_detection.json')
    if stem.startswith('operating_points_'):
        if stem.endswith('_grid'):
            candidates.append(path.with_name(stem.removesuffix('_grid') + '.json'))
        candidates.extend([path.with_suffix('.json'), path.with_suffix('.meta.json')])
    if stem.startswith('seed_study'):
        candidates.append(path.parent / 'seed_study_meta.json')
    if stem.startswith('ensemble_'):
        candidates.append(path.parent / 'ensemble_meta.json')
    if stem == 'deeplog_ranks':
        candidates.append(path.parent / 'deeplog_ranks_meta.json')
    for candidate in candidates:
        if candidate.exists():
            value = load_json(candidate)
            if isinstance(value, dict):
                meta = value.get('meta', value)
                if isinstance(meta, dict) and 'config' in meta:
                    return meta, value
    return None, None


def differences(old, cfg, checkpoint=False):
    diffs = []
    for section in ('parse', 'model', 'train', 'split', 'baseline', 'select', 'hybrid', 'robustness', 'operating'):
        if checkpoint and section not in ('model', 'train', 'split'):
            continue
        for key in set(old.get(section, {})) | set(cfg.get(section, {})):
            if (section, key) in (('train', 'device'), ('select', 'tune_mode')):
                continue
            if old.get(section, {}).get(key) != cfg.get(section, {}).get(key):
                diffs.append(f'{section}.{key}')
    for key in ('raw_log', 'label_file', 'drain_config', 'sessions', 'event_index', 'splits_dir', 'block_context'):
        if key in old.get('paths', {}) and PROJECT_ROOT / old['paths'][key] != cfg['paths'][key]:
            diffs.append(f'paths.{key}')
    return diffs


def audit(cfg, apply=False):
    base = cfg['paths']['results_dir']
    paths = sorted(p for p in base.rglob('*') if p.is_file() and p.name != '.gitkeep')
    # The chronological root also contains the separate random experiment.
    if cfg['split']['strategy'] == 'chronological':
        paths = [p for p in paths if not p.is_relative_to(base / 'random')]
    rows = []
    for path in paths:
        old_config = None
        if path.suffix == '.yaml':
            old_config = load_config(path)
            reasons = differences(old_config, cfg)
        else:
            meta, value = metadata_for(path)
            reasons = ['missing_metadata'] if meta is None else differences(meta['config'], cfg)
            if meta is not None:
                old_config = meta['config']
            if value and 'checkpoints' in value:
                for checkpoint in value['checkpoints']:
                    reasons += ['missing_checkpoint_metadata'] if checkpoint is None else differences(checkpoint['config'], cfg, checkpoint=True)
        if old_config is not None:
            expected_seed = cfg['seed']
            for part in path.relative_to(base).parts:
                if match := re.fullmatch(r'seed(\d+)', part):
                    expected_seed = int(match[1])
                    if expected_seed not in cfg['robustness']['seeds']:
                        reasons.append('unconfigured_seed')
            if old_config.get('seed') != expected_seed:
                reasons.append('seed')
        rows.append({'path': str(path.relative_to(PROJECT_ROOT)), 'reasons': sorted(set(reasons)),
                     'action': 'archive' if reasons else 'keep'})
    if apply:
        target = cfg['paths']['archive_dir'] / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        # Decide all actions before moving companion metadata files.
        for row in rows:
            if row['action'] == 'archive':
                src = PROJECT_ROOT / row['path']
                dst = target / src.relative_to(cfg['paths']['results_dir'])
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), dst)
                row['archived_to'] = str(dst.relative_to(PROJECT_ROOT))
        save_json({'meta': run_metadata(cfg),
                   'note': 'Missing provenance does not establish batch size 2048.', 'files': rows}, target / 'audit_manifest.json')
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/config.yaml')
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    rows = audit(load_config(args.config), args.apply)
    archived = [r for r in rows if r['action'] == 'archive']
    print(f'Audited {len(rows)} files; {len(archived)} archive, {len(rows)-len(archived)} keep.')
    for row in archived:
        print(row['path'], ', '.join(row['reasons']))


if __name__ == '__main__':
    main()
