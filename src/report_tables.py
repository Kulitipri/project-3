"""Generate report tables from complete CSVs; never select variants using eval metrics."""
import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import load_config
from src.seed_study import TUNE_MODES
from src.utils import ensure_dir, run_metadata, save_json

METRICS = ['f1', 'precision', 'recall', 'recall_unseen', 'recall_known', 'fp_per_hour',
           'anomalies_unseen', 'anomalies_known']
SAME_TRAIN = ['deeplog', 'rules', 'hybrid', 'pca/train_normal', 'iforest/train_normal']
ALL_FITS = ['deeplog', 'rules', 'rule_unseen', 'rule_completion', 'hybrid', 'pca', 'iforest']


def flag_selected_variants(frame, operating=False):
    """Mark winners independently from tune columns, retaining input grid order on ties."""
    frame = frame.copy()
    for column, keys in [('selected_overall', ['method']), ('selected_within_fit', ['method', 'fit_on'])]:
        frame[column] = False
        for _, group in frame.groupby(keys, sort=False, dropna=False):
            if operating:
                for _, budget_group in group.groupby('budget_fp_per_hour', sort=False):
                    winner = budget_group.sort_values(['tune_tp', 'tune_fp'], ascending=[False, True], kind='stable').index[0]
                    frame.loc[winner, column] = True
            else:
                frame.loc[group.tune_f1.idxmax(), column] = True
    return frame


def read_comparison(base, cfg):
    seeds = pd.read_csv(base / 'seed_study.csv')
    expected = {str(s) for s in cfg['robustness']['seeds']}
    for mode in TUNE_MODES:
        for method in SAME_TRAIN + ['pca', 'iforest', 'rule_unseen', 'rule_completion']:
            found = seeds.loc[(seeds.tune_mode == mode) & (seeds.method == method), 'seed'].astype(str)
            if len(found) != len(expected) or set(found) != expected:
                raise ValueError(f'{base}: missing or duplicate seeds for {mode}/{method}')
    comparison = pd.read_csv(base / 'ensemble_summary.csv')
    if comparison.duplicated(['tune_mode', 'method', 'metric']).any():
        raise ValueError(f'{base}: duplicate comparison rows')
    for mode in TUNE_MODES:
        found = set(comparison.loc[comparison.tune_mode == mode, 'method'])
        if not set(SAME_TRAIN + ALL_FITS) <= found:
            raise ValueError(f'{base}: missing ensemble methods in {mode}')
    rows = []
    for (mode, method), group in comparison.groupby(['tune_mode', 'method'], sort=False):
        index = group.set_index('metric')
        if not set(METRICS) <= set(index.index):
            raise ValueError(f'{base}: missing metrics for {mode}/{method}')
        seed_rows = seeds[(seeds.tune_mode == mode) & (seeds.method == method)]
        for metric in METRICS:
            if not np.isclose(index.loc[metric, 'seed_mean'], seed_rows[metric].mean(), equal_nan=True):
                raise ValueError(f'{base}: stale seed mean for {mode}/{method}/{metric}')
        for source, column in [('ensemble', 'ensemble'), ('5 seeds', 'seed_mean')]:
            row = {'tune_mode': mode, 'method': method, 'source': source}
            for metric in METRICS:
                row[metric] = index.loc[metric, column]
                row[f'{metric}_std'] = index.loc[metric, 'seed_std'] if source == '5 seeds' else float('nan')
            rows.append(row)
    return pd.DataFrame(rows)


def markdown_table(frame, columns):
    def formatted(value):
        if pd.isna(value):
            return '—'
        if isinstance(value, (int, float)):
            return f'{value:.4f}'
        return str(value).replace('|', '/')
    return '\n'.join(['| ' + ' | '.join(columns) + ' |',
                      '| ' + ' | '.join(['---']*len(columns)) + ' |',
                      *['| ' + ' | '.join(formatted(row[c]) for c in columns) + ' |' for _, row in frame.iterrows()]])


def generate(cfg):
    paths = cfg['paths']
    experiments = {'chronological': paths['results_dir'],
                   'random': load_config(paths['random_config'])['paths']['results_dir']}
    experiments.update({name: paths['coverage_results_dir'] / name
                        for name in ('first_sessions', 'first_2h', 'first_fraction')})
    out = ensure_dir(paths['report_tables_dir'])
    main_frames, op_frames, appendix, operating_appendix, operating_runs, sources = [], [], [], [], [], []
    for experiment, base in experiments.items():
        main_frames.append(read_comparison(base, cfg).assign(experiment=experiment))
        op = pd.read_csv(base / 'operating_points_summary.csv')
        for mode in TUNE_MODES:
            for source, count in [('1 model', len(cfg['robustness']['seeds'])), ('ensemble', 1)]:
                subset = op[(op.tune_mode == mode) & (op.source == source)]
                # Three detectors, two globally selected baselines, four fit-specific baselines.
                if len(subset) != len(cfg['operating']['budgets_fp_per_hour']) * 9:
                    raise ValueError(f'{base}: incomplete operating table {mode}/{source}')
                if not (subset.runs == count).all():
                    raise ValueError(f'{base}: incomplete operating seed count')
        op_frames.append(op.assign(experiment=experiment))
        for seed in cfg['robustness']['seeds']:
            for mode in TUNE_MODES:
                source = base / 'seeds' / f'seed{seed}' / mode / f'model_selection_{mode}_grid.csv'
                appendix.append(flag_selected_variants(pd.read_csv(source)).assign(experiment=experiment, seed=seed, tune_mode=mode))
                op_source = source.with_name(f'operating_points_{mode}_grid.csv')
                operating_appendix.append(flag_selected_variants(pd.read_csv(op_source), operating=True).assign(
                    experiment=experiment, seed=seed, tune_mode=mode))
                sources.extend([str(source), str(op_source)])
                choice_source = source.with_name(f'operating_points_{mode}.csv')
                operating_runs.append(pd.read_csv(choice_source).assign(experiment=experiment, seed=seed, tune_mode=mode))
                sources.append(str(choice_source))
        for mode in TUNE_MODES:
            source = base / 'ensemble' / mode / f'model_selection_{mode}_grid.csv'
            appendix.append(flag_selected_variants(pd.read_csv(source)).assign(experiment=experiment, seed='ensemble', tune_mode=mode))
            op_source = source.with_name(f'operating_points_{mode}_grid.csv')
            operating_appendix.append(flag_selected_variants(pd.read_csv(op_source), operating=True).assign(
                experiment=experiment, seed='ensemble', tune_mode=mode))
            sources.extend([str(source), str(op_source)])
            choice_source = source.with_name(f'operating_points_{mode}.csv')
            operating_runs.append(pd.read_csv(choice_source).assign(experiment=experiment, seed='ensemble', tune_mode=mode))
            sources.append(str(choice_source))
        sources.extend(str(base / name) for name in ('seed_study.csv', 'ensemble_summary.csv', 'operating_points_summary.csv'))
    main = pd.concat(main_frames, ignore_index=True)
    operating = pd.concat(op_frames, ignore_index=True)
    main.to_csv(out / 'f1_all_selected.csv', index=False)
    main.loc[main.method.isin(SAME_TRAIN)].to_csv(out / 'f1_same_training.csv', index=False)
    operating.to_csv(out / 'operating_all_selected.csv', index=False)
    pd.concat(appendix, ignore_index=True).to_csv(out / 'appendix_all_variants.csv', index=False)
    pd.concat(operating_appendix, ignore_index=True).to_csv(out / 'appendix_operating_variants.csv', index=False)
    selected_operating = pd.concat(operating_runs, ignore_index=True)
    selected_operating['hybrid_disabled_on_tune'] = (selected_operating.method == 'hybrid') & (
        selected_operating.params.isna() | selected_operating.params.eq('None'))
    selected_operating.to_csv(out / 'operating_selected_runs.csv', index=False)
    text = ['# Generated report tables', '',
            'Selection rule: F1 tables use the variant and threshold with highest tune F1; ties keep grid order. '
            'Operating tables maximize tune TP under the requested FP/hour budget; ties minimize tune FP, then keep grid order. '
            'Eval metrics never choose a row. All configured seeds and both tuning modes are mandatory.', '',
            '`pca` / `iforest` select across all configured fit sets. `/train_normal` restricts selection to the same '
            'Normal cohort used by DeepLog (including validation); `/unlabeled_tune` uses additional unlabeled tune data. '
            'They are distinct data-access conditions. The primary fair comparison is f1_same_training.csv.', '',
            '`source=ensemble` means the DeepLog component averages all five models. Rules/PCA remain unchanged; '
            'Isolation Forest in that run uses the configured default seed, not an ensemble of forests.', '',
            'Unseen/known groups are defined by event vocabulary in gradient-training sessions only. '
            'Recall on known events does not isolate event order: counts, lengths and completion also differ. '
            'An empty anomaly group is reported as missing, not zero.', '',
            'Time coverage is measured by session start. Full sequences can end after nominal training cutoffs; '
            'this study is offline and does not establish leakage-free online deployment. FP/hour follows the '
            'existing proportional exposure convention (stream span × eval session share).', '']
    text += ['All final coverage cohorts reserve their latest selected sessions for validation. '
             'Earlier pilots with random validation were archived under *_pilot_random_validation and excluded '
             'because they changed a second design factor.', '']
    for experiment in experiments:
        for mode in TUNE_MODES:
            frame = main[(main.experiment == experiment) & (main.tune_mode == mode) & main.method.isin(SAME_TRAIN)]
            counts = frame.iloc[0]
            text += [f'## {experiment} / {mode}', '',
                     f"Eval anomalies: {int(counts.anomalies_unseen)} unseen-event sessions; {int(counts.anomalies_known)} known-event sessions.", '',
                     markdown_table(frame, ['method', 'source', 'f1', 'f1_std', 'recall_unseen', 'recall_known', 'fp_per_hour']), '']
    (out / 'tables.md').write_text('\n'.join(text), encoding='utf-8')
    findings = ['# Kết quả kiểm tra độ tin cậy', '',
                'Các số dưới đây được sinh bằng quy tắc cố định từ toàn bộ CSV, không chọn tay biến thể.', '',
                'Cả ba cấu hình dùng validation ở phần muộn nhất. Pilot dùng validation ngẫu nhiên đã lưu trong archive '
                'và loại khỏi bảng vì thay đổi thêm một yếu tố ngoài độ phủ thời gian.', '',
                '## Độ phủ thời gian — ensemble, tune sớm nhất, cùng dữ liệu fit', '']
    sensitivity_names = ['first_sessions', 'first_2h', 'first_fraction']
    subset = main[(main.experiment.isin(sensitivity_names)) & (main.tune_mode == 'earliest')
                  & (main.source == 'ensemble') & main.method.isin(SAME_TRAIN)]
    for experiment, group in subset.groupby('experiment', sort=False):
        best = group.f1.max()
        winners = ', '.join(group.loc[np.isclose(group.f1, best, rtol=0, atol=1e-12), 'method'])
        findings += [f'- {experiment}: F1 cao nhất trong so sánh cùng dữ liệu là {winners} ({best:.4f}).']
    hybrid_always_best = all(group.loc[group.method == 'hybrid', 'f1'].iloc[0] >= group.f1.max() - 1e-12
                             for _, group in subset.groupby('experiment'))
    findings += ['', ('Hybrid đứng đầu về F1 ensemble ở cả ba cấu hình này; kết quả chỉ áp dụng cho giao thức offline đã đo.'
                       if hybrid_always_best else
                       'Hybrid không đứng đầu ở tất cả cấu hình khi tune sớm nhất. Kết luận “hybrid luôn tốt nhất” không đứng vững '
                       'trong phép kiểm tra độ phủ này; phải trình bày kết quả theo từng cohort và cách tune, không sửa tham số theo eval.')]
    findings += ['', 'Thứ hạng F1 này không thay thế yêu cầu kiểm tra ngân sách FP/giờ thực tế.', '']
    findings += [markdown_table(subset, ['experiment', 'method', 'f1', 'recall_unseen', 'recall_known', 'fp_per_hour']), '',
                 '## Điểm vận hành — ensemble, tune sớm nhất', '']
    subset_op = operating[(operating.experiment.isin(sensitivity_names)) & (operating.tune_mode == 'earliest')
                          & (operating.source == 'ensemble') & operating.method.isin(SAME_TRAIN)]
    disabled = selected_operating[(selected_operating.experiment.isin(sensitivity_names))
                                   & (selected_operating.tune_mode == 'earliest')
                                   & (selected_operating.seed.astype(str) == 'ensemble')
                                   & selected_operating.hybrid_disabled_on_tune]
    if len(disabled):
        findings += ['Hybrid phải tắt toàn bộ cảnh báo khi không có tổ hợp luật + DeepLog nào thoả ngân sách trên tune. '
                     'Không được diễn giải FP/giờ bằng 0 trong các dòng này như một kết quả phát hiện tốt. '
                     'Thiết kế hiện tại cho phép tắt DeepLog nhưng không tắt riêng tầng luật; không sửa thiết kế sau khi xem eval.', '']
        for experiment, group in disabled.groupby('experiment', sort=False):
            budgets = ', '.join(str(int(b)) for b in group.budget_fp_per_hour)
            findings += [f'- {experiment}: hybrid bị tắt ở ngân sách {budgets} FP/giờ trên tune.']
        findings += ['']
    findings += [markdown_table(subset_op, ['experiment', 'budget_fp_per_hour', 'method', 'recall_mean',
                                           'recall_unseen_mean', 'recall_known_mean', 'fp_per_hour_mean', 'within_budget']), '',
                 '## Đính chính so sánh PCA', '',
                 'PCA/train_normal và PCA/unlabeled_tune dùng hai điều kiện dữ liệu khác nhau. '
                 'Không được thay số của train_normal bằng số cao hơn của unlabeled_tune rồi gọi đó là cùng điều kiện fit. '
                 'Các biến thể trong từng điều kiện vẫn phải được chọn trên tune. Bảng dưới là chia chronological, tune random, lượt ensemble.', '']
    pca_scope = operating[(operating.experiment == 'chronological') & (operating.tune_mode == 'random')
                          & (operating.source == 'ensemble') & operating.method.isin(['pca', 'pca/train_normal', 'pca/unlabeled_tune'])]
    findings += [markdown_table(pca_scope, ['budget_fp_per_hour', 'method', 'recall_mean', 'fp_per_hour_mean', 'within_budget']), '',
                 '## Cách diễn giải', '',
                 'So sánh đầy đủ hai cách chia × hai cách tune và 5 seed có trong tables.md và các CSV đi kèm. '
                 'Ngân sách là ràng buộc trên tune; phải đọc FP/giờ eval cùng recall, đặc biệt khi tune sớm nhất. '
                 'Các nhóm có/không có sự kiện lạ được định nghĩa lại theo từng tập train, nên thành phần nhóm có thể đổi giữa cấu hình. '
                 'Không gọi nhóm known là bằng chứng riêng về học thứ tự.', '',
                 'Khoảng 9 phút chỉ là thời điểm bắt đầu phiên. Sự kiện của phiên train còn kéo dài tới khoảng 14,37 giờ; '
                 'độ phủ thời gian này là thí nghiệm offline theo StartTime, chưa xác lập khả năng triển khai online không rò rỉ tương lai. '
                 'Ngưỡng tự hiệu chỉnh để tuần 5/phần hạn chế.', '']
    (out / 'findings.md').write_text('\n'.join(findings), encoding='utf-8')
    save_json({'meta': run_metadata(cfg), 'sources': sources, 'selection': 'tune_only',
               'source_sha256': {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in sources},
               'mandatory_modes': list(TUNE_MODES), 'mandatory_seeds': cfg['robustness']['seeds']}, out / 'manifest.json')
    print(f'Wrote {len(main)} F1 rows, {len(operating)} operating rows to {out}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/config.yaml')
    args = parser.parse_args()
    generate(load_config(args.config))


if __name__ == '__main__':
    main()
