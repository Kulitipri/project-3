import unittest
from pathlib import Path
import tempfile
import json

import numpy as np
import pandas as pd

from src.evaluate import recall_by_unseen, unseen_event_mask
from src.model_selection import choose_and_evaluate, make_features, run_config
from src.baselines import iforest_scores
from src.operating_point import threshold_under_budget, evaluate_budgets
from src.time_coverage import make_cohorts
from src.report_tables import read_comparison, flag_selected_variants, SAME_TRAIN, ALL_FITS, METRICS
from src.audit_results import audit
from src.config import PROJECT_ROOT


class ExperimentIntegrityTests(unittest.TestCase):
    def test_audit_preserves_valid_companions_and_archives_unknown_files(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as directory:
            base = Path(directory)
            results = base / 'results'
            results.mkdir()
            config = {'seed': 42, 'split': {'strategy': 'random'}, 'robustness': {'seeds': [42]},
                      'paths': {'results_dir': results, 'archive_dir': base / 'archive'}}
            meta = {'config': {key: value for key, value in config.items() if key != 'paths'}}
            (results / 'hybrid_detection.json').write_text(json.dumps({'meta': meta}))
            (results / 'hybrid_drift.csv').write_text('value\n1\n')
            (results / 'unknown.json').write_text('{}')
            before = (results / 'hybrid_drift.csv').read_bytes()
            rows = audit(config, apply=True)
            self.assertEqual(sum(row['action'] == 'archive' for row in rows), 1)
            self.assertEqual((results / 'hybrid_drift.csv').read_bytes(), before)
            self.assertFalse((results / 'unknown.json').exists())
            archived = list((base / 'archive').glob('*/unknown.json'))
            self.assertEqual(len(archived), 1)
            self.assertEqual(archived[0].read_text(), '{}')

    def test_report_variant_flags_use_only_tune_and_stable_ties(self):
        grid = pd.DataFrame({'method': ['pca']*3, 'fit_on': ['train_normal']*3,
                             'tune_f1': [.8, .8, .7], 'f1': [.2, .9, 1.]})
        selected = flag_selected_variants(grid)
        self.assertEqual(selected.index[selected.selected_overall].tolist(), [0])
        grid['budget_fp_per_hour'] = 5
        grid['tune_tp'] = [3, 3, 2]
        grid['tune_fp'] = [2, 1, 0]
        self.assertEqual(flag_selected_variants(grid, operating=True).index[
            flag_selected_variants(grid, operating=True).selected_overall].tolist(), [1])

    def test_report_rejects_missing_seeds_and_stale_aggregates(self):
        cfg = {'robustness': {'seeds': [42, 1, 2, 3, 4]}}
        methods = set(SAME_TRAIN + ALL_FITS)
        rows = [{'seed': seed, 'method': method, 'tune_mode': mode, **{metric: .5 for metric in METRICS}}
                for seed in cfg['robustness']['seeds'] for method in methods for mode in ['earliest', 'random']]
        comparisons = [{'tune_mode': mode, 'method': method, 'metric': metric, 'ensemble': .6,
                        'seed_mean': .5, 'seed_std': 0}
                       for mode in ['earliest', 'random'] for method in methods for metric in METRICS]
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            seeds = pd.DataFrame(rows)
            comp = pd.DataFrame(comparisons)
            seeds.to_csv(base / 'seed_study.csv', index=False)
            comp.to_csv(base / 'ensemble_summary.csv', index=False)
            self.assertEqual(len(read_comparison(base, cfg)), 4*len(methods))
            seeds.iloc[1:].to_csv(base / 'seed_study.csv', index=False)
            with self.assertRaisesRegex(ValueError, 'seeds'):
                read_comparison(base, cfg)
            seeds.to_csv(base / 'seed_study.csv', index=False)
            comp.loc[0, 'seed_mean'] = .9
            comp.to_csv(base / 'ensemble_summary.csv', index=False)
            with self.assertRaisesRegex(ValueError, 'stale'):
                read_comparison(base, cfg)

    def test_count_deduplication_preserves_iforest_scores(self):
        rng = np.random.default_rng(7)
        X = {'train_normal': rng.integers(0, 4, (30, 5)).astype(float),
             'tune': np.tile(rng.integers(0, 4, (10, 5)), (5, 1)).astype(float),
             'eval': np.tile(rng.integers(0, 4, (8, 5)), (9, 1)).astype(float)}
        for features in ['raw', 'presence', 'log1p']:
            for fit_on in ['train_normal', 'unlabeled_tune']:
                params = {'features': features, 'fit_on': fit_on, 'param': 10}
                fit = X['train_normal'] if fit_on == 'train_normal' else X['tune']
                F_fit, other = make_features(features, fit, [X['tune'], X['eval']])
                _, expected = iforest_scores(F_fit, other, 10, 42)
                actual = run_config('iforest', params, X, 42)
                for first, second in zip(actual, expected):
                    np.testing.assert_array_equal(first, second)

    def test_unseen_reference_uses_only_train_and_keeps_groups_exhaustive(self):
        mask = unseen_event_mask([[1, 2], [1, 3], [3]], [[1, 2]])
        metrics = recall_by_unseen(np.array([1, 1, 0]), np.array([0, 1, 1]), mask)
        self.assertEqual(metrics['anomalies_known'], 1)
        self.assertEqual(metrics['anomalies_unseen'], 1)
        self.assertEqual(metrics['recall_known'], 0)
        self.assertEqual(metrics['recall_unseen'], 1)
        self.assertTrue(np.isnan(recall_by_unseen(np.array([1]), np.array([1]), np.array([True]))['recall_known']))

    def test_f1_threshold_does_not_use_eval_labels(self):
        y = {'tune': np.array([0, 1, 1]), 'eval': np.array([0, 1]), 'eval_unseen': np.array([True, False])}
        first = choose_and_evaluate({}, np.array([0., 1., 2.]), np.array([.5, 1.5]), y)
        second = choose_and_evaluate({}, np.array([0., 1., 2.]), np.array([.5, 1.5]), {**y, 'eval': 1-y['eval']})
        self.assertEqual(first['threshold'], second['threshold'])
        self.assertEqual(first['tune_f1'], second['tune_f1'])

    def test_budget_does_not_split_tied_scores(self):
        self.assertTrue(np.isinf(threshold_under_budget(np.array([0, 1]), np.array([2., 2.]), 0)[0]))
        threshold, tp, fp = threshold_under_budget(np.array([1, 0, 1]), np.array([3., 2., 1.]), 0)
        self.assertEqual((threshold, tp, fp), (3., 1, 0))

    def test_baseline_variants_are_selected_on_tune_under_budget(self):
        cfg = {'operating': {'budgets_fp_per_hour': [0]}, 'select': {'fit_on': ['train_normal', 'unlabeled_tune']}}
        data = {'y': np.array([0, 1, 0, 1]), 'is_tune': np.array([True, True, False, False]),
                'max_rank': np.array([1, 2, 1, 2]), 'rules': {1: np.array([False, True, False, True])},
                'g_values': [1, 2], 'tune_hours': 1, 'eval_hours': 1, 'unseen': np.array([False, True, False, True])}
        baselines = {'pca': [({'fit_on': 'train_normal'}, np.array([0., 1.]), np.array([1., 0.])),
                             ({'fit_on': 'unlabeled_tune'}, np.array([1., 1.]), np.array([0., 1.]))]}
        table = evaluate_budgets(cfg, data, baselines)
        selected = table[table.method == 'pca'].iloc[0]
        self.assertIn('train_normal', selected.params)
        self.assertEqual(selected.recall, 0)
        self.assertEqual(len(table.attrs['baseline_variants']), 2)

    def test_coverage_preserves_sizes_and_identical_test_without_overlap(self):
        origin = pd.Timestamp('2000-01-01')
        sessions = pd.DataFrame({'BlockId': list('abcdefghij'), 'Label': [0]*9+[1],
                                 'StartTime': [origin+pd.Timedelta(seconds=i*600) for i in range(10)],
                                 'EventSequence': [[1, 2]]*10})
        context = pd.DataFrame({'BlockId': sessions.BlockId, 'EventTimes': [np.array([i*600, i*600+1000]) for i in range(10)]})
        original = {'train': sessions.iloc[:2], 'val': sessions.iloc[2:3]}
        cohorts, descriptions = make_cohorts(sessions, original, context, {'hours': .6, 'time_fraction': .5}, 42)
        reference = cohorts['first_sessions']['test']
        for splits in cohorts.values():
            self.assertEqual(len(splits['train']), 2)
            self.assertEqual(len(splits['val']), 1)
            pd.testing.assert_frame_equal(splits['test'], reference)
            self.assertFalse(set(splits['train'].BlockId) & set(reference.BlockId))
            self.assertLessEqual(splits['train'].StartTime.max(), splits['val'].StartTime.min())
        self.assertGreater(descriptions['first_sessions']['last_training_event_hours'], 0)


if __name__ == '__main__':
    unittest.main()
