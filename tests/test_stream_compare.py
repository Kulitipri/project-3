import types
import unittest
from collections import Counter

import numpy as np
import torch

from src.stream_compare import Stream, compute_f1, feedback, pair_triggers


def bare(seqs, times, train, window=3, tune=None):
    """Stream không đọc file: chỉ các thuộc tính mà hàm cần."""
    s = Stream.__new__(Stream)
    s.window, s.seqs, s.times, s.n, s.train = window, seqs, times, len(seqs), train
    s.y = np.array([0] * len(seqs))
    s.t0 = np.array([t[0] for t in times], float)
    s.t_end = np.array([t[-1] for t in times], float)
    s.stream_end = float(s.t_end.max())
    s.is_tune = np.zeros(len(seqs), bool) if tune is None else tune
    s.session_freq = Counter(e for q in train for e in set(q))
    counts = np.zeros(31)
    for q in train:
        np.add.at(counts, q, 1)
    s.unigram = (counts + 0.1) / (counts.sum() + 3.1)
    ids, block, stamp, pair, pos = {}, [], [], [], []
    for b, (q, t) in enumerate(zip(seqs, times)):
        for i in range(window, len(q)):
            block.append(b), stamp.append(t[i]), pos.append(i)
            pair.append(ids.setdefault(tuple(q[i - window:i + 1]), len(ids)))
    s.block, s.stamp, s.pair, s.pos = map(np.array, (block, stamp, pair, pos))
    s.pairs = torch.tensor(list(ids))
    s.first_row = np.full(s.n, -1)
    s.first_row[s.block[::-1]] = np.arange(len(block))[::-1]
    return s


class RareAndSurprisalTests(unittest.TestCase):
    def test_rare_times_uses_training_session_frequency_only(self):
        s = bare([[1, 2, 3, 9], [1, 2, 4], [1, 2, 3]], [np.array([0, 1, 2, 3]), np.array([5, 6, 7]), np.array([8, 9, 10])],
                 [[1, 2, 3]] * 5 + [[1, 2, 3, 4]])
        self.assertEqual(s.rare_times(0).tolist(), [3.0, np.inf, np.inf])
        self.assertEqual(s.rare_times(1).tolist(), [3.0, 7.0, np.inf])
        self.assertEqual(s.rare_times(1, low=1).tolist(), [np.inf, 7.0, np.inf])

    def test_mixing_lowers_surprise_of_frequent_events_more_than_rare_ones(self):
        s = bare([[1, 1, 1, 2], [1, 1, 1, 3]], [np.arange(4), np.arange(4)], [[1, 2, 2, 2, 2, 2]] * 10 + [[3]])
        p = np.array([1e-6, 1e-6])
        pure, mixed = s.surprisal(p, 0.0), s.surprisal(p, 0.2)
        self.assertAlmostEqual(pure[0], pure[1])
        self.assertLess(mixed[0], mixed[1])
        self.assertTrue((mixed < pure).all())

    def test_first_crossing_is_the_earliest_hot_event(self):
        s = bare([[1, 1, 1, 2, 3, 2]], [np.array([0, 1, 2, 3, 4, 5])], [[1]])
        hot = np.array([int(p[-1]) == 2 for p in s.pairs])
        self.assertEqual(s.first_crossing(hot).tolist(), [3.0])
        self.assertEqual(pair_triggers(s, hot), {0: [3, 5]})

    def test_markov_is_a_probability_and_prefers_seen_transitions(self):
        s = bare([[1, 2, 3, 1], [1, 2, 3, 5]], [np.arange(4), np.arange(4)], [[1, 2, 3, 1, 2, 3, 1]] * 20)
        p = s.markov(2)
        self.assertTrue(((p > 0) & (p < 1)).all())
        self.assertGreater(p[0], p[1])


class FeedbackTests(unittest.TestCase):
    def run_feedback(self, kind, labels, delay=1.0, flip=0.0, radius=0.1):
        n = len(labels)
        seqs = [[1, 2, 3, 6]] * n
        times = [np.array([100 * i, 100 * i + 1, 100 * i + 2, 100 * i + 3]) for i in range(n)]
        s = bare(seqs, times, [[1, 2, 3]] * 5)
        s.y = np.array(labels)
        hidden = np.ones((len(s.pairs), 4))
        inf = np.full(n, np.inf)
        return feedback(s, {b: [3] for b in range(n)}, inf, inf, hidden, kind, radius, 0, 0, delay, flip, 0)

    def test_every_kind_suppresses_repeats_after_a_normal_verdict(self):
        for kind in ("exact", "count", "lstm"):
            alert, asked = self.run_feedback(kind, [0, 0, 0, 0])
            self.assertEqual(asked, 1)
            self.assertTrue(np.isfinite(alert[0]))
            self.assertTrue(np.isinf(alert[1:]).all())

    def test_confirmed_anomalies_are_never_suppressed(self):
        for kind in ("exact", "count"):
            alert, asked = self.run_feedback(kind, [1, 1, 1])
            self.assertEqual(asked, 3)

    def test_late_labels_suppress_nothing(self):
        alert, asked = self.run_feedback("exact", [0, 0, 0], delay=10_000.0)
        self.assertEqual(asked, 3)

    def test_compute_f1(self):
        y = np.array([1, 0, 1, 0])
        self.assertAlmostEqual(compute_f1(np.array([1.0, np.inf, np.inf, 2.0]), y, np.ones(4, bool)), 0.5)


if __name__ == "__main__":
    unittest.main()
