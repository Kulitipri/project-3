import unittest

import numpy as np
import pandas as pd
import torch

from src.hybrid_detect import first_alert_time, time_to_kth
from src.stream_detector import COMPLETION, DEEPLOG, UNKNOWN_EVENT, UNSEEN, RankScorer, StreamDetector
from src.stream_feedback import decisions, simulate
from src.stream_offline import completion_times, f1_differs, keep_streamable, shortest_equivalent

NUM_KEYS, W = 8, 3


class FixedModel(torch.nn.Module):
    """Điểm phụ thuộc sự kiện cuối của cửa sổ: sự kiện (cuối + 1) được chờ đợi nhất, rồi giảm dần."""
    calls = 0

    def forward(self, x):
        self.calls += 1
        keys = torch.arange(NUM_KEYS).unsqueeze(0)
        return -((keys - x[:, -1:] - 1) % NUM_KEYS).float()


def make(deadline=10.0, top_g=2, cache_size=100, seen=frozenset(range(1, 7))):
    return StreamDetector(RankScorer(FixedModel(), cache_size), set(seen), replica_event=1, replica_count=2,
                          deadline_sec=deadline, top_g=top_g, window_size=W)


def run(detector, rows, end=None):
    """rows: list (time, block, event) đã sắp theo thời gian."""
    alerts = []
    for t in dict.fromkeys(r[0] for r in rows):
        alerts += detector.process(t, [(b, e) for s, b, e in rows if s == t])
    return alerts + detector.finish(rows[-1][0] if end is None else end)


class StreamDetectorTests(unittest.TestCase):
    def test_deeplog_only_scores_with_a_full_real_window(self):
        rows = [(0, "a", 1), (1, "a", 5), (2, "a", 1)]
        self.assertEqual([a for a in run(make(), rows) if a.component == DEEPLOG], [])
        alerts = [a for a in run(make(), rows + [(3, "a", 6)]) if a.component == DEEPLOG]
        self.assertEqual([(a.time, a.event, a.rank) for a in alerts], [(3, 6, 5)])
        self.assertEqual(alerts[0].expected, (2, 3, 4))

    def test_prefix_decisions_do_not_depend_on_the_future(self):
        rng = np.random.default_rng(0)
        rows = [(int(t), f"b{rng.integers(5)}", int(rng.integers(1, 8))) for t in np.sort(rng.integers(0, 60, 200))]
        full = run(make(), rows)
        for cut in (10, 25, 40):
            prefix = [r for r in rows if r[0] <= cut]
            known = [a for a in run(make(), prefix, end=cut) if a.time < cut]
            self.assertEqual(known, [a for a in full if a.time < cut])

    def test_deadline_boundaries(self):
        on_time = [(0, "a", 1), (10, "a", 1), (30, "z", 1), (31, "z", 1)]
        self.assertEqual([a.block for a in run(make(), on_time) if a.component == COMPLETION], [])
        late = [(0, "a", 1), (11, "a", 1), (30, "z", 1), (31, "z", 1)]
        self.assertEqual([(a.block, a.time, a.replicas) for a in run(make(), late) if a.component == COMPLETION],
                         [("a", 10, 1)])
        self.assertEqual([a.time for a in run(make(), [(0, "a", 1)], end=10)], [10])
        self.assertEqual(run(make(), [(0, "a", 1)], end=9), [])

    def test_unknown_line_is_an_unseen_event(self):
        rows = [(0, "a", 1), (1, "a", 1), (2, "a", 2), (3, "a", UNKNOWN_EVENT), (4, "a", 1)]
        alerts = run(make(), rows)
        self.assertEqual([(a.component, a.first) for a in alerts if a.time == 3], [(UNSEEN, True), (DEEPLOG, False)])

    def test_each_component_reports_a_block_once(self):
        rows = [(t, "a", 7) for t in range(6)]
        alerts = run(make(), rows)
        self.assertEqual(sum(a.component == UNSEEN for a in alerts), 1)
        self.assertEqual(sum(a.first for a in alerts), 1)

    def test_cache_eviction_keeps_results_correct(self):
        windows = [(a, b, c) for a in range(1, 4) for b in range(1, 4) for c in range(1, 4)] * 2
        small, large = RankScorer(FixedModel(), 2), RankScorer(FixedModel(), 1000)
        for i in range(0, len(windows), 5):
            for x, y in zip(small.rank_tables(windows[i:i + 5]), large.rank_tables(windows[i:i + 5])):
                np.testing.assert_array_equal(x, y)
        self.assertLessEqual(len(small.cache), 2)
        self.assertEqual(large.misses, 27)

    def test_stream_equals_offline_on_random_sessions(self):
        rng = np.random.default_rng(1)
        model = FixedModel()
        for _ in range(20):
            sessions = {}
            for b in range(12):
                n = int(rng.integers(1, 9))
                sessions[b] = (list(map(int, rng.integers(1, 8, n))), np.sort(rng.integers(0, 40, n)))
            rows = sorted(((int(t), b, e) for b, (s, ts) in sessions.items() for e, t in zip(s, ts)), key=lambda r: r[0])
            end = max(r[0] for r in rows)
            got = {(a.block, a.component): a.time for a in run(make(), rows, end=end)}
            for b, (seq, times) in sessions.items():
                unseen = next((float(t) for e, t in zip(seq, times) if e == 7), np.inf)
                tk = time_to_kth(seq, times, 1, 2)
                comp = completion_times(np.array([times[0]]), np.array([tk]), 10.0, end)[0]
                mask, pos = [], []
                for i in range(W, len(seq)):
                    logits = model(torch.tensor([seq[i - W:i]]))[0]
                    mask.append(int((logits > logits[seq[i]]).sum()) + 1 > 2)
                    pos.append(i)
                deep = first_alert_time(times, np.array(pos, dtype=int), np.array(mask, dtype=bool))
                self.assertEqual((got.get((b, UNSEEN), np.inf), got.get((b, COMPLETION), np.inf),
                                  got.get((b, DEEPLOG), np.inf)), (unseen, comp, deep))


class FeedbackTests(unittest.TestCase):
    def build(self, k, wait):
        det = make(deadline=50.0)
        det.use_feedback, det.confirm_events, det.confirm_seconds = True, k, wait
        return det

    def stream(self, det, sessions, labels, delay):
        rows = sorted(((int(t), b, e) for b, (s, ts) in sessions.items() for e, t in zip(s, ts)), key=lambda r: r[0])
        end, verdicts, got = max(r[0] for r in rows), [], {}

        def collect(alerts):
            for a in alerts:
                if a.component == DEEPLOG:
                    got[a.block] = a.time
                    verdicts.append((max(a.time, int(sessions[a.block][1][-1])) + delay, a.pattern, labels[a.block]))

        def deliver(now):
            while True:
                due = sorted([v for v in verdicts if v[0] <= now], key=lambda v: v[0])
                if not due:
                    return
                collect(det.advance(due[0][0]))
                verdicts.remove(due[0])
                det.feedback(due[0][1], due[0][2])

        for t in dict.fromkeys(r[0] for r in rows):
            deliver(t)
            collect(det.process(t, [(b, e) for s, b, e in rows if s == t]))
        deliver(end)
        collect(det.finish(end))
        return got, end

    def test_stream_feedback_equals_offline_simulation(self):
        rng = np.random.default_rng(2)
        for trial in range(60):
            k, wait = [(0, 0.0), (2, 5.0), (3, 100.0)][trial % 3]
            sessions, labels = {}, {}
            for b in range(14):
                n = int(rng.integers(1, 12))
                sessions[b] = (list(map(int, rng.choice([1, 2, 3, 5, 6], n))), np.sort(rng.integers(0, 60, n)))
                labels[b] = int(rng.integers(0, 2))
            got, end = self.stream(self.build(k, wait), sessions, labels, delay=3)
            seqs, times = [sessions[b][0] for b in sessions], [sessions[b][1] for b in sessions]
            model, trig = FixedModel(), []
            for s in seqs:
                hits = [i for i in range(W, len(s)) if int((lambda l: (l > l[s[i]]).sum())(model(torch.tensor([s[i - W:i]]))[0])) + 1 > 2]
                trig.append(np.array(hits, dtype=int))
            unseen = np.full(len(seqs), np.inf)
            tk = np.array([time_to_kth(s, t, 1, 2) for s, t in zip(seqs, times)])
            comp = completion_times(np.array([t[0] for t in times], float), tk, 50.0, end)
            off = simulate(decisions(seqs, times, trig, k, wait), np.array([labels[b] for b in sessions]),
                           np.array([t[-1] for t in times], float), unseen, comp, end, 3.0)
            self.assertEqual([got.get(b, np.inf) for b in sessions], off.tolist())

    def test_confirmed_normal_pattern_is_suppressed_only_after_the_label_arrives(self):
        seq = [1, 2, 3, 6]
        mk = lambda: ({"a": (seq, np.array([0, 1, 2, 3])), "b": (seq, np.array([4, 5, 6, 7])), "c": (seq, np.array([20, 21, 22, 23]))},
                      {"a": 0, "b": 0, "c": 0})
        got, _ = self.stream(self.build(0, 0.0), *mk(), delay=10)
        self.assertEqual(set(got), {"a", "b"})

    def test_waiting_decides_at_the_kth_event_or_the_timeout(self):
        late = {"a": ([1, 2, 3, 6, 1, 1], np.array([0, 1, 2, 3, 5, 40]))}
        self.assertEqual(self.stream(self.build(2, 10.0), late, {"a": 1}, 1)[0], {"a": 13.0})
        quick = {"a": ([1, 2, 3, 6, 1, 1], np.array([0, 1, 2, 3, 5, 9]))}
        self.assertEqual(self.stream(self.build(2, 10.0), quick, {"a": 1}, 1)[0], {"a": 9})


class DeadlineChoiceTests(unittest.TestCase):
    def test_equal_predictions_prefer_the_shortest_deadline(self):
        y = np.array([0, 1] * 500)
        good = y == 1
        self.assertEqual(shortest_equivalent({1: good, 2: good, 10: good}, 10, y, np.ones(1000, bool), 0), 1)
        bad = good & (np.arange(1000) < 500)
        self.assertEqual(shortest_equivalent({1: bad, 2: good}, 2, y, np.ones(1000, bool), 0), 2)
        self.assertFalse(f1_differs(y, good.astype(int), good.astype(int), 0))


class OfflineRuleTests(unittest.TestCase):
    def test_only_ranks_with_full_window_are_kept(self):
        ranks = pd.DataFrame({"BlockId": ["short", "long", "edge"],
                              "Ranks": [np.array([9]), np.array([1, 2]), np.array([5])],
                              "TargetPos": [np.array([2]), np.array([3, 4]), np.array([3])]})
        out = keep_streamable(ranks, 3)
        self.assertEqual([r.tolist() for r in out["Ranks"]], [[], [1, 2], [5]])
        self.assertEqual(ranks["Ranks"][0].tolist(), [9])


if __name__ == "__main__":
    unittest.main()
