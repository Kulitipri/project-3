"""
BƯỚC 14 — Chờ xác nhận + phản hồi của người vận hành (giảm báo nhầm của DeepLog theo luồng).

Chẩn đoán (bước 12): DeepLog báo nhầm nhiều vì (1) tại lúc trượt top-g, nhiều phiên bất thường và bình thường có ĐÚNG
cùng 10 sự kiện, thêm ~10 sự kiện mới tách được; (2) hành vi bình thường đổi theo thời gian, dữ liệu train ở đầu log
không bao phủ (sao chép lại block — E9 — chỉ có trong 1/5.582 session train).

Cách làm: khi DeepLog trượt, chờ thêm k sự kiện của block (tối đa H giây) rồi quyết định theo MẪU (số đếm sự kiện của
block tới lúc đó). Người vận hành xác nhận nhãn của mỗi block DeepLog báo; mẫu được xác nhận phần lớn bình thường thì không
báo nữa. Nhãn chỉ về SAU KHI PHIÊN KẾT THÚC + delay, và chỉ có cho block đã bị báo. Không cần nhãn nào khác.

k, H chọn trên tập tune: mẫu ít lẫn lộn nhất (số block phải bỏ nhầm ít nhất), hoà thì k, H nhỏ hơn. Không dùng tập strict.

Output: outputs/results/stream/feedback.json, feedback_table.csv
Chạy (~3 phút, sau bước 12):
    python -m src.stream_feedback --config configs/config.yaml
"""
import argparse
import heapq
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from src.config import load_config
from src.hybrid_detect import ops_metrics, select_on_tune, time_to_kth
from src.model_selection import load_test_with_tune_mask
from src.seed_study import TUNE_MODES
from src.stream_demo import build_detector, by_second, replay_order
from src.stream_detector import DEEPLOG
from src.stream_offline import completion_times, keep_streamable
from src.utils import get_logger, load_json, run_metadata, save_json

logger = get_logger(__name__)


def decisions(sequences, times, triggers: list[np.ndarray], k: int, max_wait: float) -> list[list[tuple[float, tuple]]]:
    """Mỗi lần trượt (vị trí i) → (thời điểm quyết định, mẫu): sau k sự kiện nữa, hoặc sau max_wait giây nếu chưa đủ."""
    out = []
    for seq, tm, positions in zip(sequences, times, triggers):
        rows = []
        for i in positions:
            last = int(np.searchsorted(tm, tm[i] + max_wait, side="right")) - 1
            last = min(last, i + k) if k else i
            at = float(tm[last]) if last == i + k else float(tm[i] + max_wait)
            rows.append((at, tuple(sorted(Counter(seq[:last + 1]).items()))))
        out.append(rows)
    return out


def simulate(dec: list, y: np.ndarray, t_end: np.ndarray, unseen: np.ndarray, completion: np.ndarray, stream_end: float,
             delay: float | None) -> np.ndarray:
    """Thời điểm DeepLog cảnh báo từng block (inf nếu không); delay None = không có phản hồi."""
    queue = [(at, b, j) for b, rows in enumerate(dec) for j, (at, _) in enumerate(rows) if at <= stream_end]
    queue.sort()
    alert = np.full(len(y), np.inf)
    memory, pending = defaultdict(lambda: [0, 0]), []
    for at, b, j in queue:
        while pending and pending[0][0] <= at:
            _, pattern, label = heapq.heappop(pending)
            memory[pattern][label] += 1
        if np.isfinite(alert[b]) or unseen[b] <= at or completion[b] < at:
            continue
        pattern = dec[b][j][1]
        normal, anomalous = memory[pattern] if pattern in memory else (0, 0)
        if normal > anomalous:
            continue
        alert[b] = at
        if delay is not None:
            heapq.heappush(pending, (max(at, t_end[b]) + delay, pattern, int(y[b])))
    return alert


def impurity(dec: list, y: np.ndarray, rule: np.ndarray, is_tune: np.ndarray) -> tuple[int, int]:
    """(số block thuộc thiểu số trong mẫu của nó, tổng block) trên tập tune, xét lần trượt đầu của block luật chưa báo."""
    groups = defaultdict(lambda: [0, 0])
    for b in np.flatnonzero(is_tune):
        if dec[b] and rule[b] > dec[b][0][0]:
            groups[dec[b][0][1]][y[b]] += 1
    return sum(min(v) for v in groups.values()), sum(sum(v) for v in groups.values())


def verify_stream(cfg: dict, df: pd.DataFrame, y: np.ndarray, t_end: np.ndarray, k: int, wait: float, delay: float) -> np.ndarray:
    """Chạy bộ phát hiện theo luồng với người vận hành mô phỏng; trả về thời điểm DeepLog cảnh báo từng block."""
    detector, _ = build_detector(cfg)
    detector.use_feedback, detector.confirm_events, detector.confirm_seconds = True, k, wait
    alert, verdicts, order = np.full(len(df), np.inf), [], 0

    def collect(alerts):
        nonlocal order
        for a in alerts:
            if a.component == DEEPLOG:
                alert[a.block] = a.time
                order += 1
                heapq.heappush(verdicts, (max(a.time, t_end[a.block]) + delay, order, a.pattern, int(y[a.block])))

    def deliver(now):
        while verdicts and verdicts[0][0] <= now:
            collect(detector.advance(verdicts[0][0]))   # có thể sinh thêm phản hồi, nhưng luôn muộn hơn
            _, _, pattern, label = heapq.heappop(verdicts)
            detector.feedback(pattern, label)

    stream = replay_order(df["EventTimes"], df["EventSequence"])
    for second, batch in by_second(*stream):
        deliver(second)
        collect(detector.process(second, batch))
    end = float(stream[2][-1])
    deliver(end)
    collect(detector.finish(end))
    return alert


def main() -> None:
    parser = argparse.ArgumentParser(description="Bước 14: chờ xác nhận + phản hồi của người vận hành")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    paths, scfg, hcfg, sel = cfg["paths"], cfg["streaming"], cfg["hybrid"], cfg["select"]
    results_dir, out_dir = paths["results_dir"], paths["results_dir"] / "stream"
    window, mode = cfg["model"]["window_size"], scfg["tune_mode"]
    offline = load_json(out_dir / "offline.json")
    reference = pd.read_pickle(out_dir / "offline_alerts.pkl")
    test, is_tune, _ = load_test_with_tune_mask(paths["splits_dir"], sel, mode)
    df = test.merge(pd.read_pickle(paths["block_context"])[["BlockId", "EventTimes"]], on="BlockId", how="left")
    if not (df["BlockId"].to_numpy() == reference["BlockId"].to_numpy()).all():
        raise ValueError("offline_alerts.pkl không cùng tập block với test.pkl hiện tại — chạy lại bước 12")
    y = df["Label"].to_numpy()
    t0 = df["EventTimes"].map(lambda t: t[0]).to_numpy(dtype=float)
    t_end = df["EventTimes"].map(lambda t: t[-1]).to_numpy(dtype=float)
    stream_end, deadline = float(t_end.max()), offline["profile"]["deadline_sec"]
    unseen = reference["part_unseen"].to_numpy()
    replica = load_json(paths["event_index"])[hcfg["replica_event"]]
    t_kth = np.array([time_to_kth(s, t, replica, offline["profile"]["replica_count"])
                      for s, t in zip(df["EventSequence"], df["EventTimes"])])
    cut = offline["strict_start_hour"] * 3600
    subsets = {"eval": (~is_tune, offline["subsets"][f"{mode}/eval"]["hours"])}
    if f"{mode}/strict" in offline["subsets"]:
        subsets["strict"] = (t0 > cut, offline["subsets"][f"{mode}/strict"]["hours"])

    subsets = {k: v for k, v in subsets.items() if v[0].any()}  # log ngắn: có thể không có tập con nào đó
    sources = {f"seed{s}": results_dir / "seeds" / f"seed{s}" / TUNE_MODES[0] for s in cfg["robustness"]["seeds"]}
    sources["ensemble"] = results_dir / "ensemble" / TUNE_MODES[0]
    grid = [(0, 0)] + [(k, h) for k in scfg["confirm_events"] for h in scfg["confirm_seconds"]]
    report = {"meta": run_metadata(cfg), "tune_mode": mode, "sources": {}}
    rows, chosen = [], {}

    def add(source, variant, delay, alert, asked=None):
        """asked: thời điểm cảnh báo của riêng thành phần cần người vận hành xác nhận (None nếu không dùng phản hồi)."""
        for subset, (mask, hours) in subsets.items():
            rows.append({"source": source, "variant": variant, "label_delay_sec": delay, "subset": subset,
                         **ops_metrics(y[mask], alert[mask], t0[mask], hours),
                         # số lần hỏi người vận hành trên TOÀN luồng (kể cả đoạn đầu), và trong một giờ bận nhất
                         "labels_asked_whole_stream": None if asked is None else int(np.isfinite(asked).sum()),
                         "labels_asked_busiest_hour": None if asked is None else int(
                             np.bincount((asked[np.isfinite(asked)] // 3600).astype(int), minlength=1).max())})

    def positions(df_ranks, top_g):
        return [p[r > top_g] for r, p in zip(df_ranks["Ranks"], df_ranks["TargetPos"])]

    for name, directory in sources.items():
        params = offline["results"][name][mode]["params"]["hybrid_stream"]
        completion = completion_times(t0, t_kth, deadline * params["deadline_margin"], stream_end)
        rule = np.minimum(unseen, completion)
        ranks = keep_streamable(pd.read_pickle(directory / "deeplog_ranks.pkl"), window)
        merged = df[["BlockId"]].merge(ranks, on="BlockId", how="left")
        trig = positions(merged, params["top_g"])
        prepared = {g: decisions(df["EventSequence"], df["EventTimes"], trig, *g) for g in grid}
        score = {g: impurity(prepared[g], y, rule, is_tune)[0] for g in grid}
        best = min(grid, key=lambda g: (score[g], g[0], g[1]))
        chosen[name] = {"confirm_events": best[0], "confirm_seconds": best[1], "params": params,
                        "tune_impurity": {f"k={k},H={h}": score[(k, h)] for k, h in grid},
                        "tune_alerts": impurity(prepared[best], y, rule, is_tune)[1]}
        add(name, "không phản hồi", None, np.minimum(rule, simulate(prepared[(0, 0)], y, t_end, unseen, completion, stream_end, None)))
        for delay in scfg["label_delays_sec"]:
            for tag, g in (("phản hồi, không chờ", (0, 0)), ("phản hồi + chờ xác nhận", best)):
                alert = simulate(prepared[g], y, t_end, unseen, completion, stream_end, delay)
                add(name, tag, delay, np.minimum(rule, alert), alert)
                if name == "ensemble" and g == best and delay == scfg["label_delays_sec"][0]:
                    demo = alert
        add(name, "chỉ hai luật", None, rule)
        logger.info("%s: k=%d H=%d (độ lẫn trên tune %s)", name, *best, score)

    # đối chứng không học sâu: n sự kiện liên tiếp chưa từng gặp lúc train thay cho "DeepLog trượt top-g"
    train = pd.concat([pd.read_pickle(paths["splits_dir"] / f"{n}.pkl") for n in ("train", "val")])["EventSequence"]
    params = chosen["ensemble"]["params"]
    completion = completion_times(t0, t_kth, deadline * params["deadline_margin"], stream_end)
    rule = np.minimum(unseen, completion)
    ngram_alerts, ngram_trig = {}, {}
    for n in range(1, window + 2):
        seen = {tuple(s[i + 1 - n:i + 1]) for s in train for i in range(window, len(s))}
        ngram_trig[n] = [np.array([i for i in range(window, len(s)) if tuple(s[i + 1 - n:i + 1]) not in seen], dtype=int)
                         for s in df["EventSequence"]]
        first = decisions(df["EventSequence"], df["EventTimes"], ngram_trig[n], 0, 0)
        ngram_alerts[n] = np.minimum(rule, [min((a for a, _ in d), default=np.inf) for d in first])
    best_n, _ = select_on_tune({n: np.isfinite(a).astype(int) for n, a in ngram_alerts.items()}, y, is_tune)
    for n in (best_n, 2):
        add("ngram", f"n-gram n={n}, không phản hồi", None, ngram_alerts[n])
        k, h = chosen["ensemble"]["confirm_events"], chosen["ensemble"]["confirm_seconds"]
        dec = decisions(df["EventSequence"], df["EventTimes"], ngram_trig[n], k, h)
        delay = scfg["label_delays_sec"][0]
        alert = simulate(dec, y, t_end, unseen, completion, stream_end, delay)
        add("ngram", f"n-gram n={n}, phản hồi + chờ xác nhận", delay, np.minimum(rule, alert), alert)

    best = chosen["ensemble"]
    streamed = verify_stream(cfg, df, y, t_end, best["confirm_events"], best["confirm_seconds"], scfg["label_delays_sec"][0])
    diff = int((~((streamed == demo) | (np.isinf(streamed) & np.isinf(demo)))).sum())
    report["stream_vs_offline_different_blocks"] = diff
    report["chosen"], report["ngram_best_n"] = chosen, int(best_n)
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "feedback_table.csv", index=False)
    save_json(report, out_dir / "feedback.json")
    seeds = table[table["source"].str.startswith("seed")].groupby(["variant", "label_delay_sec", "subset"], dropna=False)
    logger.info("\n%s", seeds[["f1", "precision", "recall", "fp_per_hour"]].agg(["mean", "std"]).round(3).to_string())
    ens = table[table["source"].isin(["ensemble", "ngram"])]
    logger.info("\n%s", ens[["source", "variant", "label_delay_sec", "subset", "precision", "recall", "f1", "fp_per_hour",
                             "latency_median_sec"]].round(3).to_string(index=False))
    if diff:
        raise SystemExit(f"Bản chạy luồng khác bản offline ở {diff} block — xem feedback.json")
    logger.info("Phản hồi: bản chạy luồng trùng bản offline ở mọi block.")


if __name__ == "__main__":
    main()
