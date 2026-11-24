"""
BƯỚC 13 — Demo phát hiện theo luồng: phát lại log theo thời gian, cảnh báo ngay kèm lý do.

Ba chế độ:
  (mặc định)     phát lại các session test bắt đầu từ --start-hour, tua nhanh --speed lần, in từng cảnh báo.
  --source log   đọc log THÔ: streaming.sample_lines dòng đầu của HDFS.log, tự chuyển dòng log → sự kiện bằng
                 template đã đóng băng; đồng thời đối chiếu chuỗi sự kiện với bước 2.
  --verify       phát lại TOÀN BỘ session test không chờ, so từng block × từng thành phần với bước 12, đo tốc độ.
                 Kết quả: outputs/results/stream/verify.json, demo_alerts.csv

Demo mặc định và --verify phát lại sự kiện đã parse (sessions + block_context), sắp theo thời gian; không đọc lại
log thô 1,5 GB. Trong cùng một giây, thứ tự giữa các block khác nhau không ảnh hưởng kết quả (trạng thái theo block).

Chạy (sau bước 12):
    python -m src.stream_demo --start-hour 31 --speed 600 --max-alerts 20
    python -m src.stream_demo --source log --speed 0
    python -m src.stream_demo --verify
"""
import argparse
import resource
import time
from collections import Counter
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import torch

from src.build_sequences import extract_block_ids
from src.config import load_config
from src.ensemble import load_seed_models
from src.hybrid_detect import learn_normal_profile
from src.model import DeepLogEnsemble
from src.model_selection import load_test_with_tune_mask
from src.parse_logs import build_header_regex, create_template_miner, split_header
from src.stream_detector import (COMPLETION, DEEPLOG, UNKNOWN_EVENT, UNSEEN, Alert, RankScorer, StreamDetector,
                                 SurprisalScorer)
from src.stream_compare import unigram_frequencies
from src.utils import get_logger, load_json, run_metadata, save_json

logger = get_logger(__name__)

PART_COLUMNS = {UNSEEN: "part_unseen", COMPLETION: "part_completion", DEEPLOG: "part_deeplog"}


class LineEncoder:
    """Dòng log thô → (chuỗi thời gian, [block id], chỉ số sự kiện), dùng template đã có, không học thêm."""

    def __init__(self, cfg: dict):
        self.header_regex, _ = build_header_regex(cfg["parse"]["log_format"])
        self.masker = create_template_miner(cfg["paths"]["drain_config"]).masker
        index = load_json(cfg["paths"]["event_index"])
        templates = pd.read_csv(cfg["paths"]["templates"])
        self.events = {t: index[e] for t, e in zip(templates["EventTemplate"], templates["EventId"])}

    def encode(self, line: str) -> tuple[str, list[str], int] | None:
        header = split_header(line, self.header_regex)
        if header is None:
            return None
        blocks = extract_block_ids(header["Content"])
        if not blocks:
            return None
        event = self.events.get(self.masker.mask(header["Content"]).strip(), UNKNOWN_EVENT)
        return header["Date"] + header["Time"], blocks, event


def build_detector(cfg: dict, score: str = "rank") -> tuple[StreamDetector, dict]:
    """
    Ghép model 5 seed, bất biến học từ train + val, và tham số đã chọn (không chọn lại).

    score = "rank": luật top-g, tham số bước 12. score = "mixed": độ bất ngờ trộn tần suất, tham số bước 15.
    """
    paths, hcfg, scfg = cfg["paths"], cfg["hybrid"], cfg["streaming"]
    offline = load_json(paths["results_dir"] / "stream" / "offline.json")
    params = offline["results"]["ensemble"][scfg["tune_mode"]]["params"]["hybrid_stream"]
    replica_event = load_json(paths["event_index"])[hcfg["replica_event"]]
    train = pd.concat([pd.read_pickle(paths["splits_dir"] / f"{n}.pkl") for n in ("train", "val")])
    profile = learn_normal_profile(train.merge(pd.read_pickle(paths["block_context"]), on="BlockId"), replica_event)
    models, _ = load_seed_models(cfg, torch.device("cpu"))
    # lô mỗi giây rất nhỏ (trung bình < 100 cửa sổ) nên CPU nhanh hơn chuyển dữ liệu sang GPU
    model = DeepLogEnsemble(models).eval()
    if score == "mixed":
        params = load_json(paths["results_dir"] / "stream" / "compare.json")["chosen"]["ensemble"]["mixed"]
        scorer = SurprisalScorer(model, scfg["cache_size"], params["eps"], unigram_frequencies(train["EventSequence"]))
        params = {"deadline_margin": params["m"], "top_g": params["threshold"], "eps": params["eps"]}
    else:
        scorer = RankScorer(model, scfg["cache_size"])
    detector = StreamDetector(scorer, profile["seen_events"], replica_event, profile["replica_count"],
                              profile["deadline_sec"] * params["deadline_margin"], params["top_g"],
                              cfg["model"]["window_size"])
    return detector, params


def replay_order(times: pd.Series, sequences: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Trải các session thành luồng sự kiện (hàng block, sự kiện, thời điểm) sắp theo thời gian, giữ thứ tự trong block."""
    lengths = sequences.map(len).to_numpy()
    rows = np.repeat(np.arange(len(sequences)), lengths)
    events = np.concatenate(sequences.map(np.asarray).to_list())
    stamps = np.concatenate(times.to_list())
    order = np.argsort(stamps, kind="stable")
    return rows[order], events[order], stamps[order]


def by_second(rows: np.ndarray, events: np.ndarray, stamps: np.ndarray):
    """Sinh (giây, [(block, sự kiện), ...])."""
    cuts = np.flatnonzero(np.diff(stamps)) + 1
    for lo, hi in zip(np.r_[0, cuts], np.r_[cuts, len(stamps)]):
        yield float(stamps[lo]), list(zip(rows[lo:hi].tolist(), events[lo:hi].tolist()))


def describe(alert: Alert, names: dict[int, str], detector: StreamDetector) -> str:
    """Lý do cảnh báo bằng lời."""
    if alert.component == UNSEEN:
        event = names.get(alert.event, "dòng không khớp template nào")
        return f"[LUẬT sự kiện lạ] {event} chưa từng có trong session bình thường lúc train"
    if alert.component == COMPLETION:
        return (f"[LUẬT nhân bản] sau {detector.deadline:.0f} giây mới có {alert.replicas}/{detector.replica_count} "
                "lần ghi nhận bản sao")
    expected = ", ".join(names.get(e, "?") for e in alert.expected)
    event = names.get(alert.event, "?")
    if isinstance(detector.scorer, SurprisalScorer):
        return f"[DEEPLOG] {event} bất ngờ {alert.score:.1f} (ngưỡng {detector.top_g:g}, hạng {alert.rank}); model chờ {expected}"
    return f"[DEEPLOG] {event} xếp hạng {alert.rank} (ngưỡng {detector.top_g}); model chờ {expected}"


def show(alerts: list[Alert], block_name, names, detector, origin: datetime, labels=None) -> int:
    """In cảnh báo đầu tiên của từng block, trả về số cảnh báo đã in."""
    shown = 0
    for a in alerts:
        if not a.first:
            continue
        label = "" if labels is None else ("  ✔ đúng" if labels[a.block] == 1 else "  ✘ báo nhầm")
        print(f"{origin + timedelta(seconds=a.time):%d/%m %H:%M:%S}  {block_name(a.block):<26} "
              f"{describe(a, names, detector)}{label}", flush=True)
        shown += 1
    return shown


def log_origin(cfg: dict) -> datetime:
    with open(cfg["paths"]["raw_log"], encoding="utf-8", errors="replace") as f:
        return datetime.strptime(f.readline()[:13], "%y%m%d %H%M%S")


def load_test(cfg: dict) -> tuple[pd.DataFrame, np.ndarray]:
    paths, scfg = cfg["paths"], cfg["streaming"]
    test, is_tune, _ = load_test_with_tune_mask(paths["splits_dir"], cfg["select"], scfg["tune_mode"])
    df = test.merge(pd.read_pickle(paths["block_context"])[["BlockId", "EventTimes"]], on="BlockId", how="left")
    return df, is_tune


def run_replay(cfg: dict, args) -> None:
    detector, params = build_detector(cfg, args.score)
    df, _ = load_test(cfg)
    t0 = df["EventTimes"].map(lambda t: t[0]).to_numpy()
    df = df[t0 >= args.start_hour * 3600].reset_index(drop=True)
    names = {i: e for e, i in load_json(cfg["paths"]["event_index"]).items()}
    origin, labels = log_origin(cfg), df["Label"].to_numpy()
    print(f"Phát lại {len(df)} block từ giờ {args.start_hour} | tham số: {params} | tua nhanh x{args.speed:g}\n")
    shown, last = 0, None
    for second, batch in by_second(*replay_order(df["EventTimes"], df["EventSequence"])):
        if args.speed > 0 and last is not None:
            time.sleep((second - last) / args.speed)
        last = second
        shown += show(detector.process(second, batch), lambda b: df["BlockId"][b], names, detector, origin,
                      labels if args.show_labels else None)
        if args.max_alerts and shown >= args.max_alerts:
            break
    print(f"\nĐã xử lý {detector.events} sự kiện, {len(detector.history)} block, {shown} block bị cảnh báo.")


def run_log_sample(cfg: dict, args) -> None:
    detector, params = build_detector(cfg, args.score)
    encoder = LineEncoder(cfg)
    names = {i: e for e, i in load_json(cfg["paths"]["event_index"]).items()}
    origin, limit = log_origin(cfg), cfg["streaming"]["sample_lines"]
    print(f"Đọc {limit} dòng đầu của {cfg['paths']['raw_log'].name} | tham số: {params}\n")
    seen: dict[str, list[tuple[int, int]]] = {}
    second, batch, shown, lines, skipped, unknown, backwards = None, [], 0, 0, 0, 0, 0
    parse_time = 0.0
    start = time.perf_counter()
    with open(cfg["paths"]["raw_log"], encoding="utf-8", errors="replace") as f:
        for line in f:
            if lines >= limit:
                break
            lines += 1
            tick = time.perf_counter()
            parsed = encoder.encode(line)
            parse_time += time.perf_counter() - tick
            if parsed is None:
                skipped += 1
                continue
            stamp, blocks, event = parsed
            now = int((datetime.strptime(stamp, "%y%m%d%H%M%S") - origin).total_seconds())
            unknown += event == UNKNOWN_EVENT
            if second is not None and now != second:
                backwards += now < second
                shown += show(detector.process(second, batch), str, names, detector, origin)
                if args.speed > 0 and now > second:
                    time.sleep((now - second) / args.speed)
                batch = []
            second = now
            for b in blocks:
                batch.append((b, event))
                seen.setdefault(b, []).append((event, now))
    shown += show(detector.process(second, batch), str, names, detector, origin)
    shown += show(detector.finish(second), str, names, detector, origin)
    total = time.perf_counter() - start

    # chuỗi sự kiện đọc theo luồng phải là phần đầu của chuỗi bước 2 (block chưa hết log thì ngắn hơn)
    paths = cfg["paths"]
    ref = pd.read_pickle(paths["sessions"])[["BlockId", "EventSequence"]].merge(
        pd.read_pickle(paths["block_context"])[["BlockId", "EventTimes"]], on="BlockId")
    ref = ref[ref["BlockId"].isin(seen)]
    wrong = sum([(e, t) for e, t in zip(s[:len(seen[b])], ts[:len(seen[b])].tolist())] != seen[b]
                for b, s, ts in zip(ref["BlockId"], ref["EventSequence"], ref["EventTimes"]))
    report = {"meta": run_metadata(cfg), "lines": lines, "lines_without_block": skipped, "unknown_template_lines": unknown,
              "time_going_backwards": backwards, "blocks": len(seen), "blocks_missing_in_step2": len(seen) - len(ref),
              "blocks_with_different_prefix": wrong, "alerted_blocks": shown,
              "seconds_total": total if args.speed == 0 else None, "seconds_parsing": parse_time,
              "lines_per_second_parsing": lines / parse_time,
              "lines_per_second_end_to_end": lines / total if args.speed == 0 else None}
    save_json(report, paths["results_dir"] / "stream" / "log_sample.json")
    logger.info("Mẫu log thô: %s", {k: v for k, v in report.items() if k != "meta"})


def run_verify(cfg: dict) -> None:
    out_dir = cfg["paths"]["results_dir"] / "stream"
    reference = pd.read_pickle(out_dir / "offline_alerts.pkl")
    detector, params = build_detector(cfg)
    df, _ = load_test(cfg)
    if not (df["BlockId"].to_numpy() == reference["BlockId"].to_numpy()).all():
        raise ValueError("offline_alerts.pkl không cùng tập block với test.pkl hiện tại — chạy lại bước 12")

    stream = replay_order(df["EventTimes"], df["EventSequence"])
    per_second = np.unique(stream[2], return_counts=True)[1]
    alerts: list[Alert] = []
    start = time.perf_counter()
    for second, batch in by_second(*stream):
        alerts += detector.process(second, batch)
    alerts += detector.finish(float(stream[2][-1]))
    elapsed = time.perf_counter() - start

    got = {c: np.full(len(df), np.inf) for c in PART_COLUMNS}
    first = np.full(len(df), np.inf)
    for a in alerts:
        got[a.component][a.block] = a.time
        if a.first:
            first[a.block] = a.time
    checks = {}
    for name, values, column in [(c, got[c], col) for c, col in PART_COLUMNS.items()] + [("first", first, "hybrid_stream")]:
        ref = reference[column].to_numpy()
        both = np.isfinite(values) & np.isfinite(ref)
        checks[name] = {"offline_blocks": int(np.isfinite(ref).sum()), "stream_blocks": int(np.isfinite(values).sum()),
                        "only_offline": int((np.isfinite(ref) & ~np.isfinite(values)).sum()),
                        "only_stream": int((~np.isfinite(ref) & np.isfinite(values)).sum()),
                        "different_time": int((values[both] != ref[both]).sum())}
    ok = all(c["only_offline"] == c["only_stream"] == c["different_time"] == 0 for c in checks.values())

    # tốc độ model khi KHÔNG dùng bộ nhớ đệm, chấm từng lô 80 cửa sổ (số sự kiện trung bình mỗi giây của log)
    bare = RankScorer(detector.scorer.model, cache_size=0)
    windows = list(detector.scorer.cache)[:8000]
    tick = time.perf_counter()
    for i in range(0, len(windows), 80):
        bare._score(windows[i:i + 80])
    bare_rate = len(windows) / (time.perf_counter() - tick)

    report = {"meta": run_metadata(cfg), "params": params, "identical_to_offline": ok, "checks": checks,
              "events": detector.events, "seconds": elapsed, "events_per_second": detector.events / elapsed,
              "log_events_per_second": {"mean": detector.events / float(stream[2][-1] - stream[2][0]),
                                        "p99_of_active_seconds": float(np.percentile(per_second, 99)), "max": int(per_second.max())},
              "cache_hit_rate": 1 - detector.scorer.misses / max(detector.scorer.requests, 1),
              "scored_windows": detector.scorer.requests, "model_windows_per_second_no_cache": bare_rate,
              "max_rss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9,
              "first_alert_component": dict(Counter(a.component for a in alerts if a.first))}
    save_json(report, out_dir / "verify.json")
    names = {i: e for e, i in load_json(cfg["paths"]["event_index"]).items()}
    ids, labels = df["BlockId"].to_numpy(), df["Label"].to_numpy()
    pd.DataFrame([{"block": ids[a.block], "time_sec": a.time, "component": a.component, "first": a.first,
                   "event": names.get(a.event), "rank": a.rank, "label": int(labels[a.block]),
                   "reason": describe(a, names, detector)} for a in alerts]).to_csv(out_dir / "demo_alerts.csv", index=False)
    logger.info("%s", {k: v for k, v in report.items() if k != "meta"})
    if not ok:
        raise SystemExit("Demo KHÔNG trùng kết quả offline — xem verify.json")
    logger.info("Demo trùng kết quả offline ở mọi block, mọi thành phần.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Bước 13: demo phát hiện theo luồng")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--source", choices=["sessions", "log"], default="sessions")
    parser.add_argument("--start-hour", type=float, default=14.5, help="giờ bắt đầu phát lại, tính từ đầu log")
    parser.add_argument("--speed", type=float, default=600, help="số lần tua nhanh; 0 = không chờ")
    parser.add_argument("--max-alerts", type=int, default=0, help="dừng sau N cảnh báo; 0 = chạy hết")
    parser.add_argument("--show-labels", action="store_true", help="in thêm nhãn thật (chỉ để trình bày)")
    parser.add_argument("--score", choices=["rank", "mixed"], default="rank", help="cách DeepLog chấm (mixed = bước 15)")
    parser.add_argument("--verify", action="store_true", help="kiểm tra cách chấm rank với bước 12; mixed tự kiểm ở bước 15")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.verify:
        run_verify(cfg)
    elif args.source == "log":
        run_log_sample(cfg, args)
    else:
        run_replay(cfg, args)


if __name__ == "__main__":
    main()
