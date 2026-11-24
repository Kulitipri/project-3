"""
BƯỚC 12 — Kết quả offline theo quy tắc CHẠY ĐƯỢC THEO LUỒNG (mốc để kiểm tra demo ở bước 13).

Vì sao cần: bước 5–11 chấm session có <= window_size sự kiện ở SỰ KIỆN CUỐI của nó. Khi log đang chảy, hệ thống
không biết sự kiện nào là cuối, nên DeepLog chạy theo luồng chỉ được chấm từ sự kiện thứ window_size + 1
(đủ cửa sổ thật, không đệm PAD). Hai luật bất biến phụ trách phần đầu session.

Ba chỗ khác bước 8:
  1. Bỏ mọi rank ở vị trí < window_size (không dùng thông tin "session đã hết").
  2. g và hệ số thời hạn m được CHỌN LẠI trên tập tune theo quy tắc mới; trong các m cho F1 không khác nhau có ý nghĩa
     (bootstrap theo cặp) thì lấy m nhỏ nhất, vì F1 không phản ánh độ trễ.
  3. Thêm tập đánh giá "strict": chỉ gồm session bắt đầu SAU khi mọi session train / val / tune đã kết thúc, tức
     model, luật và ngưỡng đều có sẵn trước khi session đầu tiên của tập này xuất hiện (chỉ có với tune earliest).

Input : splits/*.pkl, block_context.pkl, deeplog_ranks.pkl của từng seed và của ensemble (bước 9, 10)
Output: outputs/results/stream/offline.json         — tham số + chỉ số theo nguồn × tune_mode × tập × bộ phát hiện
        outputs/results/stream/offline_table.csv    — cùng nội dung, dạng bảng dài
        outputs/results/stream/offline_summary.csv  — trung bình ± độ lệch chuẩn qua seed, và ensemble
        outputs/results/stream/offline_alerts.pkl   — thời điểm cảnh báo từng block của ensemble (tune = streaming.tune_mode)

Chạy (~3 phút, sau bước 10):
    python -m src.stream_offline --config configs/config.yaml
"""
import argparse
import itertools

import numpy as np
import pandas as pd

from src.config import load_config
from src.deeplog_scores import load_ranks_source
from src.evaluate import recall_by_unseen, unseen_event_mask
from src.hybrid_detect import (deeplog_alert_times, learn_normal_profile, ops_metrics, select_on_tune, time_to_kth,
                               unseen_alert_times)
from src.model_selection import load_test_with_tune_mask
from src.seed_study import TUNE_MODES
from src.utils import ensure_dir, get_logger, load_json, run_metadata, save_json

logger = get_logger(__name__)


def keep_streamable(ranks: pd.DataFrame, window_size: int) -> pd.DataFrame:
    """Chỉ giữ rank tính được theo luồng: vị trí dự đoán có đủ window_size sự kiện thật đứng trước."""
    out = ranks.copy()
    keep = [pos >= window_size for pos in ranks["TargetPos"]]
    out["Ranks"] = [r[k] for r, k in zip(ranks["Ranks"], keep)]
    out["TargetPos"] = [p[k] for p, k in zip(ranks["TargetPos"], keep)]
    return out


def max_rank(ranks: pd.Series) -> np.ndarray:
    return ranks.map(lambda r: int(r.max()) if len(r) else 0).to_numpy()


def completion_times(t0: np.ndarray, t_kth: np.ndarray, deadline: float, stream_end: float) -> np.ndarray:
    """Thời điểm luật nhân bản đúng hạn phát cảnh báo (inf nếu không)."""
    fire = (t_kth > deadline) & (t0 + deadline <= stream_end)
    return np.where(fire, t0 + deadline, np.inf)


def f1_differs(y: np.ndarray, a: np.ndarray, b: np.ndarray, seed: int, n_boot: int = 2000) -> bool:
    """
    Bootstrap theo cặp trên tập tune: F1 của hai cách báo a, b có khác nhau (khoảng tin cậy 95% không chứa 0) không.

    Gom session theo (nhãn, a, b) thành 8 nhóm rồi lấy mẫu lại số lượng mỗi nhóm, tương đương lấy mẫu lại từng session.
    """
    cells = np.bincount(y * 4 + a * 2 + b, minlength=8)
    draws = np.random.default_rng(seed).multinomial(cells.sum(), cells / cells.sum(), size=n_boot)
    label, pa, pb = np.arange(8) // 4, np.arange(8) // 2 % 2, np.arange(8) % 2

    def f1(pred):
        return 2 * draws[:, (label == 1) & (pred == 1)].sum(1) / np.maximum(
            draws[:, label == 1].sum(1) + draws[:, pred == 1].sum(1), 1)

    low, high = np.percentile(f1(pa) - f1(pb), [2.5, 97.5])
    return bool(low > 0 or high < 0)


def shortest_equivalent(preds: dict, best, y: np.ndarray, is_tune: np.ndarray, seed: int):
    """Khoá nhỏ nhất (thời hạn ngắn nhất) có F1 trên tune không khác có ý nghĩa so với khoá tốt nhất."""
    for key in sorted(preds):
        if key == best or not f1_differs(y[is_tune], preds[best][is_tune].astype(int), preds[key][is_tune].astype(int), seed):
            return key
    return best


def evaluate_source(df: pd.DataFrame, is_tune: np.ndarray, profile: dict, hcfg: dict, common: dict,
                    seed: int) -> tuple[dict, dict]:
    """
    Chọn tham số trên tune rồi trả về (tham số, {tên bộ phát hiện: thời điểm cảnh báo từng session}).

    common: các mảng không phụ thuộc DeepLog (y, t0, unseen, t_kth, stream_end) — tính 1 lần cho mọi seed.
    """
    y, t0, unseen = common["y"], common["t0"], common["unseen"]
    margins, g_values = hcfg["deadline_margins"], list(range(1, hcfg["max_top_g"] + 1))
    completion = {m: completion_times(t0, common["t_kth"], profile["deadline_sec"] * m, common["stream_end"])
                  for m in margins}
    rules = {m: np.isfinite(unseen) | np.isfinite(completion[m]) for m in margins}
    top_old, top_new = max_rank(df["RanksAll"]), max_rank(df["Ranks"])

    g_old, _ = select_on_tune({g: (top_old > g).astype(int) for g in g_values}, y, is_tune)
    g_new, _ = select_on_tune({g: (top_new > g).astype(int) for g in g_values}, y, is_tune)
    m_rules, _ = select_on_tune({m: rules[m].astype(int) for m in margins}, y, is_tune)
    (m_hyb, g_hyb), _ = select_on_tune({(m, g): (rules[m] | (top_new > g)).astype(int)
                                        for m, g in itertools.product(margins, g_values)}, y, is_tune)
    # F1 gần như không phụ thuộc m, còn độ trễ thì tỉ lệ với m: lấy thời hạn ngắn nhất không kém có ý nghĩa
    m_rules = shortest_equivalent(rules, m_rules, y, is_tune, seed)
    m_hyb = shortest_equivalent({m: rules[m] | (top_new > g_hyb) for m in margins}, m_hyb, y, is_tune, seed)
    whole = pd.DataFrame({"EventTimes": df["EventTimes"], "TargetPos": df["TargetPosAll"], "Ranks": df["RanksAll"]})
    deeplog_hyb = deeplog_alert_times(df, g_hyb)
    detectors = {
        "deeplog_whole_session": deeplog_alert_times(whole, g_old),  # cách chấm của bước 5–11, để đối chiếu
        "deeplog_stream": deeplog_alert_times(df, g_new),
        "rules": np.minimum(unseen, completion[m_rules]),
        "hybrid_stream": np.minimum.reduce([unseen, completion[m_hyb], deeplog_hyb]),
        # ba thành phần của hybrid_stream, để bước 13 so từng thành phần
        "part_unseen": unseen, "part_completion": completion[m_hyb], "part_deeplog": deeplog_hyb,
    }
    for m in margins:  # độ nhạy theo thời hạn (không dùng để chọn): đổi m, giữ g đã chọn
        detectors[f"margin{m}_rules"] = np.minimum(unseen, completion[m])
        detectors[f"margin{m}_hybrid"] = np.minimum(detectors[f"margin{m}_rules"], deeplog_hyb)
    params = {"deeplog_whole_session": {"top_g": g_old}, "deeplog_stream": {"top_g": g_new},
              "rules": {"deadline_margin": m_rules}, "hybrid_stream": {"deadline_margin": m_hyb, "top_g": g_hyb}}
    return params, detectors


def deeplog_contribution(y: np.ndarray, t0: np.ndarray, detectors: dict, mask: np.ndarray) -> dict:
    """DeepLog bắt thêm được gì ngoài hai luật (trong hybrid_stream), và phải trả bao nhiêu cảnh báo giả."""
    rules = np.minimum(detectors["part_unseen"], detectors["part_completion"])[mask]
    deep, pos, start = detectors["part_deeplog"][mask], y[mask] == 1, t0[mask]
    only = np.isfinite(deep) & ~np.isfinite(rules)
    both = np.isfinite(deep) & np.isfinite(rules) & pos
    earlier = both & (deep < rules)
    n_missed, tp, fp = int((pos & ~np.isfinite(rules)).sum()), int((only & pos).sum()), int((only & ~pos).sum())
    return {"anomalies_missed_by_rules": n_missed, "of_which_deeplog_finds": tp, "extra_false_alarms": fp,
            "precision_of_extra_alerts": tp / max(tp + fp, 1),
            "median_latency_of_extra_sec": float(np.median((deep - start)[only & pos])) if tp else None,
            "found_by_both": int(both.sum()), "deeplog_earlier_than_rules": int(earlier.sum()),
            "median_lead_when_earlier_sec": float(np.median((rules - deep)[earlier])) if earlier.any() else None}


def main() -> None:
    parser = argparse.ArgumentParser(description="Bước 12: kết quả offline theo quy tắc chạy theo luồng")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths, hcfg, sel = cfg["paths"], cfg["hybrid"], cfg["select"]
    window, results_dir = cfg["model"]["window_size"], paths["results_dir"]
    out_dir = ensure_dir(results_dir / "stream")
    replica_event = load_json(paths["event_index"])[hcfg["replica_event"]]

    context = pd.read_pickle(paths["block_context"])
    train = pd.concat([pd.read_pickle(paths["splits_dir"] / f"{n}.pkl") for n in ("train", "val")]).merge(context, on="BlockId")
    profile = learn_normal_profile(train, replica_event)
    train_end = float(train["EventTimes"].map(lambda t: t[-1]).max())

    base = load_test_with_tune_mask(paths["splits_dir"], sel)[0].merge(context, on="BlockId", how="left")
    t0 = base["EventTimes"].map(lambda t: t[0]).to_numpy(dtype=float)
    t_end = base["EventTimes"].map(lambda t: t[-1]).to_numpy(dtype=float)
    common = {"y": base["Label"].to_numpy(), "t0": t0, "stream_end": float(t_end.max()),
              "unseen": unseen_alert_times(base, profile["seen_events"]),
              "t_kth": np.array([time_to_kth(s, t, replica_event, profile["replica_count"])
                                 for s, t in zip(base["EventSequence"], base["EventTimes"])])}
    y = common["y"]
    group_unseen = unseen_event_mask(base["EventSequence"], pd.read_pickle(paths["splits_dir"] / "train.pkl")["EventSequence"])
    short = base["EventSequence"].map(len).to_numpy() <= window
    span_hours = (t_end.max() - t0.min()) / 3600

    sources = {f"seed{s}": results_dir / "seeds" / f"seed{s}" / TUNE_MODES[0] for s in cfg["robustness"]["seeds"]}
    sources["ensemble"] = results_dir / "ensemble" / TUNE_MODES[0]
    summary = {"meta": run_metadata(cfg), "window_size": window, "stream_hours": span_hours,
               "profile": {"num_seen_events": len(profile["seen_events"]), "replica_count": profile["replica_count"],
                           "deadline_sec": profile["deadline_sec"]}, "subsets": {}, "sources": {}, "results": {}}
    rows, demo_alerts = [], None

    for name, ranks_dir in sources.items():
        ranks = pd.read_pickle(ranks_dir / "deeplog_ranks.pkl")
        old = ranks.rename(columns={"Ranks": "RanksAll", "TargetPos": "TargetPosAll"})
        df = base.merge(keep_streamable(ranks, window), on="BlockId", how="left").merge(old, on="BlockId", how="left")
        assert len(df) == len(base) and df["Ranks"].notna().all()
        seed_cfg = cfg if name == "ensemble" else {**cfg, "seed": int(name.removeprefix("seed"))}
        summary["sources"][name] = load_ranks_source(ranks_dir, seed_cfg)
        for mode in TUNE_MODES:
            is_tune = load_test_with_tune_mask(paths["splits_dir"], sel, mode)[1]
            params, detectors = evaluate_source(df, is_tune, profile, hcfg, common, cfg["seed"])
            subsets = {"eval": (~is_tune, span_hours * (1 - sel["tune_ratio"]))}  # quy ước FP/giờ như bước 8
            if mode == "earliest":
                cut = max(train_end, float(t_end[is_tune].max()))
                summary["strict_start_hour"] = cut / 3600
                # chia random: train trải khắp log nên không còn session nào sau mốc này → không có tập strict
                if len(np.unique(y[t0 > cut])) == 2:
                    subsets["strict"] = (t0 > cut, (common["stream_end"] - cut) / 3600)
            result = {"params": params, "metrics": {}, "deeplog_contribution": {}}
            for subset, (mask, hours) in subsets.items():
                summary["subsets"][f"{mode}/{subset}"] = {
                    "sessions": int(mask.sum()), "anomalies": int(y[mask].sum()), "hours": hours,
                    "short_anomalies": int((short & (y == 1) & mask).sum()),
                    "short_normals": int((short & (y == 0) & mask).sum())}
                result["deeplog_contribution"][subset] = deeplog_contribution(y, t0, detectors, mask)
                for det, alert in detectors.items():
                    m = ops_metrics(y[mask], alert[mask], t0[mask], hours)
                    m.update(recall_by_unseen(y[mask], np.isfinite(alert[mask]).astype(int), group_unseen[mask]))
                    sh = mask & short & (y == 1)
                    m["recall_short"] = float(np.isfinite(alert[sh]).mean()) if sh.any() else None
                    result["metrics"].setdefault(subset, {})[det] = m
                    rows.append({"source": name, "tune_mode": mode, "subset": subset, "detector": det, **m})
            summary["results"].setdefault(name, {})[mode] = result
            if name == "ensemble" and mode == cfg["streaming"]["tune_mode"]:
                demo_alerts = pd.DataFrame({"BlockId": df["BlockId"], "Label": y, "is_tune": is_tune, "t0": t0, "t_end": t_end,
                                            **{k: detectors[k] for k in ("hybrid_stream", "part_unseen",
                                                                         "part_completion", "part_deeplog")}})
        logger.info("%s xong", name)

    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "offline_table.csv", index=False)
    metrics = ["precision", "recall", "f1", "fp_per_hour", "latency_median_sec", "latency_p90_sec", "detected_within_60s",
               "recall_unseen", "recall_known", "recall_short"]
    keys = ["tune_mode", "subset", "detector"]
    seeds = table[table["source"] != "ensemble"].groupby(keys)[metrics].agg(["mean", "std"])
    seeds.columns = [f"seed_{m}_{s}" for m, s in seeds.columns]
    ens = table[table["source"] == "ensemble"].set_index(keys)[metrics].add_prefix("ensemble_")
    seeds.join(ens).reset_index().to_csv(out_dir / "offline_summary.csv", index=False)
    demo_alerts.to_pickle(out_dir / "offline_alerts.pkl")
    save_json(summary, out_dir / "offline.json")
    for _, r in seeds.join(ens).reset_index().iterrows():
        if not r["detector"].startswith(("part_", "margin")):
            logger.info("%-8s %-6s %-21s ensemble F1 %.3f | seed %.3f ± %.3f | FP/giờ %5.1f | trễ %s s", r["tune_mode"], r["subset"],
                        r["detector"], r["ensemble_f1"], r["seed_f1_mean"], r["seed_f1_std"],
                        r["ensemble_fp_per_hour"], r["ensemble_latency_median_sec"])


if __name__ == "__main__":
    main()
