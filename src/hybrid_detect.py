"""
BƯỚC 8 — Bộ phát hiện lai (hybrid): luật bất biến học từ dữ liệu Normal + DeepLog, đánh giá theo góc nhìn vận hành.

Vì sao cần: DeepLog chỉ hỏi "sự kiện KẾ TIẾP có hợp lý không?", nên mù với 2 loại bất thường:
  (1) sự kiện KHÔNG xảy ra — block được cấp phát rồi dừng giữa chừng (vd chuỗi "E2 E1"): không có
      sự kiện sai nào để DeepLog bắt, chỉ có sự im lặng;
  (2) sự kiện chưa từng thấy lúc train — DeepLog chỉ bắt được nếu g đủ nhỏ, kéo theo nhiều báo động giả.
Cả 2 được mô tả bằng "bất biến" (invariant) học từ CHÍNH các session Normal dùng để train DeepLog,
không cần nhãn:
  - unseen     : session chứa EventId chưa từng xuất hiện trong train → cảnh báo ngay tại sự kiện đó.
  - completion : mọi block Normal trong train đều nhận đủ k xác nhận lưu trữ (E5 "addStoredBlock",
                 k = số lần ít nhất trong train) trong vòng T giây (T = lâu nhất trong train).
                 Block chưa đủ k lần E5 sau T × margin giây → cảnh báo tại thời điểm t0 + T × margin
                 (bỏ qua nếu thời điểm đó nằm sau dòng log cuối cùng — luồng log đã dừng).
  - deeplog    : luật top-g như detect.py, cảnh báo tại sự kiện đầu tiên nằm ngoài top-g.
Hybrid = unseen OR completion OR deeplog. Thời điểm cảnh báo = sớm nhất trong các thành phần.

Chọn tham số (g, margin) trên tập tune, báo cáo trên tập eval — CÙNG cách chia với model_selection.py.

Ngoài F1, báo cáo thêm các chỉ số vận hành:
  - fp_per_hour  : số cảnh báo giả mỗi giờ (quy về toàn bộ luồng log) — gánh nặng cho người trực.
  - latency      : thời gian từ sự kiện đầu tiên của block tới lúc cảnh báo (chỉ tính TP).
                   Các phương pháp đếm sự kiện (PCA, IForest) phải chờ session kết thúc mới chấm được,
                   nên độ trễ của chúng = thời lượng session (end_of_session).
  - drift        : FPR theo từng khoảng thời gian — train chỉ có ~9 phút log đầu, test kéo dài ~38 giờ.
Phân tích phụ (kết quả âm tính, giữ lại để tái lập): rủi ro theo datanode — tỉ lệ block đi qua cùng
datanode bị cảnh báo trong risk_window_sec giây trước đó (chỉ dùng cảnh báo trong quá khứ, không dùng nhãn).

Input : data/processed/splits/{train,val,test}.pkl, event_index.json
        data/interim/block_context.pkl         (python -m src.block_context)
        outputs/results/deeplog_ranks.pkl      (python -m src.deeplog_scores)
        outputs/results/model_selection_<tune_mode>.json  (nếu có — để đưa PCA/IForest vào bảng so sánh)
Output: outputs/results/hybrid_detection.json   — bảng so sánh trên tập eval + tham số đã chọn
        outputs/results/hybrid_drift.csv        — FPR / recall theo khoảng thời gian
        outputs/results/hybrid_deadline.csv     — độ nhạy của luật completion theo margin
        outputs/results/hybrid_alerts.pkl       — thời điểm cảnh báo từng session (vẽ hình)

Chạy:
    python -m src.hybrid_detect --config configs/config.yaml
"""
import argparse
import itertools

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.config import load_config
from src.deeplog_scores import load_ranks_source
from src.evaluate import compute_metrics, save_results, recall_by_unseen, unseen_event_mask
from src.model_selection import load_test_with_tune_mask
from src.utils import get_logger, load_json, run_metadata

logger = get_logger(__name__)


def learn_normal_profile(train_df: pd.DataFrame, replica_event: int) -> dict:
    """
    Học các bất biến từ session Normal (train + val) — đã ghép block_context.

    Output: {"seen_events": set, "replica_count": k, "deadline_sec": T}
    """
    counts = train_df["EventSequence"].map(lambda s: s.count(replica_event))
    k = int(counts.min())
    times = [time_to_kth(seq, t, replica_event, k) for seq, t in zip(train_df["EventSequence"], train_df["EventTimes"])]
    return {
        "seen_events": {e for seq in train_df["EventSequence"] for e in seq},
        "replica_count": k,
        "deadline_sec": float(max(times)),
    }


def time_to_kth(sequence: list[int], times: np.ndarray, event: int, k: int) -> float:
    """Số giây từ sự kiện đầu tiên tới lần thứ k của `event`; inf nếu không đủ k lần."""
    idx = np.flatnonzero(np.asarray(sequence) == event)
    return float(times[idx[k - 1]] - times[0]) if len(idx) >= k else np.inf


def first_alert_time(times: np.ndarray, positions: np.ndarray, mask: np.ndarray) -> float:
    """Thời điểm của vị trí đầu tiên có mask = True; inf nếu không có."""
    hit = np.flatnonzero(mask)
    return float(times[positions[hit[0]]]) if len(hit) else np.inf


def unseen_alert_times(df: pd.DataFrame, seen_events: set) -> np.ndarray:
    """Thời điểm của sự kiện chưa-từng-thấy đầu tiên trong mỗi session (inf nếu không có)."""
    out = np.full(len(df), np.inf)
    for i, (seq, times) in enumerate(zip(df["EventSequence"], df["EventTimes"])):
        for pos, event in enumerate(seq):
            if event not in seen_events:
                out[i] = times[pos]
                break
    return out


def deeplog_alert_times(df: pd.DataFrame, top_g: int) -> np.ndarray:
    """Thời điểm của cửa sổ đầu tiên có rank > g (inf nếu session Normal theo luật top-g)."""
    return np.array([
        first_alert_time(times, pos, ranks > top_g)
        for times, pos, ranks in zip(df["EventTimes"], df["TargetPos"], df["Ranks"])
    ])


def node_risk(df: pd.DataFrame, alert_times: np.ndarray, window_sec: int) -> np.ndarray:
    """
    Rủi ro theo datanode, tính tại thời điểm block bắt đầu (t0), chỉ nhìn về quá khứ.

    risk(node, t0) = số cảnh báo trên các block đi qua node trong [t0 - window, t0)
                     / số block đi qua node bắt đầu trong [t0 - window, t0)
    Rủi ro của block = max trên các datanode của nó. Không dùng nhãn.
    """
    exploded = pd.DataFrame({
        "row": np.arange(len(df)), "t0": df["t0"].to_numpy(), "alert": alert_times, "node": df["Nodes"],
    }).explode("node").dropna(subset=["node"])
    risk = np.zeros(len(df))
    for _, grp in exploded.groupby("node"):
        starts = np.sort(grp["t0"].to_numpy())
        alerts = np.sort(grp["alert"].to_numpy()[np.isfinite(grp["alert"].to_numpy())])
        q = grp["t0"].to_numpy()
        n_blocks = np.searchsorted(starts, q, "left") - np.searchsorted(starts, q - window_sec, "left")
        n_alerts = np.searchsorted(alerts, q, "left") - np.searchsorted(alerts, q - window_sec, "left")
        np.maximum.at(risk, grp["row"].to_numpy(), n_alerts / np.maximum(n_blocks, 1))
    return risk


def ops_metrics(y: np.ndarray, alert: np.ndarray, t0: np.ndarray, hours: float) -> dict:
    """
    P/R/F1 + chỉ số vận hành cho một bộ phát hiện (alert = thời điểm cảnh báo, inf = không cảnh báo).

    hours — số giờ của luồng log mà tập này đại diện (để quy FP về mỗi giờ).
    """
    pred = np.isfinite(alert).astype(int)
    metrics = compute_metrics(y, pred)
    delay = (alert - t0)[(y == 1) & (pred == 1)]
    metrics.update({
        "fp_per_hour": metrics["fp"] / hours,
        "latency_median_sec": float(np.median(delay)) if len(delay) else None,
        "latency_p90_sec": float(np.percentile(delay, 90)) if len(delay) else None,
        "detected_within_60s": float((delay <= 60).mean()) if len(delay) else None,
    })
    return metrics


def select_on_tune(candidates: dict, y: np.ndarray, is_tune: np.ndarray) -> tuple:
    """Chọn khoá có F1 trên tập tune cao nhất (bằng điểm → giữ khoá gặp trước). candidates: khoá → mảng 0/1."""
    best_key, best_f1 = None, -1.0
    for key, pred in candidates.items():
        f1 = compute_metrics(y[is_tune], pred[is_tune])["f1"]
        if f1 > best_f1:
            best_key, best_f1 = key, f1
    return best_key, best_f1


def main() -> None:
    parser = argparse.ArgumentParser(description="Bước 8: bộ phát hiện lai + chỉ số vận hành")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths, hcfg, sel = cfg["paths"], cfg["hybrid"], cfg["select"]
    results_dir = paths["results_dir"]
    replica_event = load_json(paths["event_index"])[hcfg["replica_event"]]

    context = pd.read_pickle(paths["block_context"])
    train = pd.concat([pd.read_pickle(paths["splits_dir"] / f"{n}.pkl") for n in ("train", "val")]).merge(context, on="BlockId")
    test, is_tune, tune_mode = load_test_with_tune_mask(paths["splits_dir"], sel)
    df = test.merge(context, on="BlockId", how="left").merge(pd.read_pickle(results_dir / "deeplog_ranks.pkl"), on="BlockId", how="left")
    assert len(df) == len(test) and df["EventTimes"].notna().all() and df["Ranks"].notna().all()
    df["t0"] = df["EventTimes"].map(lambda t: t[0])
    df["t_end"] = df["EventTimes"].map(lambda t: t[-1])
    y, t0 = df["Label"].to_numpy(), df["t0"].to_numpy(dtype=float)
    max_rank = df["Ranks"].map(lambda r: int(r.max()) if len(r) else 0).to_numpy()

    profile = learn_normal_profile(train, replica_event)
    k, deadline = profile["replica_count"], profile["deadline_sec"]
    logger.info("Bất biến học từ %d session Normal: %d EventId đã thấy | cần %d lần %s trong %.0f giây",
                len(train), len(profile["seen_events"]), k, hcfg["replica_event"], deadline)

    # ---- các thành phần
    unseen = unseen_alert_times(df, profile["seen_events"])
    t_kth = np.array([time_to_kth(s, t, replica_event, k) for s, t in zip(df["EventSequence"], df["EventTimes"])])

    stream_end = float(df["t_end"].max())

    def completion(margin: float) -> np.ndarray:
        # cảnh báo chỉ phát được nếu deadline rơi trước khi log kết thúc
        fire = (t_kth > deadline * margin) & (t0 + deadline * margin <= stream_end)
        return np.where(fire, t0 + deadline * margin, np.inf)

    margins, g_values = hcfg["deadline_margins"], list(range(1, hcfg["max_top_g"] + 1))
    # ---- chọn tham số trên tune
    g_deeplog, _ = select_on_tune({g: (max_rank > g).astype(int) for g in g_values}, y, is_tune)
    m_rules, _ = select_on_tune({m: (np.isfinite(unseen) | np.isfinite(completion(m))).astype(int) for m in margins}, y, is_tune)
    rules_base = {m: np.isfinite(unseen) | np.isfinite(completion(m)) for m in margins}
    (m_hybrid, g_hybrid), _ = select_on_tune(
        {(m, g): (rules_base[m] | (max_rank > g)).astype(int) for m, g in itertools.product(margins, g_values)}, y, is_tune)
    logger.info("Tham số chọn trên tune: DeepLog g=%d | Rules margin=%s | Hybrid margin=%s, g=%d",
                g_deeplog, m_rules, m_hybrid, g_hybrid)

    detectors = {
        "deeplog": deeplog_alert_times(df, g_deeplog),
        "rule_unseen": unseen,
        "rule_completion": completion(m_rules),
        "rules": np.minimum(unseen, completion(m_rules)),
    }
    detectors["hybrid"] = np.minimum.reduce([unseen, completion(m_hybrid), deeplog_alert_times(df, g_hybrid)])

    # ---- phân tích phụ: rủi ro theo datanode (gate DeepLog g thấp hơn khi rủi ro cao)
    risk = node_risk(df, detectors["hybrid"], hcfg["risk_window_sec"])
    gated = {(thr, g): np.isfinite(detectors["hybrid"]) | ((risk > thr) & (max_rank > g))
             for thr, g in itertools.product(hcfg["risk_thresholds"], g_values) if g < g_hybrid}
    (thr_risk, g_risk), _ = select_on_tune({key: p.astype(int) for key, p in gated.items()}, y, is_tune)
    detectors["hybrid_node_risk"] = np.where(
        (risk > thr_risk) & (max_rank > g_risk), np.minimum(detectors["hybrid"], deeplog_alert_times(df, g_risk)), detectors["hybrid"])

    # ---- đánh giá trên eval
    span_hours = (df["t_end"].max() - df["t0"].min()) / 3600
    eval_hours = span_hours * (1 - sel["tune_ratio"])  # FP của eval quy về cả luồng log
    ev = ~is_tune
    table = {name: ops_metrics(y[ev], alert[ev], t0[ev], eval_hours) for name, alert in detectors.items()}
    group_unseen = unseen_event_mask(df["EventSequence"], pd.read_pickle(paths["splits_dir"] / "train.pkl")["EventSequence"])
    for name, alert in detectors.items():
        table[name].update(recall_by_unseen(y[ev], np.isfinite(alert[ev]).astype(int), group_unseen[ev]))
    end_of_session = df["t_end"].to_numpy(dtype=float)
    session_delay = (end_of_session - t0)[ev & (y == 1)]
    unalerted = ~np.isfinite(detectors["hybrid"])
    summary = {
        "meta": run_metadata(cfg),
        "deeplog_source": load_ranks_source(results_dir, cfg),
        "tune_mode": tune_mode,
        "profile": {"num_seen_events": len(profile["seen_events"]), "replica_event": hcfg["replica_event"],
                    "replica_count": k, "deadline_sec": deadline},
        "params": {"deeplog": {"top_g": g_deeplog}, "rules": {"deadline_margin": m_rules},
                   "hybrid": {"deadline_margin": m_hybrid, "top_g": g_hybrid},
                   "hybrid_node_risk": {"risk_threshold": thr_risk, "top_g_when_risky": g_risk,
                                        "risk_window_sec": hcfg["risk_window_sec"]}},
        "eval": table,
        "node_risk_auc": {"all_sessions": float(roc_auc_score(y, risk)),
                          "not_alerted_by_hybrid": float(roc_auc_score(y[unalerted], risk[unalerted]))
                          if len(np.unique(y[unalerted])) == 2 else None},
        "end_of_session_latency": {"median_sec": float(np.median(session_delay)),
                                   "p90_sec": float(np.percentile(session_delay, 90))},
        "stream_hours": span_hours,
    }
    ms_path = results_dir / f"model_selection_{tune_mode}.json"
    if ms_path.exists():
        ms = load_json(ms_path)
        for name in ("pca", "iforest"):
            if name in ms:
                m = dict(ms[name]["eval"])
                m.update({"fp_per_hour": m["fp"] / eval_hours, "latency_median_sec": summary["end_of_session_latency"]["median_sec"],
                          "latency_p90_sec": summary["end_of_session_latency"]["p90_sec"], "params": ms[name]["params"]})
                table[name] = m
        # Tách riêng PCA học trên cùng dữ liệu với DeepLog (train_normal) để so sánh công bằng trên bảng và biểu đồ
        pca_norm = ms.get("by_fit_on", {}).get("pca", {}).get("train_normal")
        if pca_norm and ms.get("pca", {}).get("params", {}).get("fit_on") != "train_normal":
            m_norm = dict(pca_norm["eval"])
            m_norm.update({
                "fp_per_hour": m_norm["fp"] / eval_hours,
                "latency_median_sec": summary["end_of_session_latency"]["median_sec"],
                "latency_p90_sec": summary["end_of_session_latency"]["p90_sec"],
                "params": pca_norm["params"],
            })
            table["pca_train_normal"] = m_norm
    save_results(summary, results_dir / "hybrid_detection.json")

    # ---- drift: FPR và recall theo khoảng thời gian bắt đầu block (toàn bộ test)
    bins = (t0 // (hcfg["drift_bin_hours"] * 3600)).astype(int)
    rows = []
    for b in np.unique(bins):
        sel_b = bins == b
        row = {"bin_start_hour": b * hcfg["drift_bin_hours"], "sessions": int(sel_b.sum()), "anomalies": int(y[sel_b].sum())}
        for name in ("deeplog", "rules", "hybrid"):
            pred = np.isfinite(detectors[name][sel_b])
            normal, anomaly = y[sel_b] == 0, y[sel_b] == 1
            row[f"{name}_fpr"] = float(pred[normal].mean()) if normal.any() else np.nan
            row[f"{name}_recall"] = float(pred[anomaly].mean()) if anomaly.any() else np.nan
        rows.append(row)
    pd.DataFrame(rows).to_csv(results_dir / "hybrid_drift.csv", index=False)

    # ---- độ nhạy của luật completion theo margin (chỉ luật completion, trên eval)
    rows = [{"deadline_margin": m, "deadline_sec": deadline * m,
             **ops_metrics(y[ev], completion(m)[ev], t0[ev], eval_hours)} for m in margins]
    pd.DataFrame(rows).to_csv(results_dir / "hybrid_deadline.csv", index=False)

    pd.DataFrame({"BlockId": df["BlockId"], "Label": y, "is_tune": is_tune, "t0": t0, "t_end": end_of_session,
                  "risk": risk, **{f"alert_{n}": a for n, a in detectors.items()}}).to_pickle(results_dir / "hybrid_alerts.pkl")

    logger.info("Kết quả trên tập eval (%d session, %d bất thường):", int(ev.sum()), int(y[ev].sum()))
    for name, m in table.items():
        logger.info("%-17s P %.3f R %.3f F1 %.3f | FP/giờ %6.1f | trễ trung vị %s s",
                    name, m["precision"], m["recall"], m["f1"], m["fp_per_hour"], m.get("latency_median_sec"))
    logger.info("AUC rủi ro datanode: %s", summary["node_risk_auc"])


if __name__ == "__main__":
    main()
