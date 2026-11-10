"""
Tính chỉ số đánh giá và phân tích lỗi.

- compute_metrics(), save_results(): được detect.py và baselines.py gọi.
- score_metrics(): chỉ số không phụ thuộc ngưỡng, được baselines.py gọi.
- structural_anomaly_mask(), recall_by_group(): recall theo 2 nhóm bất thường (model_selection.py gọi).
- error_analysis(): tìm điểm chung của các session bị dự đoán sai.

Chạy phân tích lỗi (sau khi đã chạy detect.py):
    python -m src.evaluate --config configs/config.yaml --top-g 9
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)

from src.config import load_config
from src.utils import get_logger, load_json, save_json

logger = get_logger(__name__)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Tính precision, recall, F1 và ma trận nhầm lẫn (lớp dương = Anomaly = 1).

    TP: bất thường bị bắt đúng | FP: bình thường bị báo nhầm
    FN: bất thường bị bỏ sót    | TN: bình thường được nhận đúng
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def score_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict:
    """
    Chỉ số không phụ thuộc ngưỡng, tính từ điểm bất thường (càng lớn càng bất thường).

    roc_auc           — xác suất một session Anomaly có điểm cao hơn một session Normal (0,5 = đoán mò).
                        AUC thấp: mô hình không xếp hạng được, chỉnh ngưỡng cũng không cứu được.
    average_precision — diện tích dưới đường precision–recall; đoán mò chỉ được bằng tỉ lệ Anomaly.
    """
    return {
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "average_precision": float(average_precision_score(y_true, scores)),
    }


def structural_anomaly_mask(sequences: list[list[int]], sessions: pd.DataFrame) -> np.ndarray:
    """
    Đánh dấu session có "dấu hiệu cấu trúc" — cách chia 2 nhóm bất thường của notebook EDA (Phần 5).

    Input : sequences — các session cần đánh dấu (list chỉ số sự kiện)
            sessions  — toàn bộ session CÓ NHÃN (sessions.pkl), dùng để lấy mốc tham chiếu từ session Normal
    Output: mảng bool; True nếu session thoả ít nhất 1 điều kiện:
              - chứa sự kiện không xuất hiện trong bất kỳ session Normal nào
              - thiếu một sự kiện có mặt trong MỌI session Normal
              - ngắn hơn session Normal ngắn nhất
    Mốc tham chiếu dùng nhãn của toàn bộ dữ liệu, nên chỉ dùng để PHÂN TÍCH lỗi theo nhóm, không dùng để phát hiện.
    Nhóm còn lại (False) chỉ gồm sự kiện bình thường, chỉ khác ở thứ tự hoặc số lần xuất hiện: nhóm này cho biết
    mô hình chuỗi (DeepLog) có lợi thế hơn phương pháp đếm hay không.
    """
    normal = [set(seq) for seq in sessions.loc[sessions["Label"] == 0, "EventSequence"]]
    seen_in_normal = set().union(*normal)
    always_in_normal = set.intersection(*normal)
    min_normal_len = int(sessions.loc[sessions["Label"] == 0, "EventSequence"].str.len().min())
    return np.array([
        bool(set(seq) - seen_in_normal) or not always_in_normal <= set(seq) or len(seq) < min_normal_len
        for seq in sequences
    ], dtype=bool)


def recall_by_group(y_true: np.ndarray, y_pred: np.ndarray, structural: np.ndarray) -> dict:
    """Recall riêng cho 2 nhóm bất thường: có dấu hiệu cấu trúc / chỉ khác thứ tự hoặc số lần (structural_anomaly_mask)."""
    result = {}
    for name, mask in (("structural", structural), ("sequence", ~structural)):
        anomaly = (y_true == 1) & mask
        result[f"anomalies_{name}"] = int(anomaly.sum())
        result[f"recall_{name}"] = float(y_pred[anomaly].mean()) if anomaly.any() else float("nan")
    return result


def save_results(metrics: dict, path) -> None:
    """Lưu dict kết quả ra file JSON."""
    save_json(metrics, path)
    logger.info("Đã lưu kết quả: %s", path)


def error_analysis(pred_df: pd.DataFrame, event_index: dict[str, int], top_n: int = 10) -> pd.DataFrame:
    """
    Tìm điểm chung của các session bị dự đoán sai.

    Input : pred_df     — DataFrame cột BlockId, EventSequence, Label, Pred (từ detect.py)
            event_index — dict EventId → số nguyên (để đổi ngược số → EventId khi in kết quả)
            top_n       — số dòng kết quả cần giữ
    Output: DataFrame cột:
            ErrorType      — "FP" | "FN"
            EventId        — sự kiện
            Count          — tổng số lần sự kiện xuất hiện trong nhóm lỗi
            SessionShare   — tỉ lệ session trong nhóm lỗi có chứa sự kiện
            ReferenceShare — tỉ lệ tương ứng ở nhóm dự đoán đúng cùng nhãn (FP ↔ TN, FN ↔ TP)
            ShareDiff      — SessionShare - ReferenceShare; lớn → sự kiện đặc trưng cho nhóm lỗi
            Giữ top_n sự kiện có ShareDiff lớn nhất cho mỗi loại lỗi.
    Ngoài ra logger.info so sánh độ dài session của nhóm lỗi với nhóm đúng tương ứng
    (session FP dài hay ngắn hơn session Normal đúng?).
    """
    index_to_event = {idx: event_id for event_id, idx in event_index.items()}
    index_to_event[0] = "PAD"
    groups = {
        "TP": pred_df[(pred_df["Label"] == 1) & (pred_df["Pred"] == 1)],
        "FP": pred_df[(pred_df["Label"] == 0) & (pred_df["Pred"] == 1)],
        "FN": pred_df[(pred_df["Label"] == 1) & (pred_df["Pred"] == 0)],
        "TN": pred_df[(pred_df["Label"] == 0) & (pred_df["Pred"] == 0)],
    }

    def event_stats(df: pd.DataFrame) -> pd.DataFrame:
        exploded = df[["EventSequence"]].reset_index(drop=True).reset_index().explode("EventSequence")
        count = exploded["EventSequence"].value_counts()
        share = exploded.drop_duplicates()["EventSequence"].value_counts() / max(len(df), 1)
        return pd.DataFrame({"Count": count, "SessionShare": share})

    rows = []
    for error, reference in (("FP", "TN"), ("FN", "TP")):
        err_df, ref_df = groups[error], groups[reference]
        err_len, ref_len = err_df["EventSequence"].str.len(), ref_df["EventSequence"].str.len()
        logger.info(
            "%s: %d session, độ dài trung vị %.0f (trung bình %.1f) | %s: %d session, trung vị %.0f (trung bình %.1f)",
            error, len(err_df), err_len.median(), err_len.mean(),
            reference, len(ref_df), ref_len.median(), ref_len.mean(),
        )
        if err_df.empty:
            continue
        stats = event_stats(err_df)
        stats["ReferenceShare"] = event_stats(ref_df)["SessionShare"].reindex(stats.index).fillna(0.0)
        stats["ShareDiff"] = stats["SessionShare"] - stats["ReferenceShare"]
        stats = stats.sort_values(["ShareDiff", "Count"], ascending=False).head(top_n)
        stats.insert(0, "EventId", [index_to_event.get(int(i), str(i)) for i in stats.index])
        stats.insert(0, "ErrorType", error)
        rows.append(stats.reset_index(drop=True))

    columns = ["ErrorType", "EventId", "Count", "SessionShare", "ReferenceShare", "ShareDiff"]
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=columns)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phân tích lỗi dự đoán của DeepLog")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--top-g", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = cfg["paths"]
    top_g = args.top_g or cfg["detect"]["top_g"]

    pred_df = pd.read_pickle(paths["results_dir"] / f"deeplog_predictions_g{top_g}.pkl")
    event_index = load_json(paths["event_index"])

    structural = structural_anomaly_mask(pred_df["EventSequence"].tolist(), pd.read_pickle(paths["sessions"]))
    logger.info("Recall theo nhóm bất thường (g=%d): %s", top_g,
                recall_by_group(pred_df["Label"].to_numpy(), pred_df["Pred"].to_numpy(), structural))

    summary = error_analysis(pred_df, event_index)
    out_path = paths["results_dir"] / f"error_analysis_g{top_g}.csv"
    summary.to_csv(out_path, index=False)
    logger.info("Đã lưu phân tích lỗi: %s\n%s", out_path, summary.to_string())


if __name__ == "__main__":
    main()
