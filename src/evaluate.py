"""
Tính chỉ số đánh giá (lớp dương = Anomaly = 1).

- compute_metrics(), save_results(): được baselines.py gọi.
- score_metrics(): chỉ số không phụ thuộc ngưỡng, được baselines.py gọi.
"""
import numpy as np
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)

from src.utils import get_logger, save_json

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


def save_results(metrics: dict, path) -> None:
    """Lưu dict kết quả ra file JSON."""
    save_json(metrics, path)
    logger.info("Đã lưu kết quả: %s", path)
