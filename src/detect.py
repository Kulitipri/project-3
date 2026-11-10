"""
BƯỚC 5 — Phát hiện session bất thường trên tập test bằng luật top-g của DeepLog.

Luật top-g: với mỗi cửa sổ w sự kiện, model dự đoán phân phối sự kiện kế tiếp.
Nếu sự kiện thật KHÔNG nằm trong g sự kiện có xác suất cao nhất → session bị đánh dấu bất thường.

Input : outputs/models/deeplog_best.pt
        data/processed/splits/test.pkl
Output: outputs/results/deeplog_metrics_g{g}.json
        outputs/results/deeplog_predictions_g{g}.pkl  (dùng cho phân tích lỗi ở evaluate.py)

Chạy:
    python -m src.detect --config configs/config.yaml
    python -m src.detect --top-g 5 --max-sessions 20000     # thử g khác, chạy trên tập con
"""
import argparse

import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm

from src.config import load_config
from src.dataset import make_windows
from src.evaluate import compute_metrics, save_results
from src.model import DeepLog
from src.utils import get_device, get_logger

logger = get_logger(__name__)


@torch.no_grad()
def is_session_anomalous(
    model: nn.Module,
    sequence: list[int],
    window_size: int,
    top_g: int,
    device: torch.device,
) -> bool:
    """
    Kiểm tra 1 session có bất thường hay không.

    Input : model       — DeepLog đã load trọng số, đang ở chế độ eval()
            sequence    — list chỉ số sự kiện của 1 session
            window_size — độ dài cửa sổ w (phải giống lúc train)
            top_g       — tham số g
    Output: True nếu có ít nhất 1 vị trí mà sự kiện thật nằm ngoài top-g dự đoán; ngược lại False.
    Toàn bộ cửa sổ của session được đưa vào model trong 1 lần forward (1 batch).
    Edge case:
      - Session ngắn hơn window_size → make_windows() đã pad, không cần xử lý thêm.
      - Session rỗng → không có cửa sổ nào → coi là Normal.
    """
    windows, targets = make_windows(sequence, window_size)
    if not windows:
        return False
    X = torch.tensor(windows, dtype=torch.long, device=device)
    y = torch.tensor(targets, dtype=torch.long, device=device)
    logits = model(X)
    top = torch.topk(logits, k=min(top_g, logits.size(1)), dim=1).indices  # (num_windows, g)
    hit = (top == y.unsqueeze(1)).any(dim=1)
    return not bool(hit.all())


def predict_sessions(
    model: nn.Module,
    sequences: list[list[int]],
    window_size: int,
    top_g: int,
    device: torch.device,
) -> np.ndarray:
    """
    Dự đoán nhãn cho nhiều session.

    Output: np.ndarray kiểu int, giá trị 0 (Normal) hoặc 1 (Anomaly), cùng độ dài với sequences.
    Lưu ý hiệu năng: tập test HDFS có hàng trăm nghìn session nhưng nhiều session có
    EventSequence giống hệt nhau → chỉ dự đoán 1 lần cho mỗi chuỗi duy nhất rồi ánh xạ lại.
    """
    model.eval()
    keys = [tuple(seq) for seq in sequences]
    unique = list(dict.fromkeys(keys))
    logger.info("Số chuỗi sự kiện duy nhất: %d / %d session", len(unique), len(keys))

    result = {
        seq: int(is_session_anomalous(model, list(seq), window_size, top_g, device))
        for seq in tqdm(unique, desc="Detect")
    }
    return np.array([result[k] for k in keys], dtype=int)


def load_model(checkpoint_path, device: torch.device) -> DeepLog:
    """Load checkpoint đã lưu ở train.py và trả về model ở chế độ eval()."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    model_cfg = ckpt["model_cfg"]
    model = DeepLog(
        num_keys=ckpt["num_keys"],
        embedding_dim=model_cfg["embedding_dim"],
        hidden_size=model_cfg["hidden_size"],
        num_layers=model_cfg["num_layers"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Bước 5: phát hiện bất thường bằng top-g")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--top-g", type=int, default=None, help="Ghi đè detect.top_g trong config")
    parser.add_argument("--max-sessions", type=int, default=None, help="Chỉ chạy trên N session đầu (thử nhanh)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = cfg["paths"]
    top_g = args.top_g or cfg["detect"]["top_g"]
    window_size = cfg["model"]["window_size"]

    device = get_device(cfg["train"]["device"])
    model = load_model(paths["model_dir"] / "deeplog_best.pt", device)

    test_df = pd.read_pickle(paths["splits_dir"] / "test.pkl")
    if args.max_sessions:
        test_df = test_df.sample(n=min(args.max_sessions, len(test_df)), random_state=cfg["seed"])
    logger.info("Số session test: %d | top_g=%d | device=%s", len(test_df), top_g, device)

    y_pred = predict_sessions(model, test_df["EventSequence"].tolist(), window_size, top_g, device)
    metrics = compute_metrics(test_df["Label"].to_numpy(), y_pred)
    metrics.update({"top_g": top_g, "num_sessions": int(len(test_df))})

    save_results(metrics, paths["results_dir"] / f"deeplog_metrics_g{top_g}.json")
    test_df.assign(Pred=y_pred).to_pickle(paths["results_dir"] / f"deeplog_predictions_g{top_g}.pkl")
    logger.info("Kết quả: %s", metrics)


if __name__ == "__main__":
    main()
