"""
Tính "thứ hạng" của sự kiện thật tại từng vị trí của mỗi session test — dạng điểm liên tục của DeepLog.

detect.py chỉ trả về 0/1 cho một giá trị g. Ở đây lưu lại cho mỗi cửa sổ: sự kiện thật đứng thứ mấy
trong dự đoán của model (rank = 1 nghĩa là model đoán đúng nhất). Từ mảng rank này suy ra được
kết quả của MỌI g mà không phải chạy lại model:
    session bất thường với g  ⇔  max(Ranks) > g
    vị trí cảnh báo đầu tiên   =  vị trí đầu tiên có Ranks > g   (dùng để đo độ trễ phát hiện)

Vị trí trong session của từng rank (TargetPos): với session dài hơn window_size, cửa sổ thứ k dự đoán
sự kiện thứ window_size + k; với session ngắn (đã pad), chỉ có 1 cửa sổ dự đoán sự kiện cuối cùng.

Input : outputs/models/deeplog_best.pt, data/processed/splits/test.pkl
Output: outputs/results/deeplog_ranks.pkl — DataFrame cột BlockId, Ranks (np.ndarray int16), TargetPos (np.ndarray int32)

Chạy:
    python -m src.deeplog_scores --config configs/config.yaml
"""
import argparse

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.config import load_config
from src.dataset import make_windows
from src.detect import load_model
from src.utils import get_device, get_logger

logger = get_logger(__name__)


@torch.no_grad()
def sequence_ranks(model, sequence: list[int], window_size: int, device: torch.device) -> np.ndarray:
    """Rank (bắt đầu từ 1) của sự kiện thật trong dự đoán của model, cho từng cửa sổ của session."""
    windows, targets = make_windows(sequence, window_size)
    if not windows:
        return np.zeros(0, dtype=np.int16)
    logits = model(torch.tensor(windows, dtype=torch.long, device=device))
    y = torch.tensor(targets, dtype=torch.long, device=device)
    target_logit = logits.gather(1, y.unsqueeze(1))
    return ((logits > target_logit).sum(dim=1) + 1).cpu().numpy().astype(np.int16)


def target_positions(length: int, window_size: int) -> np.ndarray:
    """Vị trí (0-based) trong session của sự kiện mà từng cửa sổ dự đoán — khớp với make_windows()."""
    if length == 0:
        return np.zeros(0, dtype=np.int32)
    if length <= window_size:
        return np.array([length - 1], dtype=np.int32)
    return np.arange(window_size, length, dtype=np.int32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tính rank của sự kiện thật cho mọi session test")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = cfg["paths"]
    window_size = cfg["model"]["window_size"]
    device = get_device(cfg["train"]["device"])
    model = load_model(paths["model_dir"] / "deeplog_best.pt", device)

    test_df = pd.read_pickle(paths["splits_dir"] / "test.pkl")
    keys = test_df["EventSequence"].map(tuple)
    unique = list(dict.fromkeys(keys))
    logger.info("Số chuỗi duy nhất: %d / %d session | device=%s", len(unique), len(keys), device)
    ranks = {seq: sequence_ranks(model, list(seq), window_size, device) for seq in tqdm(unique, desc="Rank")}

    out = pd.DataFrame({
        "BlockId": test_df["BlockId"],
        "Ranks": [ranks[k] for k in keys],
        "TargetPos": [target_positions(len(k), window_size) for k in keys],
    })
    out_path = paths["results_dir"] / "deeplog_ranks.pkl"
    out.to_pickle(out_path)
    logger.info("Đã lưu: %s", out_path)


if __name__ == "__main__":
    main()
