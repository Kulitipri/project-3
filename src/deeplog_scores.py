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
        outputs/results/deeplog_ranks_meta.json — nguồn gốc: thời điểm, commit, config, checkpoint đã dùng

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
from src.detect import checkpoint_summary, config_differences, load_model
from src.utils import ensure_dir, get_device, get_logger, load_json, run_metadata, save_json

logger = get_logger(__name__)

RANKS_FILE = "deeplog_ranks.pkl"
RANKS_META_FILE = "deeplog_ranks_meta.json"  # nguồn gốc của RANKS_FILE (checkpoint nào, cấu hình nào)


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


def compute_ranks(model, test_df: pd.DataFrame, window_size: int, device: torch.device) -> pd.DataFrame:
    """
    Rank của sự kiện thật ở mọi cửa sổ cho từng session test (chuỗi trùng nhau chỉ chấm 1 lần).

    model có thể là DeepLog hoặc DeepLogEnsemble (model.py) — chỉ cần forward trả về điểm để xếp hạng.
    Output: DataFrame cột BlockId, Ranks, TargetPos
    """
    keys = test_df["EventSequence"].map(tuple)
    unique = list(dict.fromkeys(keys))
    logger.info("Số chuỗi duy nhất: %d / %d session | device=%s", len(unique), len(keys), device)
    ranks = {seq: sequence_ranks(model, list(seq), window_size, device) for seq in tqdm(unique, desc="Rank")}
    return pd.DataFrame({
        "BlockId": test_df["BlockId"].to_numpy(),
        "Ranks": [ranks[k] for k in keys],
        "TargetPos": [target_positions(len(k), window_size) for k in keys],
    })


def save_ranks(ranks: pd.DataFrame, results_dir, cfg: dict, checkpoint_metas: list[dict | None]) -> None:
    """Lưu deeplog_ranks.pkl kèm deeplog_ranks_meta.json (nguồn gốc: lần chạy này + checkpoint đã dùng)."""
    ensure_dir(results_dir)
    ranks.to_pickle(results_dir / RANKS_FILE)
    save_json({**run_metadata(cfg), "checkpoints": checkpoint_metas}, results_dir / RANKS_META_FILE)
    logger.info("Đã lưu: %s (+ %s)", results_dir / RANKS_FILE, RANKS_META_FILE)


def load_ranks_source(results_dir, cfg: dict) -> dict | None:
    """
    Đọc nguồn gốc của deeplog_ranks.pkl và CẢNH BÁO nếu checkpoint được train với cấu hình khác cfg hiện tại
    (vd ranks còn từ model batch 2048 trong khi config đã là 512). Với ensemble (nhiều checkpoint) không so seed.
    Output: {"ranks_created_at", "checkpoints": [tóm tắt từng checkpoint]} để ghi kèm kết quả; None nếu chưa có file nguồn gốc.
    """
    path = results_dir / RANKS_META_FILE
    if not path.exists():
        logger.warning("Không có %s — không rõ %s sinh từ checkpoint nào", path, RANKS_FILE)
        return None
    source = load_json(path)
    metas = source["checkpoints"]
    for i, meta in enumerate(metas):
        if diffs := config_differences(meta, cfg, check_seed=len(metas) == 1):
            logger.warning("%s: checkpoint %d được train với cấu hình KHÁC config hiện tại: %s", RANKS_FILE, i, diffs)
    return {"ranks_created_at": source["created_at"], "checkpoints": [checkpoint_summary(m) for m in metas]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Tính rank của sự kiện thật cho mọi session test")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = cfg["paths"]
    device = get_device(cfg["train"]["device"])
    model = load_model(paths["model_dir"] / "deeplog_best.pt", device)
    if diffs := config_differences(model.meta, cfg):
        logger.warning("Checkpoint được train với cấu hình KHÁC config hiện tại: %s", diffs)

    test_df = pd.read_pickle(paths["splits_dir"] / "test.pkl")
    ranks = compute_ranks(model, test_df, cfg["model"]["window_size"], device)
    save_ranks(ranks, paths["results_dir"], cfg, [model.meta])


if __name__ == "__main__":
    main()
