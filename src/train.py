"""
BƯỚC 4 — Train model DeepLog trên các session Normal.

Input : data/processed/splits/train.pkl, val.pkl
        data/processed/event_index.json
Output: outputs/models/deeplog_best.pt      (checkpoint có val loss thấp nhất)
        outputs/results/train_history.json  (loss theo từng epoch)

Chạy:
    python -m src.train --config configs/config.yaml
"""
import argparse
import math

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.config import load_config
from src.dataset import DeepLogWindowDataset
from src.model import DeepLog
from src.utils import ensure_dir, get_device, get_logger, load_json, save_json, set_seed

logger = get_logger(__name__)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """
    Train model qua toàn bộ train loader 1 lần.

    Output: loss trung bình của epoch (float)
    Các bước cho mỗi batch: chuyển X, y sang device → forward → tính loss
                            → xoá gradient cũ → backward → optimizer.step()
    Lưu ý:
      - Gọi model.train() ở đầu hàm.
      - Cộng dồn loss.item() * batch_size rồi chia cho tổng số mẫu để ra loss trung bình đúng.
    """
    model.train()
    total_loss, total = 0.0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        loss = criterion(model(X), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
        total += len(y)
    return total_loss / max(total, 1)


@torch.no_grad()
def evaluate_loss(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """
    Tính loss trung bình trên val loader, KHÔNG cập nhật trọng số.

    Lưu ý: gọi model.eval() ở đầu hàm; decorator @torch.no_grad() đã tắt tính gradient.
    """
    model.eval()
    total_loss, total = 0.0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        total_loss += criterion(model(X), y).item() * len(y)
        total += len(y)
    return total_loss / max(total, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Bước 4: train DeepLog")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths, model_cfg, train_cfg = cfg["paths"], cfg["model"], cfg["train"]

    set_seed(cfg["seed"])
    device = get_device(train_cfg["device"])
    logger.info("Device: %s", device)

    event_index = load_json(paths["event_index"])
    num_keys = len(event_index) + 1  # +1 cho PAD = 0

    train_df = pd.read_pickle(paths["splits_dir"] / "train.pkl")
    val_df = pd.read_pickle(paths["splits_dir"] / "val.pkl")

    train_ds = DeepLogWindowDataset(train_df["EventSequence"].tolist(), model_cfg["window_size"])
    val_ds = DeepLogWindowDataset(val_df["EventSequence"].tolist(), model_cfg["window_size"])
    logger.info("Số cặp (window, target): train=%d | val=%d", len(train_ds), len(val_ds))

    # num_workers=0: tránh lỗi multiprocessing trên macOS; dữ liệu nhỏ nên không cần worker
    train_loader = DataLoader(train_ds, batch_size=train_cfg["batch_size"], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=train_cfg["batch_size"], shuffle=False, num_workers=0)

    model = DeepLog(
        num_keys=num_keys,
        embedding_dim=model_cfg["embedding_dim"],
        hidden_size=model_cfg["hidden_size"],
        num_layers=model_cfg["num_layers"],
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["lr"])

    model_dir = ensure_dir(paths["model_dir"])
    best_val, epochs_no_improve = math.inf, 0
    history = []

    for epoch in range(1, train_cfg["epochs"] + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate_loss(model, val_loader, criterion, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        logger.info("Epoch %3d | train_loss=%.4f | val_loss=%.4f", epoch, train_loss, val_loss)

        if val_loss < best_val:
            best_val, epochs_no_improve = val_loss, 0
            torch.save(
                {"model_state": model.state_dict(), "num_keys": num_keys, "model_cfg": model_cfg},
                model_dir / "deeplog_best.pt",
            )
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= train_cfg["patience"]:
                logger.info("Early stopping tại epoch %d", epoch)
                break

    save_json(history, paths["results_dir"] / "train_history.json")
    logger.info("Val loss tốt nhất: %.4f — checkpoint: %s", best_val, model_dir / "deeplog_best.pt")


if __name__ == "__main__":
    main()
