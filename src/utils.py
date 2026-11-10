"""Các hàm tiện ích dùng chung: seed, chọn device, logger, đọc/ghi JSON."""
import json
import logging
import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Cố định seed cho random, numpy, torch để kết quả chạy lại giống nhau."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device(preference: str = "auto") -> torch.device:
    """
    Chọn device cho PyTorch.

    preference = "auto": ưu tiên MPS (GPU Apple) → CUDA → CPU.
    preference = "mps" : dùng MPS, nếu máy không có MPS thì lùi về CPU.
    preference = "cpu" : luôn dùng CPU (dùng khi MPS báo lỗi op của LSTM).
    """
    if preference == "cpu":
        return torch.device("cpu")
    if preference in ("auto", "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if preference in ("auto", "cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    if preference in ("mps", "cuda"):
        logging.getLogger(__name__).warning("Không có %s, chuyển sang CPU.", preference)
    return torch.device("cpu")


def get_logger(name: str) -> logging.Logger:
    """Tạo logger in ra terminal với định dạng thời gian | mức độ | tên module | nội dung."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s", "%H:%M:%S")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def ensure_dir(path: str | Path) -> Path:
    """Tạo thư mục (kể cả thư mục cha) nếu chưa có, trả về Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(obj, path: str | Path) -> None:
    """Ghi obj ra file JSON (UTF-8, thụt lề 2 khoảng trắng)."""
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str | Path):
    """Đọc file JSON."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)
