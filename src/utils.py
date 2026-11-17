"""Các hàm tiện ích dùng chung: seed, chọn device, logger, đọc/ghi JSON, nguồn gốc kết quả."""
import json
import logging
import random
import subprocess
from datetime import datetime
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


def _git(*args: str) -> str | None:
    """Chạy 1 lệnh git ở thư mục gốc project; None nếu không có git / không phải repo."""
    try:
        out = subprocess.run(["git", *args], cwd=Path(__file__).resolve().parent.parent,
                             capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run_metadata(cfg: dict) -> dict:
    """
    Nguồn gốc của một file kết quả: thời điểm chạy, commit git, file config và TOÀN BỘ config đã dùng.

    git_dirty = True nghĩa là code/config đang khác commit (kể cả file chưa commit trong src/, configs/, scripts/),
    tức commit một mình không đủ để tái lập kết quả. Ghi kèm vào mọi file kết quả để không còn tình trạng
    số liệu của cấu hình cũ (vd batch_size khác) nằm lẫn với số liệu mới mà không ai biết.
    """
    project_root = Path(__file__).resolve().parent.parent
    config = {key: value for key, value in cfg.items() if key not in ("paths", "config_file")}
    config["paths"] = {key: str(Path(value).relative_to(project_root)) if Path(value).is_relative_to(project_root)
                       else str(value) for key, value in cfg.get("paths", {}).items()}
    status = _git("status", "--porcelain", "--", "src", "configs", "scripts")
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "git_commit": _git("rev-parse", "--short", "HEAD"),
        "git_dirty": None if status is None else bool(status),
        "config_file": cfg.get("config_file"),
        "config": config,
    }
