"""Các hàm tiện ích dùng chung: logger, tạo thư mục, đọc/ghi JSON."""
import json
import logging
from pathlib import Path


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
