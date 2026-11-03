"""Đọc file cấu hình YAML và chuẩn hoá mọi đường dẫn về thư mục gốc của project."""
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(config_path: str | Path = "configs/config.yaml") -> dict:
    """
    Đọc configs/config.yaml thành dict.

    Mọi giá trị trong mục `paths` được đổi thành pathlib.Path tuyệt đối
    (tính từ PROJECT_ROOT), nên chạy script từ thư mục nào cũng không sai đường dẫn.
    """
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg["paths"] = {key: PROJECT_ROOT / value for key, value in cfg["paths"].items()}
    return cfg
