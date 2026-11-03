"""
Thống kê vị trí và thời gian của từng block trong log thô (không cần Drain3).

Quét HDFS.log đúng 1 lần, với mỗi block id ghi lại:
  - FirstLine / LastLine : số thứ tự dòng (LineId, bắt đầu từ 1) đầu tiên / cuối cùng nhắc tới block
  - StartTime / EndTime  : thời điểm của 2 dòng đó
  - NumLines             : số dòng log nhắc tới block

Dùng cho:
  - validate_parsing.py : kiểm tra con số 4.855 session Normal trong 100k dòng đầu (bài báo DeepLog)
  - notebooks/01_eda_hdfs.ipynb : phân bố bất thường theo thời gian, chọn mốc cắt chronological

Input : data/raw/HDFS_v1/HDFS.log
Output: data/interim/block_stats.csv

Chạy:
    python -m src.block_stats --config configs/config.yaml
"""
import argparse
from collections import Counter
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.build_sequences import BLOCK_ID_PATTERN
from src.config import load_config
from src.utils import ensure_dir, get_logger

logger = get_logger(__name__)

TOTAL_LINES_HDFS_V1 = 11_175_629  # chỉ để tqdm hiển thị %


def compute_block_stats(raw_path: Path, max_lines: int | None = None) -> pd.DataFrame:
    """
    Quét log thô, trả về DataFrame cột BlockId, FirstLine, LastLine, StartTime, EndTime, NumLines.

    Một dòng nhắc tới nhiều block khác nhau thì được tính cho từng block.
    """
    first_line: dict[str, int] = {}
    last_line: dict[str, int] = {}
    first_ts: dict[str, str] = {}
    last_ts: dict[str, str] = {}
    num_lines: Counter = Counter()

    with open(raw_path, encoding="utf-8", errors="replace") as f:
        for line_id, line in enumerate(tqdm(f, total=max_lines or TOTAL_LINES_HDFS_V1, desc="Quét block"), 1):
            if max_lines and line_id > max_lines:
                break
            blocks = BLOCK_ID_PATTERN.findall(line)
            if not blocks:
                continue
            ts = line[:13]  # "081109 203615"
            for block in dict.fromkeys(blocks):
                if block not in first_line:
                    first_line[block] = line_id
                    first_ts[block] = ts
                last_line[block] = line_id
                last_ts[block] = ts
                num_lines[block] += 1

    df = pd.DataFrame({
        "BlockId": list(first_line),
        "FirstLine": list(first_line.values()),
        "LastLine": [last_line[b] for b in first_line],
        "StartTime": [first_ts[b] for b in first_line],
        "EndTime": [last_ts[b] for b in first_line],
        "NumLines": [num_lines[b] for b in first_line],
    })
    for col in ["StartTime", "EndTime"]:
        df[col] = pd.to_datetime(df[col], format="%y%m%d %H%M%S")
    return df


def load_block_stats(path: Path) -> pd.DataFrame:
    """Đọc lại block_stats.csv với cột thời gian đúng kiểu datetime."""
    return pd.read_csv(path, parse_dates=["StartTime", "EndTime"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Thống kê vị trí/thời gian của từng block")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = cfg["paths"]

    stats = compute_block_stats(paths["raw_log"], cfg["parse"]["max_lines"])

    ensure_dir(paths["block_stats"].parent)
    stats.to_csv(paths["block_stats"], index=False)
    logger.info("Số block: %d", len(stats))
    logger.info("Khoảng thời gian: %s → %s", stats["StartTime"].min(), stats["EndTime"].max())
    logger.info("Đã lưu: %s", paths["block_stats"])


if __name__ == "__main__":
    main()
