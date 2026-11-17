"""
Ngữ cảnh của từng block lấy từ log thô: thời điểm của từng sự kiện và các datanode lưu block.

DeepLog chỉ nhìn chuỗi EventId của từng block riêng lẻ. File này bổ sung 2 loại thông tin
mà chuỗi EventId đã bỏ mất khi Drain3 thay IP / thời gian bằng <*>:
  - EventTimes : thời điểm (giây, tính từ dòng log đầu tiên) của từng dòng nhắc tới block,
                 cùng thứ tự và cùng độ dài với EventSequence ở bước 2
                 → dùng để đo độ trễ phát hiện (detection latency).
  - Nodes      : tập datanode (IP có cổng :50010) xuất hiện trong các dòng của block
                 → dùng để nối các block dùng chung datanode (ngữ cảnh chéo session).

Input : data/raw/HDFS_v1/HDFS.log
Output: data/interim/block_context.pkl — DataFrame cột BlockId, EventTimes (np.ndarray int32), Nodes (tuple[str])

Chạy:
    python -m src.block_context --config configs/config.yaml
"""
import argparse
import re
from array import array
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.block_stats import TOTAL_LINES_HDFS_V1
from src.build_sequences import BLOCK_ID_PATTERN
from src.config import load_config
from src.utils import ensure_dir, get_logger

logger = get_logger(__name__)

DATANODE_PATTERN = re.compile(r"(\d+\.\d+\.\d+\.\d+):50010")


def compute_block_context(raw_path: Path, max_lines: int | None = None) -> pd.DataFrame:
    """
    Quét log thô đúng 1 lần, trả về DataFrame cột BlockId, EventTimes, Nodes.

    Một dòng nhắc tới nhiều block thì được tính cho từng block (giống build_sequences.py),
    nhờ vậy len(EventTimes) == len(EventSequence) với mọi block.
    Thời gian: "081109 203518" → số giây kể từ dòng đầu tiên của file (cache theo chuỗi vì
    hàng nghìn dòng dùng chung một giây).
    """
    times: dict[str, array] = {}
    nodes: dict[str, set] = {}
    ts_cache: dict[str, int] = {}
    t0 = None

    with open(raw_path, encoding="utf-8", errors="replace") as f:
        for line_id, line in enumerate(tqdm(f, total=max_lines or TOTAL_LINES_HDFS_V1, desc="Quét ngữ cảnh"), 1):
            if max_lines and line_id > max_lines:
                break
            blocks = BLOCK_ID_PATTERN.findall(line)
            if not blocks:
                continue
            ts = line[:13]
            sec = ts_cache.get(ts)
            if sec is None:
                dt = datetime.strptime(ts, "%y%m%d %H%M%S")
                t0 = t0 or dt
                sec = ts_cache[ts] = int((dt - t0).total_seconds())
            line_nodes = DATANODE_PATTERN.findall(line)
            for block in dict.fromkeys(blocks):
                if block not in times:
                    times[block] = array("i")
                    nodes[block] = set()
                times[block].append(sec)
                nodes[block].update(line_nodes)

    logger.info("Mốc thời gian 0 = %s", t0)
    return pd.DataFrame({
        "BlockId": list(times),
        "EventTimes": [np.frombuffer(a, dtype=np.int32).copy() for a in times.values()],
        "Nodes": [tuple(sorted(nodes[b])) for b in times],
    })


def main() -> None:
    parser = argparse.ArgumentParser(description="Thời điểm sự kiện + datanode của từng block")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = cfg["paths"]

    context = compute_block_context(paths["raw_log"], cfg["parse"]["max_lines"])
    ensure_dir(paths["block_context"].parent)
    context.to_pickle(paths["block_context"])

    num_nodes = context["Nodes"].str.len()
    logger.info("Số block: %d | số datanode khác nhau: %d", len(context), len({n for ns in context["Nodes"] for n in ns}))
    logger.info("Số datanode mỗi block:\n%s", num_nodes.describe().to_string())
    logger.info("Đã lưu: %s", paths["block_context"])


if __name__ == "__main__":
    main()
