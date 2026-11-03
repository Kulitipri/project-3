"""
BƯỚC 1 — Parse log thô HDFS thành log có cấu trúc bằng Drain3.

Input : data/raw/HDFS.log
        Mỗi dòng có dạng:
        081109 203615 148 INFO dfs.DataNode$PacketResponder: PacketResponder 1 for block blk_38865049064139660 terminating
Output: data/interim/HDFS.log_structured.csv
            cột: LineId, Date, Time, Pid, Level, Component, Content, ClusterId, EventId, EventTemplate
        data/interim/HDFS.log_templates.csv
            cột: ClusterId, EventId, EventTemplate, Occurrences

Chạy (từ thư mục ~/project-3, đã activate venv):
    python -m src.parse_logs --config configs/config.yaml
"""
import argparse
import re
from pathlib import Path

import pandas as pd
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig
from tqdm import tqdm

from src.block_stats import TOTAL_LINES_HDFS_V1
from src.config import load_config
from src.utils import ensure_dir, get_logger

logger = get_logger(__name__)


def build_header_regex(log_format: str) -> tuple[re.Pattern, list[str]]:
    """
    Chuyển chuỗi định dạng log thành regex có named group.

    Ví dụ: log_format = "<Date> <Time> <Pid> <Level> <Component>: <Content>"
    → regex khớp được "081109 203615 148 INFO dfs.DataNode$PacketResponder: PacketResponder 1 ..."
    → danh sách header = ["Date", "Time", "Pid", "Level", "Component", "Content"]
    """
    headers: list[str] = []
    regex = ""
    parts = re.split(r"(<[^<>]+>)", log_format)
    for k, part in enumerate(parts):
        if k % 2 == 0:
            # Phần chữ cố định (dấu cách, dấu ":") → escape, khoảng trắng → \s+
            literal = re.escape(part)
            regex += re.sub(r"(\\ )+", r"\\s+", literal)
        else:
            header = part.strip("<>")
            regex += f"(?P<{header}>.*?)"
            headers.append(header)
    return re.compile("^" + regex + "$"), headers


def split_header(line: str, header_regex: re.Pattern) -> dict | None:
    """
    Tách 1 dòng log thô thành các trường header và phần Content.

    Input : line         — 1 dòng log thô (đã bỏ ký tự xuống dòng ở cuối)
            header_regex — regex trả về từ build_header_regex()
    Output: dict {"Date": "081109", "Time": "203615", "Pid": "148", "Level": "INFO",
                  "Component": "dfs.DataNode$PacketResponder", "Content": "PacketResponder 1 for block ..."}
            hoặc None nếu dòng không khớp định dạng.
    Edge case:
      - Dòng rỗng hoặc dòng bị cắt dở → trả về None, KHÔNG raise lỗi.
    """
    match = header_regex.match(line.strip())
    return match.groupdict() if match else None


def create_template_miner(ini_path: Path) -> TemplateMiner:
    """Tạo TemplateMiner của Drain3 từ file configs/drain3.ini."""
    config = TemplateMinerConfig()
    config.load(str(ini_path))
    config.profiling_enabled = False
    return TemplateMiner(persistence_handler=None, config=config)


def parse_file(
    raw_path: Path,
    miner: TemplateMiner,
    header_regex: re.Pattern,
    max_lines: int | None = None,
) -> pd.DataFrame:
    """
    Đọc file log thô theo từng dòng, tách header, đưa phần Content vào Drain3.

    Input : raw_path     — đường dẫn data/raw/HDFS.log
            miner        — TemplateMiner tạo từ create_template_miner()
            header_regex — regex header từ build_header_regex()
            max_lines    — chỉ đọc N dòng đầu để chạy thử; None = đọc toàn bộ file
    Output: DataFrame, mỗi dòng log là 1 hàng, gồm các cột:
            LineId (int, bắt đầu từ 1), Date, Time, Pid, Level, Component, Content, ClusterId (int)

    Lưu ý quan trọng:
      - Gọi miner.add_log_message(content) cho từng Content; kết quả là dict.
      - Lưu result["cluster_id"], KHÔNG lưu result["template_mined"].
        Lý do: Drain3 cập nhật template liên tục; template lúc gặp dòng thứ 100 có thể khác
        template cuối cùng của cùng cluster. Template cuối cùng được lấy ở build_template_table().
      - Đọc bằng vòng for trên file handle (for line in f), không dùng f.readlines(),
        để không nạp cả file 1,5GB vào RAM cùng lúc.
      - Bọc vòng lặp bằng tqdm(...) để theo dõi tiến độ (11 triệu dòng mất vài chục phút).
      - Gom kết quả vào list các dict rồi tạo DataFrame 1 lần ở cuối (không append từng hàng vào DataFrame).
    Edge case:
      - Dòng không khớp header_regex → bỏ qua, đếm số dòng bị bỏ và logger.warning ở cuối hàm.
      - File có ký tự lỗi encoding → mở file với errors="replace".
    """
    rows: list[dict] = []
    num_skipped = 0
    with open(raw_path, encoding="utf-8", errors="replace") as f:
        for line_id, line in enumerate(tqdm(f, total=max_lines or TOTAL_LINES_HDFS_V1, desc="Parse"), 1):
            if max_lines and line_id > max_lines:
                break
            fields = split_header(line, header_regex)
            if fields is None:
                num_skipped += 1
                continue
            result = miner.add_log_message(fields["Content"])
            rows.append({"LineId": line_id, **fields, "ClusterId": result["cluster_id"]})

    if num_skipped:
        logger.warning("Bỏ qua %d dòng không khớp định dạng log", num_skipped)
    return pd.DataFrame(rows)


def build_template_table(miner: TemplateMiner) -> pd.DataFrame:
    """
    Lấy bảng template cuối cùng từ Drain3.

    Output: DataFrame cột ClusterId, EventId ("E" + ClusterId), EventTemplate, Occurrences,
            sắp xếp theo Occurrences giảm dần.
    """
    rows = [
        {
            "ClusterId": cluster.cluster_id,
            "EventId": f"E{cluster.cluster_id}",
            "EventTemplate": cluster.get_template(),
            "Occurrences": cluster.size,
        }
        for cluster in miner.drain.clusters
    ]
    return (
        pd.DataFrame(rows)
        .sort_values("Occurrences", ascending=False)
        .reset_index(drop=True)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Bước 1: parse log HDFS bằng Drain3")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = cfg["paths"]

    header_regex, _ = build_header_regex(cfg["parse"]["log_format"])
    miner = create_template_miner(paths["drain_config"])

    logger.info("Bắt đầu parse %s", paths["raw_log"])
    structured = parse_file(paths["raw_log"], miner, header_regex, cfg["parse"]["max_lines"])

    templates = build_template_table(miner)
    structured = structured.merge(
        templates[["ClusterId", "EventId", "EventTemplate"]], on="ClusterId", how="left"
    )

    ensure_dir(paths["structured_log"].parent)
    structured.to_csv(paths["structured_log"], index=False)
    templates.to_csv(paths["templates"], index=False)

    logger.info("Số dòng log đã parse: %d", len(structured))
    logger.info("Số template tìm được: %d", len(templates))
    logger.info("Đã lưu: %s và %s", paths["structured_log"], paths["templates"])


if __name__ == "__main__":
    main()
