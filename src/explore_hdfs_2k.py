"""
TUẦN 1 — Khảo sát nhanh bộ mẫu HDFS_2k (2.000 dòng) bằng Drain3, trước khi làm với HDFS_v1 đầy đủ.

Mục đích: xem log sau khi parse trông như thế nào và kiểm tra bộ mẫu có dùng để dựng chuỗi theo block được không.
Các bước:
    1. Tách mỗi dòng thành Date, Time, Pid, Level, Component, Content bằng một regex duy nhất.
    2. Đưa Content vào Drain3 (cấu hình cơ bản: chỉ che mã block, IP, con số).
    3. So template Drain3 tìm được với đáp án của Loghub (HDFS_2k.log_structured.csv).
    4. Gom các dòng theo mã block để xem mỗi block có bao nhiêu dòng.

Input : data/raw/HDFS_2k/HDFS_2k.log, data/raw/HDFS_2k/HDFS_2k.log_structured.csv (tải từ GitHub logpai/loghub)
        configs/drain3_basic.ini
Output: outputs/results/hdfs2k_survey.json      — các con số thống kê (kể cả số dòng không khớp regex)
        outputs/results/hdfs2k_templates.csv    — template Drain3, số dòng, EventId tương ứng của Loghub
        outputs/figures/hdfs2k_template_freq.png — tần suất các template (Hình 2 báo cáo tuần 1)

Chạy (từ thư mục ~/project-3, đã activate venv):
    python -m src.explore_hdfs_2k
"""
import argparse
import re
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig

from src.config import load_config
from src.utils import ensure_dir, get_logger, save_json

logger = get_logger(__name__)

# <Date> <Time> <Pid> <Level> <Component>: <Content>
LINE_REGEX = re.compile(r"^(?P<Date>\d{6}) (?P<Time>\d{6}) (?P<Pid>\d+) (?P<Level>\w+) (?P<Component>[^:]+): (?P<Content>.*)$")
BLOCK_REGEX = re.compile(r"blk_-?\d+")

# Tên ngắn cho các template khi vẽ hình (khớp theo chuỗi con của template)
SHORT_NAMES = [
    ("addStoredBlock", "addStoredBlock"),
    ("PacketResponder", "PacketResponder terminating"),
    ("Received block", "Received block (size)"),
    ("Receiving block", "Receiving block"),
    ("Deleting block", "Deleting block"),
    ("NameSystem.delete", "NameSystem.delete"),
    ("allocateBlock", "allocateBlock"),
    ("Got exception while serving", "Got exception while serving"),
    ("Served block", "Served block"),
    ("Verification succeeded", "Verification succeeded"),
]
TOP_K_PLOT = 10


def short_name(template: str) -> str:
    """Trả về tên ngắn của template để vẽ hình; không khớp thì trả về chính template."""
    for key, name in SHORT_NAMES:
        if key in template:
            return name
    return template


def parse_lines(log_path) -> tuple[pd.DataFrame, int]:
    """
    Đọc file log, tách mỗi dòng thành các trường bằng LINE_REGEX.

    Output: (DataFrame cột LineId (bắt đầu từ 1), Date, Time, Pid, Level, Component, Content;
             số dòng không khớp regex). Dòng không khớp bị bỏ qua và in cảnh báo.
    """
    rows, n_bad = [], 0
    with open(log_path, encoding="utf-8") as f:
        for line_id, line in enumerate(f, start=1):
            m = LINE_REGEX.match(line.rstrip("\n"))
            if m:
                rows.append({"LineId": line_id, **m.groupdict()})
            else:
                n_bad += 1
    if n_bad:
        logger.warning("%d dòng không khớp regex", n_bad)
    return pd.DataFrame(rows), n_bad


def run_drain(contents: pd.Series, drain_config) -> pd.Series:
    """Đưa từng Content vào Drain3, trả về cluster_id của mỗi dòng (cùng index với contents)."""
    config = TemplateMinerConfig()
    config.load(str(drain_config))
    config.profiling_enabled = False
    miner = TemplateMiner(config=config)
    cluster_ids = [miner.add_log_message(c)["cluster_id"] for c in contents]
    # template cuối cùng của mỗi cluster (template có thể được tổng quát hoá dần trong lúc chạy)
    templates = {c.cluster_id: c.get_template() for c in miner.drain.clusters}
    return pd.Series(cluster_ids, index=contents.index).map(templates)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tuần 1: khảo sát HDFS_2k bằng Drain3")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    paths = load_config(args.config)["paths"]
    df, n_unmatched = parse_lines(paths["sample_log"])
    logger.info("Đọc %d dòng từ %s (%d dòng không khớp regex)", len(df), paths["sample_log"].name, n_unmatched)

    df["Template"] = run_drain(df["Content"], paths["sample_drain_config"])
    ref = pd.read_csv(paths["sample_ref_structured"], usecols=["LineId", "EventId"])
    df = df.merge(ref, on="LineId", how="left")

    # ---- template: Drain3 so với đáp án Loghub
    tmpl = (df.groupby("Template")
              .agg(Lines=("LineId", "size"), RefEventIds=("EventId", lambda s: " ".join(sorted(s.unique()))))
              .sort_values("Lines", ascending=False)
              .reset_index())
    split_refs = {e: int(n) for e, n in df.groupby("EventId")["Template"].nunique().items() if n > 1}
    logger.info("Drain3: %d template | Loghub: %d template", df["Template"].nunique(), df["EventId"].nunique())
    for event_id, n in split_refs.items():
        logger.info("  %s của Loghub bị tách thành %d template:", event_id, n)
        for t in tmpl.loc[tmpl["RefEventIds"].str.contains(event_id), "Template"]:
            logger.info("    %s", t if len(t) <= 80 else t[:80] + " ...")

    # ---- gom theo mã block (1 dòng có thể nhắc tới nhiều block)
    block_lines = Counter(b for content in df["Content"] for b in set(BLOCK_REGEX.findall(content)))
    n_single = sum(1 for n in block_lines.values() if n == 1)
    logger.info("%d mã block khác nhau, %d mã chỉ xuất hiện trong đúng 1 dòng", len(block_lines), n_single)

    warn = df[df["Level"] == "WARN"]
    summary = {
        "lines": len(df),
        "unmatched_lines": n_unmatched,
        "levels": df["Level"].value_counts().to_dict(),
        "components": df["Component"].value_counts().to_dict(),
        "drain_templates": int(df["Template"].nunique()),
        "loghub_templates": int(df["EventId"].nunique()),
        "loghub_events_split_by_drain": split_refs,
        "warn_templates": warn["Template"].map(short_name).value_counts().to_dict(),
        "distinct_blocks": len(block_lines),
        "blocks_in_single_line": n_single,
        "max_lines_per_block": max(block_lines.values()),
    }
    results_dir = ensure_dir(paths["results_dir"])
    save_json(summary, results_dir / "hdfs2k_survey.json")
    tmpl.to_csv(results_dir / "hdfs2k_templates.csv", index=False)
    logger.info("Level: %s | %d component", summary["levels"], len(summary["components"]))

    # ---- hình: top-k template, phần còn lại gộp thành "Khác"
    top = tmpl.head(TOP_K_PLOT)
    rest = tmpl.iloc[TOP_K_PLOT:]
    names = [short_name(t) for t in top["Template"]] + ([f"Khác ({len(rest)} mẫu)"] if len(rest) else [])
    counts = list(top["Lines"]) + ([int(rest["Lines"].sum())] if len(rest) else [])

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(names[::-1], counts[::-1], color="#555555")
    ax.bar_label(bars, padding=4, fontsize=8)
    ax.set_xlabel("Số dòng log")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig_path = ensure_dir(paths["figures_dir"]) / "hdfs2k_template_freq.png"
    fig.savefig(fig_path, dpi=200)
    logger.info("Đã lưu %s", fig_path)


if __name__ == "__main__":
    main()
