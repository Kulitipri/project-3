"""
Kiểm tra kết quả parse (Drain3) và cách gom session so với dữ liệu chuẩn của Loghub.

Phần A — Template: so từng cluster của Drain3 với 29 template chuẩn (E1–E29).
    Mỗi dòng log được gán "nhãn chuẩn" bằng cách khớp Content với regex dựng từ HDFS.log_templates.csv
    của Loghub. Từ đó tính:
      - bảng ánh xạ cluster Drain3 → template chuẩn (độ thuần, cluster nào bị gộp/tách sai)
      - Grouping Accuracy (thước đo chuẩn của benchmark log parsing, Zhu et al. ICSE-SEIP 2019):
        1 dòng được tính là đúng nếu tập dòng cùng cluster với nó TRÙNG KHÍT tập dòng cùng template chuẩn.
Phần B — Chuỗi theo block: ánh xạ cluster → template chuẩn (theo đa số), dựng chuỗi sự kiện từng block
    rồi so với Event_traces.csv. Đây là tiêu chí ở Mục 8 báo cáo tuần 1:
    nếu tỉ lệ khớp thấp ở cuối tuần 2 thì train trên chuỗi đã tiền xử lý của Loghub.
    Khi parse thử với max_lines, chuỗi bị cắt nên chỉ kiểm tra "là tiền tố của chuỗi chuẩn".
Phần C — Số session train của DeepLog: bài báo nói lấy 4.855 session Normal từ 100k dòng đầu.
    Đếm lại bằng data/interim/block_stats.csv (không cần Drain3).
Phần D — (nếu đã có data/processed/sessions.pkl) so độ dài từng session với Event_traces.csv.

Input : data/interim/HDFS.log_structured.csv  (bước 1; cần cột LineId, Content, EventId, EventTemplate)
        data/interim/block_stats.csv           (python -m src.block_stats)
        data/raw/HDFS_v1/preprocessed/HDFS.log_templates.csv, Event_traces.csv, anomaly_label.csv
Output: outputs/results/parsing_validation.json
        outputs/results/drain_vs_reference.csv

Chạy:
    python -m src.validate_parsing --config configs/config.yaml
    python -m src.validate_parsing --structured đường/dẫn/khác.csv   # kiểm tra 1 file parse khác
"""
import argparse
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.block_stats import load_block_stats
from src.build_sequences import BLOCK_ID_PATTERN
from src.config import load_config
from src.utils import ensure_dir, get_logger, save_json

logger = get_logger(__name__)

NO_MATCH = "NONE"  # nhãn cho dòng không khớp template chuẩn nào


# ---------------------------------------------------------------- dữ liệu chuẩn Loghub

def load_reference_templates(path: Path) -> pd.DataFrame:
    """Đọc 29 template chuẩn: cột EventId (E1..E29), EventTemplate (dạng "[*]Served block[*]to[*]")."""
    return pd.read_csv(path)


def build_reference_matcher(ref_templates: pd.DataFrame):
    """
    Dựng hàm Content → EventId chuẩn (hoặc NO_MATCH).

    Mỗi "[*]" thành ".*?", phần chữ cố định được escape. Các template gộp thành 1 regex dạng
    ^(?:(?P<E26>...)|(?P<E5>...)|...)$, xếp template có nhiều chữ cố định hơn lên trước
    để khi 1 dòng khớp nhiều template thì template cụ thể hơn thắng.
    """
    def literal_len(template: str) -> int:
        return len(template.replace("[*]", ""))

    ordered = sorted(ref_templates.itertuples(index=False), key=lambda r: -literal_len(r.EventTemplate))
    alternatives = [
        f"(?P<{r.EventId}>" + ".*?".join(re.escape(p) for p in r.EventTemplate.split("[*]")) + ")"
        for r in ordered
    ]
    pattern = re.compile("^(?:" + "|".join(alternatives) + ")$")

    def match(content: str) -> str:
        m = pattern.match(content)
        return m.lastgroup if m else NO_MATCH

    return match


def load_reference_traces(path: Path) -> dict[str, list[str]]:
    """Đọc Event_traces.csv → dict BlockId → ["E5", "E22", ...]."""
    df = pd.read_csv(path, usecols=["BlockId", "Features"])
    return dict(zip(df["BlockId"], df["Features"].str.strip("[]").str.split(",")))


# ---------------------------------------------------------------- phần A + B

def scan_structured(structured_path: Path, ref_match, chunksize: int = 1_000_000):
    """
    Đọc file structured theo từng khối (không nạp 1,5GB vào RAM một lần).

    Trả về:
      pair_counts    — DataFrame cột DrainEventId, RefEventId, Count
      drain_seqs     — dict BlockId → list EventId của Drain3 (theo thứ tự dòng)
      drain_template — dict EventId Drain3 → template cuối cùng
      examples       — dict (DrainEventId, RefEventId) → 1 Content ví dụ
      num_lines, max_line_id
    """
    pair_counts: dict[tuple[str, str], int] = defaultdict(int)
    drain_seqs: dict[str, list[str]] = defaultdict(list)
    drain_template: dict[str, str] = {}
    examples: dict[tuple[str, str], str] = {}
    num_lines = 0
    max_line_id = 0

    reader = pd.read_csv(
        structured_path,
        usecols=["LineId", "Content", "EventId", "EventTemplate"],
        dtype={"Content": str, "EventId": str},
        chunksize=chunksize,
    )
    for chunk in tqdm(reader, desc="Đọc structured (khối 1 triệu dòng)"):
        chunk = chunk.sort_values("LineId")
        num_lines += len(chunk)
        max_line_id = max(max_line_id, int(chunk["LineId"].max()))
        drain_template.update(zip(chunk["EventId"], chunk["EventTemplate"]))
        for content, drain_id in zip(chunk["Content"].fillna(""), chunk["EventId"]):
            ref_id = ref_match(content)
            key = (drain_id, ref_id)
            pair_counts[key] += 1
            if key not in examples:
                examples[key] = content
            for block in dict.fromkeys(BLOCK_ID_PATTERN.findall(content)):
                drain_seqs[block].append(drain_id)

    pairs = pd.DataFrame(
        [(d, r, c) for (d, r), c in pair_counts.items()],
        columns=["DrainEventId", "RefEventId", "Count"],
    )
    return pairs, drain_seqs, drain_template, examples, num_lines, max_line_id


def grouping_accuracy(pairs: pd.DataFrame) -> float:
    """
    Tỉ lệ dòng log mà cluster Drain3 chứa nó trùng khít với nhóm template chuẩn của nó.

    Cặp (cluster c, template r) có n dòng là "đúng" khi n = |c| = |r|, tức c và r là cùng 1 tập dòng.
    """
    drain_size = pairs.groupby("DrainEventId")["Count"].transform("sum")
    ref_size = pairs.groupby("RefEventId")["Count"].transform("sum")
    correct = pairs.loc[(pairs["Count"] == drain_size) & (pairs["Count"] == ref_size), "Count"].sum()
    return float(correct / pairs["Count"].sum())


def build_mapping_table(pairs, drain_template, ref_templates, examples) -> pd.DataFrame:
    """
    Mỗi cluster Drain3 1 hàng: template chuẩn chiếm đa số, độ thuần, và trạng thái:
      ok        — cluster thuần (100%) và là cluster duy nhất của template chuẩn đó
      split     — thuần nhưng template chuẩn bị Drain3 tách ra nhiều cluster
      merged    — cluster lẫn dòng của template chuẩn khác (dù chỉ 1 dòng)
      no_ref    — đa số dòng không khớp template chuẩn nào
    """
    ref_text = dict(zip(ref_templates["EventId"], ref_templates["EventTemplate"]))
    totals = pairs.groupby("DrainEventId")["Count"].sum()
    top = pairs.sort_values("Count", ascending=False).drop_duplicates("DrainEventId").set_index("DrainEventId")

    table = pd.DataFrame({
        "DrainEventId": totals.index,
        "DrainTemplate": [drain_template.get(d, "") for d in totals.index],
        "Occurrences": totals.values,
        "RefEventId": top.loc[totals.index, "RefEventId"].values,
        "Purity": (top.loc[totals.index, "Count"] / totals).values,
    })
    table["RefTemplate"] = table["RefEventId"].map(ref_text).fillna("")
    # các template chuẩn khác bị gộp vào cluster này
    minority = (
        pairs.merge(table[["DrainEventId", "RefEventId"]], on="DrainEventId", suffixes=("", "_top"))
        .query("RefEventId != RefEventId_top")
        .sort_values("Count", ascending=False)
    )
    others: dict[str, list[str]] = defaultdict(list)
    for d, r, c in zip(minority["DrainEventId"], minority["RefEventId"], minority["Count"]):
        others[d].append(f"{r}×{c}")
    table["AlsoContains"] = ["; ".join(others.get(d, [])) for d in table["DrainEventId"]]
    table["Example"] = [examples[(d, r)] for d, r in zip(table["DrainEventId"], table["RefEventId"])]

    pure = table["Purity"] == 1.0
    clusters_per_ref = table.loc[pure].groupby("RefEventId")["DrainEventId"].transform("count")
    table["Status"] = "merged"
    table.loc[pure, "Status"] = "ok"
    table.loc[clusters_per_ref.index[clusters_per_ref > 1], "Status"] = "split"
    table.loc[table["RefEventId"] == NO_MATCH, "Status"] = "no_ref"
    return table.sort_values(["RefEventId", "Occurrences"], ascending=[True, False]).reset_index(drop=True)


def compare_sequences(drain_seqs, mapping: dict[str, str], ref_traces, full_file: bool) -> dict:
    """
    Đổi chuỗi EventId Drain3 → EventId chuẩn (theo mapping đa số) rồi so với Event_traces.csv.

    full_file=True : yêu cầu trùng khít.
    full_file=False: chỉ parse 1 phần file → chuỗi bị cắt, kiểm tra là tiền tố của chuỗi chuẩn.
    """
    exact = prefix = mismatch = missing = 0
    mismatch_examples = []
    for block, seq in drain_seqs.items():
        ref = ref_traces.get(block)
        if ref is None:
            missing += 1
            continue
        mapped = [mapping[e] for e in seq]
        if mapped == ref:
            exact += 1
        elif not full_file and mapped == ref[: len(mapped)]:
            prefix += 1
        else:
            mismatch += 1
            if len(mismatch_examples) < 5:
                pos = next((i for i, (a, b) in enumerate(zip(mapped, ref)) if a != b), min(len(mapped), len(ref)))
                mismatch_examples.append({
                    "BlockId": block,
                    "first_diff_position": pos,
                    "drain": mapped[max(0, pos - 2): pos + 3],
                    "reference": ref[max(0, pos - 2): pos + 3],
                })
    total = exact + prefix + mismatch
    return {
        "num_blocks": total,
        "blocks_not_in_reference": missing,
        "exact_match": exact,
        "prefix_match": prefix,
        "mismatch": mismatch,
        "match_rate": round((exact + prefix) / total, 6) if total else None,
        "mismatch_examples": mismatch_examples,
    }


# ---------------------------------------------------------------- phần C + D

def check_deeplog_train_count(block_stats: pd.DataFrame, labels: pd.DataFrame, n_lines: int, expected: int) -> dict:
    """
    Đếm session Normal trong n_lines dòng đầu theo vài cách hiểu, so với con số bài báo.

      touched  : block có ít nhất 1 dòng trong n_lines dòng đầu
      complete : block có TẤT CẢ các dòng nằm trong n_lines dòng đầu
      budget   : lấy block theo thứ tự xuất hiện, cộng dồn số dòng đến khi đủ n_lines
    """
    df = block_stats.merge(labels, on="BlockId").sort_values("FirstLine").reset_index(drop=True)
    normal = df["Label"] == "Normal"
    budget = df["NumLines"].cumsum() <= n_lines

    def count(mask):
        return {"normal": int((mask & normal).sum()), "anomaly": int((mask & ~normal).sum())}

    first_line_of_expected = int(df.loc[normal, "FirstLine"].iloc[expected - 1])
    return {
        "n_lines": n_lines,
        "paper_normal_sessions": expected,
        "touched": count(df["FirstLine"] <= n_lines),
        "complete": count(df["LastLine"] <= n_lines),
        "budget": count(budget),
        "line_where_normal_count_reaches_paper_value": first_line_of_expected,
    }


def check_sessions_file(sessions_path: Path, ref_traces) -> dict | None:
    """Nếu bước 2 đã chạy: so số session và độ dài từng session với Event_traces.csv."""
    if not sessions_path.exists():
        return None
    sessions = pd.read_pickle(sessions_path)
    ref_len = sessions["BlockId"].map(lambda b: len(ref_traces[b]) if b in ref_traces else -1)
    own_len = sessions["EventSequence"].str.len()
    return {
        "num_sessions": int(len(sessions)),
        "num_reference_blocks": len(ref_traces),
        "not_in_reference": int((ref_len == -1).sum()),
        "same_length": int((ref_len == own_len).sum()),
        "shorter_than_reference": int(((own_len < ref_len) & (ref_len != -1)).sum()),
        "longer_than_reference": int((own_len > ref_len).sum()),
    }


# ---------------------------------------------------------------- main

def main() -> None:
    parser = argparse.ArgumentParser(description="Kiểm tra kết quả parse so với dữ liệu chuẩn Loghub")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--structured", default=None, help="Ghi đè paths.structured_log")
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = cfg["paths"]
    structured_path = Path(args.structured) if args.structured else paths["structured_log"]
    results_dir = ensure_dir(paths["results_dir"])

    ref_templates = load_reference_templates(paths["ref_templates"])
    ref_traces = load_reference_traces(paths["ref_traces"])
    labels = pd.read_csv(paths["label_file"])
    report: dict = {}

    # Phần C — không cần Drain3
    if paths["block_stats"].exists():
        report["deeplog_train_count"] = check_deeplog_train_count(
            load_block_stats(paths["block_stats"]), labels,
            cfg["validate"]["deeplog_train_lines"], cfg["validate"]["deeplog_train_normal_sessions"],
        )
        logger.info("Số session train DeepLog: %s", report["deeplog_train_count"])
    else:
        logger.warning("Chưa có %s — chạy python -m src.block_stats trước.", paths["block_stats"])

    # Phần A + B
    if structured_path.exists():
        ref_match = build_reference_matcher(ref_templates)
        pairs, drain_seqs, drain_template, examples, num_lines, max_line_id = scan_structured(structured_path, ref_match)

        table = build_mapping_table(pairs, drain_template, ref_templates, examples)
        table.to_csv(results_dir / "drain_vs_reference.csv", index=False)

        ref_ids_found = set(pairs.loc[pairs["RefEventId"] != NO_MATCH, "RefEventId"])
        ok_refs = set(table.loc[table["Status"] == "ok", "RefEventId"])
        full_file = max_line_id >= sum(1 for _ in open(paths["raw_log"], "rb"))
        mapping = dict(zip(table["DrainEventId"], table["RefEventId"]))

        report["structured_file"] = str(structured_path)
        report["num_lines"] = num_lines
        report["full_file"] = full_file
        report["templates"] = {
            "num_drain_templates": int(len(table)),
            "num_reference_templates_seen": len(ref_ids_found),
            "reference_templates_not_seen": sorted(set(ref_templates["EventId"]) - ref_ids_found, key=lambda e: int(e[1:])),
            "reference_templates_parsed_cleanly": len(ok_refs),
            "lines_without_reference_template": int(pairs.loc[pairs["RefEventId"] == NO_MATCH, "Count"].sum()),
            "status_counts": table["Status"].value_counts().to_dict(),
            "grouping_accuracy": round(grouping_accuracy(pairs), 6),
        }
        report["sequences"] = compare_sequences(drain_seqs, mapping, ref_traces, full_file)

        logger.info("Template: %s", report["templates"])
        logger.info("Chuỗi theo block: %s", {k: v for k, v in report["sequences"].items() if k != "mismatch_examples"})
        bad = table[table["Status"] != "ok"]
        if len(bad):
            logger.info("Cluster chưa sạch:\n%s", bad[["DrainEventId", "Occurrences", "RefEventId", "Purity", "Status", "DrainTemplate"]].to_string(index=False))
    else:
        logger.warning("Chưa có %s — viết xong và chạy python -m src.parse_logs trước.", structured_path)

    # Phần D
    sessions_check = check_sessions_file(paths["sessions"], ref_traces)
    if sessions_check:
        report["sessions_file"] = sessions_check
        logger.info("sessions.pkl so với Event_traces: %s", sessions_check)

    save_json(report, results_dir / "parsing_validation.json")
    logger.info("Đã lưu %s", results_dir / "parsing_validation.json")


if __name__ == "__main__":
    main()
