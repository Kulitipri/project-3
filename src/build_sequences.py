"""
BƯỚC 2 — Gom log có cấu trúc thành chuỗi sự kiện (session) theo block_id và gắn nhãn.

Input : data/interim/HDFS.log_structured.csv   (từ bước 1)
        data/interim/HDFS.log_templates.csv
        data/raw/anomaly_label.csv              (cột BlockId, Label = "Normal" | "Anomaly")
Output: data/processed/sessions.pkl
            DataFrame cột: BlockId, EventSequence (list[int]), StartTime (datetime), Label (0 | 1)
        data/processed/event_index.json
            dict {"E5": 1, "E3": 2, ...}  — chỉ số 0 dành riêng cho PAD

Chạy:
    python -m src.build_sequences --config configs/config.yaml
"""
import argparse
import re

import pandas as pd

from src.config import load_config
from src.utils import ensure_dir, get_logger, save_json

logger = get_logger(__name__)

BLOCK_ID_PATTERN = re.compile(r"blk_-?\d+")
PAD_INDEX = 0


def extract_block_ids(content: str) -> list[str]:
    """
    Tìm các block id trong phần Content của 1 dòng log.

    Input : content — vd "Receiving block blk_-1608999687919862906 src: /10.250.19.102:54106 ..."
    Output: list block id KHÔNG trùng lặp, giữ thứ tự xuất hiện,
            vd ["blk_-1608999687919862906"]
    Edge case:
      - Block id có thể âm (có dấu "-") → BLOCK_ID_PATTERN đã xử lý.
      - Một dòng có thể chứa nhiều block id khác nhau → trả về tất cả;
        dòng đó được tính vào session của từng block.
      - Dòng không có block id → trả về [].
    """
    return list(dict.fromkeys(BLOCK_ID_PATTERN.findall(content)))


def build_event_index(templates_df: pd.DataFrame) -> dict[str, int]:
    """
    Đánh số nguyên cho từng EventId để đưa vào model.

    Input : templates_df — DataFrame từ HDFS.log_templates.csv (đã sort theo Occurrences giảm dần)
    Output: dict EventId → số nguyên, bắt đầu từ 1, vd {"E5": 1, "E3": 2, ...}
    Lưu ý:
      - Số 0 (PAD_INDEX) dành riêng cho padding, KHÔNG gán cho EventId nào.
      - Số lượng lớp đầu ra của model = len(event_index) + 1.
    """
    return {event_id: i for i, event_id in enumerate(templates_df["EventId"], start=PAD_INDEX + 1)}


def group_by_block(structured_df: pd.DataFrame, event_index: dict[str, int]) -> pd.DataFrame:
    """
    Gom các dòng log thành session theo block id.

    Input : structured_df — DataFrame từ bước 1 (đã theo thứ tự LineId = thứ tự thời gian)
            event_index   — dict từ build_event_index()
    Output: DataFrame cột:
            BlockId       (str)
            EventSequence (list[int]) — chuỗi chỉ số sự kiện theo đúng thứ tự xuất hiện
            StartTime     (pd.Timestamp) — thời điểm dòng log đầu tiên của block
    Lưu ý hiệu năng (11 triệu dòng):
      - KHÔNG dùng df.iterrows() — chậm hàng chục lần.
      - Cách 1: vòng for trên zip(df["Content"], df["EventId"], ...) và dict BlockId → list.
      - Cách 2 (vectorized): df["Content"].str.findall(BLOCK_ID_PATTERN) → .explode() → .groupby().
    Gợi ý thời gian:
      - Date "081109" + Time "203615" → pd.to_datetime(..., format="%y%m%d%H%M%S").
    Edge case:
      - Nếu Date/Time bị pandas đọc thành số (mất số 0 đầu: 081109 → 81109) thì cần .str.zfill(6).
        Hàm main() đã đọc 2 cột này dạng str để tránh lỗi trên.
      - Dòng không có block id → bỏ qua.
    """
    df = structured_df.sort_values("LineId", kind="stable")
    df = pd.DataFrame({
        "BlockId": df["Content"].map(extract_block_ids),
        "Event": df["EventId"].map(event_index),
        "Timestamp": df["Date"].str.zfill(6) + df["Time"].str.zfill(6),
    })
    unknown = df["Event"].isna().sum()
    if unknown:
        logger.warning("Bỏ %d dòng có EventId không nằm trong event_index", unknown)
    df = df.dropna(subset=["Event"]).explode("BlockId").dropna(subset=["BlockId"])

    sessions = (
        df.groupby("BlockId", sort=False)
        .agg(EventSequence=("Event", lambda s: s.astype(int).tolist()), StartTime=("Timestamp", "first"))
        .reset_index()
    )
    sessions["StartTime"] = pd.to_datetime(sessions["StartTime"], format="%y%m%d%H%M%S")
    return sessions


def attach_labels(sessions_df: pd.DataFrame, label_path) -> pd.DataFrame:
    """
    Gắn nhãn Normal/Anomaly cho từng session.

    Input : sessions_df — DataFrame từ group_by_block()
            label_path  — đường dẫn anomaly_label.csv (cột BlockId, Label)
    Output: sessions_df thêm cột Label (int): 0 = Normal, 1 = Anomaly
    Edge case:
      - Block không có trong file nhãn → bỏ session đó, logger.warning số lượng bị bỏ.
      - Khi chạy thử với max_lines nhỏ, file nhãn có nhiều block hơn sessions_df → chỉ giữ block có trong sessions_df.
    """
    labels = pd.read_csv(label_path)
    label_map = dict(zip(labels["BlockId"], (labels["Label"] == "Anomaly").astype(int)))

    out = sessions_df.assign(Label=sessions_df["BlockId"].map(label_map))
    num_missing = out["Label"].isna().sum()
    if num_missing:
        logger.warning("Bỏ %d session không có trong file nhãn", num_missing)
    out = out.dropna(subset=["Label"]).reset_index(drop=True)
    out["Label"] = out["Label"].astype(int)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Bước 2: tạo session theo block_id")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = cfg["paths"]

    logger.info("Đọc %s", paths["structured_log"])
    structured = pd.read_csv(
        paths["structured_log"],
        usecols=["LineId", "Date", "Time", "Content", "EventId"],
        dtype={"Date": str, "Time": str},
    )
    templates = pd.read_csv(paths["templates"])

    event_index = build_event_index(templates)
    sessions = group_by_block(structured, event_index)
    sessions = attach_labels(sessions, paths["label_file"])

    ensure_dir(paths["sessions"].parent)
    sessions.to_pickle(paths["sessions"])
    save_json(event_index, paths["event_index"])

    logger.info("Số session: %d", len(sessions))
    logger.info("Phân bố nhãn:\n%s", sessions["Label"].value_counts().to_string())
    logger.info("Độ dài session (số sự kiện):\n%s", sessions["EventSequence"].str.len().describe().to_string())


if __name__ == "__main__":
    main()
