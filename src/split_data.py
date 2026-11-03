"""
BƯỚC 3 — Chia session thành tập train / val / test.

Nguyên tắc DeepLog:
  - Train và val CHỈ chứa session Normal (model học "hành vi bình thường").
  - Test chứa session Normal còn lại + session Anomaly.

Input : data/processed/sessions.pkl
Output: data/processed/splits/train.pkl, val.pkl, test.pkl, summary.json

Chạy:
    python -m src.split_data --config configs/config.yaml
"""
import argparse

import pandas as pd

from src.config import load_config
from src.utils import ensure_dir, get_logger, save_json

logger = get_logger(__name__)


def split_sessions(
    sessions_df: pd.DataFrame,
    strategy: str,
    train_normal_ratio: float,
    val_ratio: float,
    seed: int,
) -> dict[str, pd.DataFrame]:
    """
    Chia session theo 1 trong 2 chiến lược.

    Input : sessions_df        — DataFrame cột BlockId, EventSequence, StartTime, Label
            strategy           — "random" hoặc "chronological"
            train_normal_ratio — tỉ lệ session Normal dùng cho train + val (vd 0.01)
            val_ratio          — tỉ lệ trích từ phần train để làm val (vd 0.1)
            seed               — seed cho chia ngẫu nhiên
    Output: dict {"train": df, "val": df, "test": df}

    Hai chiến lược:
      - "random"       : lấy ngẫu nhiên train_normal_ratio session Normal làm train + val.
                         Đây là cách nhiều bài làm theo, dễ cho F1 cao "ảo".
      - "chronological": sắp xếp session Normal theo StartTime, lấy phần SỚM NHẤT làm train + val;
                         test là các session bắt đầu SAU mốc thời gian cắt.
                         Mô phỏng đúng thực tế: train trên quá khứ, phát hiện trên tương lai.

    Với "chronological":
      - Val là phần MUỘN NHẤT của train + val (val nằm sau train về thời gian).
      - Session Anomaly có StartTime TRƯỚC mốc cắt bị bỏ, để test chỉ chứa "tương lai".
        Train ~1% nên mốc cắt rất sớm, số session bị bỏ nhỏ (được logger.info ra).

    Edge case:
      - Train không được chứa session Anomaly nào → assert ở cuối hàm.
      - Một BlockId không được xuất hiện ở 2 tập cùng lúc → assert giao nhau rỗng.
    """
    normal = sessions_df[sessions_df["Label"] == 0]
    anomaly = sessions_df[sessions_df["Label"] == 1]
    n_train_val = max(2, round(len(normal) * train_normal_ratio))
    n_val = max(1, round(n_train_val * val_ratio))

    if strategy == "random":
        normal = normal.sample(frac=1.0, random_state=seed)
        train_val, test_normal = normal.iloc[:n_train_val], normal.iloc[n_train_val:]
        val = train_val.sample(n=n_val, random_state=seed)
        train = train_val.drop(val.index)
    elif strategy == "chronological":
        normal = normal.sort_values("StartTime", kind="stable")
        train_val, test_normal = normal.iloc[:n_train_val], normal.iloc[n_train_val:]
        train, val = train_val.iloc[:-n_val], train_val.iloc[-n_val:]
        cutoff = train_val["StartTime"].max()
        before_cutoff = anomaly["StartTime"] < cutoff
        logger.info("Mốc cắt: %s | bỏ %d session Anomaly bắt đầu trước mốc cắt", cutoff, before_cutoff.sum())
        anomaly = anomaly[~before_cutoff]
    else:
        raise ValueError(f"strategy không hợp lệ: {strategy!r} (chỉ nhận 'random' hoặc 'chronological')")

    test = pd.concat([test_normal, anomaly]).sort_values("StartTime", kind="stable")
    splits = {name: df.reset_index(drop=True) for name, df in (("train", train), ("val", val), ("test", test))}

    assert (splits["train"]["Label"] == 0).all() and (splits["val"]["Label"] == 0).all()
    ids = [set(df["BlockId"]) for df in splits.values()]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2]), "BlockId bị trùng giữa các tập"
    return splits


def summarize_split(splits: dict[str, pd.DataFrame]) -> dict:
    """Đếm số session Normal / Anomaly trong từng tập."""
    summary = {}
    for name, df in splits.items():
        summary[name] = {
            "total": int(len(df)),
            "normal": int((df["Label"] == 0).sum()),
            "anomaly": int((df["Label"] == 1).sum()),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Bước 3: chia train/val/test")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--strategy", default=None, help="Ghi đè split.strategy trong config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = cfg["paths"]
    split_cfg = cfg["split"]
    strategy = args.strategy or split_cfg["strategy"]

    sessions = pd.read_pickle(paths["sessions"])
    splits = split_sessions(
        sessions,
        strategy=strategy,
        train_normal_ratio=split_cfg["train_normal_ratio"],
        val_ratio=split_cfg["val_ratio"],
        seed=cfg["seed"],
    )

    out_dir = ensure_dir(paths["splits_dir"])
    for name, df in splits.items():
        df.to_pickle(out_dir / f"{name}.pkl")

    summary = {"strategy": strategy, **summarize_split(splits)}
    save_json(summary, out_dir / "summary.json")
    logger.info("Kết quả chia dữ liệu: %s", summary)


if __name__ == "__main__":
    main()
