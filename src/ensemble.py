"""
BƯỚC 10 — DeepLog ensemble: ghép các model của bước 9 (cùng cấu hình, khác seed) thành MỘT bộ phát hiện.

Vì sao: F1 của một model DeepLog đơn lẻ dao động mạnh theo seed. Nguyên nhân chính: session có <= window_size
sự kiện (≈ 37% session bất thường) chỉ được chấm ở sự kiện cuối, với cửa sổ toàn PAD mà lúc train model chưa
từng gặp (session train ngắn nhất dài 22) — dự đoán ở đó là tuỳ ý, mỗi seed một kiểu. Lấy trung bình phân phối
xác suất của nhiều seed (model.DeepLogEnsemble) làm mất phần tuỳ ý đó; luật top-g giữ nguyên ý nghĩa
("sự kiện thật nằm trong top-g của dự đoán trung bình").

Cách làm: tính rank bằng ensemble → chạy lại bước 7 (model_selection) và bước 8 (hybrid_detect) với cả 2
tune_mode, trên đúng tập tune / eval như mọi phương pháp khác → so với trung bình từng seed (seed_study.csv).
Mọi checkpoint phải được train với cấu hình hiện tại (trừ seed); lệch thì dừng.

Input : outputs/models/seeds/seed<s>/deeplog_best.pt  (python -m src.seed_study), s ∈ robustness.seeds
        data/processed/splits/*.pkl, data/interim/block_context.pkl
        outputs/results/seed_study.csv                (nếu có — để so sánh)
Output: outputs/results/ensemble/<tune_mode>/        — config.yaml, deeplog_ranks.pkl (+ meta), kết quả bước 7, 8
        outputs/results/ensemble_summary.csv         — ensemble vs trung bình ± độ lệch chuẩn của từng seed
        outputs/results/ensemble_meta.json           — nguồn gốc của file CSV trên

Chạy (~2 phút, sau seed_study):
    python -m src.ensemble --config configs/config.yaml
"""
import argparse

import pandas as pd

from src.config import load_config
from src.deeplog_scores import compute_ranks, save_ranks
from src.detect import config_differences, load_model
from src.model import DeepLogEnsemble
from src.seed_study import TUNE_MODES, collect, run_step, share_ranks, write_config
from src.utils import get_device, get_logger, run_metadata, save_json

logger = get_logger(__name__)

COMPARE_METHODS = ["deeplog", "hybrid", "rules", "rule_unseen", "rule_completion", "pca", "iforest",
                   "pca/train_normal", "pca/unlabeled_tune", "iforest/train_normal", "iforest/unlabeled_tune"]
COMPARE_METRICS = ["f1", "precision", "recall", "fp_per_hour", "roc_auc", "average_precision",
                   "recall_unseen", "recall_known", "anomalies_unseen", "anomalies_known"]


def load_seed_models(cfg: dict, device) -> tuple[list, list]:
    """Load checkpoint của mọi seed trong robustness.seeds; dừng nếu thiếu hoặc train với cấu hình khác cfg."""
    models = []
    for seed in cfg["robustness"]["seeds"]:
        path = cfg["paths"]["model_dir"] / "seeds" / f"seed{seed}" / "deeplog_best.pt"
        if not path.exists():
            raise FileNotFoundError(f"Thiếu {path} — chạy python -m src.seed_study trước")
        model = load_model(path, device)
        if diffs := config_differences(model.meta, cfg, check_seed=False):
            raise ValueError(f"{path} được train với cấu hình khác config hiện tại: {diffs} — chạy lại seed_study")
        models.append(model)
    return models, [m.meta for m in models]


def compare(ensemble_rows: list[dict], seed_study_path) -> pd.DataFrame:
    """
    Bảng so sánh: mỗi dòng tune_mode × phương pháp × chỉ số, cột ensemble | seed_mean | seed_std | seed_min | seed_max.

    "deeplog"/"hybrid" của ensemble so với chính chúng khi dùng 1 model; các phương pháp không dùng DeepLog
    (rules, PCA, IForest) đưa vào làm mốc tham chiếu.
    """
    ens = pd.DataFrame(ensemble_rows).set_index(["tune_mode", "method"])
    seeds = pd.read_csv(seed_study_path) if seed_study_path.exists() else None
    rows = []
    for (mode, method), row in ens.iterrows():
        if method not in COMPARE_METHODS:
            continue
        for metric in COMPARE_METRICS:
            out = {"tune_mode": mode, "method": method, "metric": metric, "ensemble": row[metric]}
            if seeds is not None:
                values = seeds.loc[(seeds["tune_mode"] == mode) & (seeds["method"] == method), metric].dropna()
                out.update(seed_mean=values.mean(), seed_std=values.std(), seed_min=values.min(), seed_max=values.max())
            rows.append(out)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Bước 10: DeepLog ensemble qua các seed")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = cfg["paths"]
    device = get_device(cfg["train"]["device"])
    models, metas = load_seed_models(cfg, device)
    ensemble = DeepLogEnsemble(models).to(device).eval()
    logger.info("Ensemble %d model (seed %s) | device=%s", len(models), cfg["robustness"]["seeds"], device)

    base = paths["results_dir"] / "ensemble"
    dirs = {mode: base / mode for mode in TUNE_MODES}
    configs = {mode: write_config(cfg, cfg["seed"], mode, paths["model_dir"], dirs[mode]) for mode in TUNE_MODES}

    test_df = pd.read_pickle(paths["splits_dir"] / "test.pkl")
    ranks = compute_ranks(ensemble, test_df, cfg["model"]["window_size"], device)
    save_ranks(ranks, dirs[TUNE_MODES[0]], cfg, metas)
    for mode in TUNE_MODES[1:]:
        share_ranks(dirs[TUNE_MODES[0]], dirs[mode])

    rows = []
    for mode in TUNE_MODES:
        run_step("src.model_selection", configs[mode])
        run_step("src.hybrid_detect", configs[mode])
        (dirs[mode] / "hybrid_alerts.pkl").unlink(missing_ok=True)
        rows += collect("ensemble", mode, dirs[mode])

    table = compare(rows, paths["results_dir"] / "seed_study.csv")
    table.to_csv(paths["results_dir"] / "ensemble_summary.csv", index=False)
    save_json(run_metadata(cfg), paths["results_dir"] / "ensemble_meta.json")
    for _, r in table[table["metric"].isin(["f1", "fp_per_hour"])].iterrows():
        logger.info("%-8s %-17s %-11s ensemble %8.3f | 1 model: %8.3f ± %.3f (min %.3f, max %.3f)", r["tune_mode"],
                    r["method"], r["metric"], r["ensemble"], r.get("seed_mean", float("nan")),
                    r.get("seed_std", float("nan")), r.get("seed_min", float("nan")), r.get("seed_max", float("nan")))


if __name__ == "__main__":
    main()
