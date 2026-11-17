"""
BƯỚC 9 — Độ ổn định theo seed: kết quả của MỘT lần train DeepLog có đáng tin không?

Train lại DeepLog với từng seed trong robustness.seeds, rồi chạy lại bước 7 (chọn phương án) và bước 8
(bộ phát hiện lai) với CẢ HAI tune_mode. Tập tune / eval cố định (select.tune_seed); chỉ seed của mô hình
đổi: khởi tạo trọng số và thứ tự batch của DeepLog, random_state của Isolation Forest.
PCA và 2 luật bất biến không có yếu tố ngẫu nhiên nên không đổi theo seed.

Mỗi seed chạy trong thư mục riêng, không ghi đè kết quả chính:
  outputs/models/seeds/seed<s>/              — checkpoint DeepLog
  outputs/results/seeds/seed<s>/<tune_mode>/ — config.yaml đã sinh (để tái lập) + kết quả bước 7, 8

Input : như bước 4, 7, 8 (splits, event_index.json, block_context.pkl)
Output: outputs/results/seed_study.csv         — mỗi dòng: seed × tune_mode × phương pháp
        outputs/results/seed_study_summary.csv — trung bình / độ lệch chuẩn / min / max theo tune_mode × phương pháp
        outputs/results/seed_study_meta.json   — nguồn gốc (commit, config) của 2 file CSV trên

Chạy (~4 phút mỗi seed trên Mac M-series):
    python -m src.seed_study --config configs/config.yaml
"""
import argparse
import copy
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

from src.config import PROJECT_ROOT, load_config
from src.deeplog_scores import RANKS_FILE, RANKS_META_FILE
from src.utils import ensure_dir, get_logger, load_json, run_metadata, save_json

logger = get_logger(__name__)

TUNE_MODES = ("earliest", "random")
METRICS = ["precision", "recall", "f1", "fp_per_hour", "latency_median_sec", "roc_auc", "average_precision",
           "recall_unseen", "recall_known", "anomalies_unseen", "anomalies_known"]


def write_config(cfg: dict, seed: int, tune_mode: str, model_dir: Path, results_dir: Path) -> Path:
    """Sinh config cho 1 seed × 1 tune_mode (chỉ đổi seed, tune_mode, model_dir, results_dir) và lưu vào results_dir."""
    run_cfg = copy.deepcopy(cfg)
    run_cfg.pop("config_file", None)
    run_cfg["seed"] = seed
    run_cfg["select"]["tune_mode"] = tune_mode
    run_cfg["paths"] = {key: str(value) for key, value in cfg["paths"].items()}
    run_cfg["paths"].update(model_dir=str(model_dir), results_dir=str(results_dir))
    path = ensure_dir(results_dir) / "config.yaml"
    path.write_text(yaml.safe_dump(run_cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path


def run_step(module: str, config_path: Path) -> None:
    """Chạy `python -m <module> --config <config_path>` từ thư mục gốc project; lỗi thì dừng luôn."""
    logger.info("  %s", module)
    subprocess.run([sys.executable, "-m", module, "--config", str(config_path)], check=True, cwd=PROJECT_ROOT,
                   stdout=subprocess.DEVNULL)


def share_ranks(src_dir: Path, dst_dir: Path) -> None:
    """Symlink TƯƠNG ĐỐI deeplog_ranks.pkl + file nguồn gốc từ src_dir sang dst_dir (chép cả thư mục vẫn đúng)."""
    for name in (RANKS_FILE, RANKS_META_FILE):
        link = dst_dir / name
        link.unlink(missing_ok=True)
        link.symlink_to(Path("..") / src_dir.name / name)


def run_seed(cfg: dict, seed: int, reuse_models=False) -> dict[str, Path]:
    """
    Train + chấm rank DeepLog một lần, rồi chạy bước 7 và 8 cho từng tune_mode.

    Output: dict tune_mode → thư mục kết quả của seed đó
    """
    model_dir = cfg["paths"]["model_dir"] / "seeds" / f"seed{seed}"
    base = cfg["paths"]["results_dir"] / "seeds" / f"seed{seed}"
    dirs = {mode: base / mode for mode in TUNE_MODES}
    configs = {mode: write_config(cfg, seed, mode, model_dir, dirs[mode]) for mode in TUNE_MODES}

    first = TUNE_MODES[0]
    if reuse_models:
        from src.detect import config_differences, load_model
        model = load_model(model_dir / "deeplog_best.pt", "cpu")
        if diffs := config_differences(model.meta, {**cfg, "seed": seed}):
            raise ValueError(f"Cannot reuse checkpoint: {diffs}")
    else:
        run_step("src.train", configs[first])
    run_step("src.deeplog_scores", configs[first])
    for mode in TUNE_MODES[1:]:  # rank không phụ thuộc tune_mode: dùng chung file, không chấm lại
        share_ranks(dirs[first], dirs[mode])
    for mode in TUNE_MODES:
        run_step("src.model_selection", configs[mode])
        run_step("src.hybrid_detect", configs[mode])
        (dirs[mode] / "hybrid_alerts.pkl").unlink(missing_ok=True)  # chỉ dùng để vẽ hình, ~60 MB mỗi lần chạy
    return dirs


def collect(seed: int, tune_mode: str, results_dir: Path) -> list[dict]:
    """
    Đọc kết quả trên tập eval của 1 seed × 1 tune_mode.

    - hybrid_detection.json: DeepLog, các luật, hybrid, PCA / IForest đã chọn (có FP/giờ và độ trễ)
    - model_selection_<tune_mode>.json: ROC-AUC / AP, và PCA / IForest tốt nhất cho từng fit_on
    """
    selection = load_json(results_dir / f"model_selection_{tune_mode}.json")
    rows = []
    for method, metrics in load_json(results_dir / "hybrid_detection.json")["eval"].items():
        extra = selection[method]["eval"] if method in ("deeplog", "pca", "iforest") else {}
        rows.append({"seed": seed, "tune_mode": tune_mode, "method": method,
                     **{key: {**metrics, **extra}.get(key) for key in METRICS}})
    for method, by_fit in selection["by_fit_on"].items():
        for fit_on, choice in by_fit.items():
            rows.append({"seed": seed, "tune_mode": tune_mode, "method": f"{method}/{fit_on}",
                         **{key: (choice["eval"]["fp"] / (load_json(results_dir / "hybrid_detection.json")["stream_hours"]
                                   * (1 - selection["tune_ratio"]))) if key == "fp_per_hour"
                            else choice["eval"].get(key) for key in METRICS}})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Bước 9: độ ổn định theo seed")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--reuse-models", action="store_true", help="Recompute evaluations without retraining compatible checkpoints")
    args = parser.parse_args()

    cfg = load_config(args.config)
    rows = []
    for seed in cfg["robustness"]["seeds"]:
        logger.info("Seed %d", seed)
        for mode, results_dir in run_seed(cfg, seed, args.reuse_models).items():
            rows += collect(seed, mode, results_dir)

    df = pd.DataFrame(rows)
    results_dir = cfg["paths"]["results_dir"]
    df.to_csv(results_dir / "seed_study.csv", index=False)
    summary = df.groupby(["tune_mode", "method"], sort=False)[METRICS].agg(["mean", "std", "min", "max"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary.reset_index().to_csv(results_dir / "seed_study_summary.csv", index=False)
    save_json(run_metadata(cfg), results_dir / "seed_study_meta.json")  # cấu hình (batch_size...) của 2 file CSV trên
    for (mode, method), row in summary.iterrows():
        logger.info("%-8s %-24s F1 %.3f ± %.3f (min %.3f, max %.3f)", mode, method,
                    row["f1_mean"], row["f1_std"], row["f1_min"], row["f1_max"])


if __name__ == "__main__":
    main()
