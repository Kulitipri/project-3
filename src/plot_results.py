"""
Vẽ hình cho báo cáo từ kết quả của hybrid_detect.py.

Input : outputs/results/hybrid_detection.json, hybrid_drift.csv, hybrid_alerts.pkl
Output: outputs/figures/fig_comparison.png — F1 và cảnh báo giả/giờ của từng phương pháp (tập eval)
        outputs/figures/fig_drift.png      — FPR theo thời gian (drift)
        outputs/figures/fig_latency.png    — phân bố độ trễ phát hiện (chỉ TP trên tập eval)

Chạy:
    python -m src.plot_results --config configs/config.yaml
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT, load_config
from src.utils import ensure_dir, get_logger, load_json

logger = get_logger(__name__)

# Bảng màu tham chiếu (đã kiểm tra mù màu bằng validate_palette.js) — mỗi phương pháp giữ 1 màu cố định
COLORS = {"deeplog": "#2a78d6", "rules": "#eb6834", "hybrid": "#1baf7a", "end_of_session": "#eda100"}
INK, INK_2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
LABELS = {
    "pca": "PCA", "pca_train_normal": "PCA", "iforest": "Isolation Forest", "deeplog": "DeepLog",
    "rule_unseen": "Luật: sự kiện lạ", "rule_completion": "Luật: nhân bản đúng hạn",
    "rules": "2 luật bất biến", "hybrid": "Hybrid (luật + DeepLog)",
    "hybrid_node_risk": "Hybrid + rủi ro datanode",
}


def style_axes(ax) -> None:
    """Trục và lưới mờ, không viền trên/phải."""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def method_label(method: str, metrics: dict) -> str:
    """Tên hiển thị; PCA / IForest ghi thêm dữ liệu fit (session Normal hay log không nhãn của tập tune)."""
    fit_on = metrics.get("params", {}).get("fit_on")
    suffix = {"train_normal": " (fit Normal)", "unlabeled_tune": " (fit log không nhãn)"}.get(fit_on, "")
    return LABELS[method] + suffix


def plot_comparison(table: dict, heading: str, out_path) -> None:
    """2 biểu đồ cạnh nhau (không dùng 2 trục y): F1 và cảnh báo giả/giờ."""
    order = [
        m for m in (
            "iforest", "pca_train_normal", "pca", "deeplog",
            "rule_unseen", "rule_completion", "rules", "hybrid",
        )
        if m in table
    ]
    names = [method_label(m, table[m]) for m in order]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURFACE, sharey=True)
    for ax, key, title, fmt in ((axes[0], "f1", "F1 trên tập eval (cao hơn = tốt hơn)", "{:.3f}"),
                                (axes[1], "fp_per_hour", "Cảnh báo giả mỗi giờ (thấp hơn = tốt hơn)", "{:.1f}")):
        values = [table[m][key] for m in order]
        colors = [COLORS["hybrid"] if m == "hybrid" else COLORS["deeplog"] for m in order]
        bars = ax.barh(names, values, color=colors, height=0.6, edgecolor=SURFACE, linewidth=2)
        for bar, v in zip(bars, values):
            ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, " " + fmt.format(v),
                    va="center", ha="left", fontsize=9, color=INK)
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, max(values) * 1.18)
        style_axes(ax)
        ax.grid(axis="y", visible=False)
    fig.suptitle(heading, fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_drift(drift: pd.DataFrame, out_path) -> None:
    """FPR theo khoảng thời gian bắt đầu block; bỏ các khoảng có quá ít session (nhiễu) và ngắt đường ở đó."""
    # khoảng có < 1000 session → NaN để đường bị ngắt ở đó, không nối qua chỗ không có dữ liệu
    step = int(drift["bin_start_hour"].diff().min())
    drift = drift.set_index("bin_start_hour").reindex(range(0, int(drift["bin_start_hour"].max()) + 1, step))
    drift.loc[~(drift["sessions"] >= 1000), :] = np.nan
    drift = drift.reset_index()
    fig, ax = plt.subplots(figsize=(10, 4), facecolor=SURFACE)
    for name in ("deeplog", "rules", "hybrid"):
        ax.plot(drift["bin_start_hour"], drift[f"{name}_fpr"] * 100, color=COLORS[name], linewidth=2,
                marker="o", markersize=5, label=LABELS[name])
    peak = drift.loc[drift["deeplog_fpr"].idxmax()]
    ax.annotate(f"DeepLog báo nhầm {peak['deeplog_fpr'] * 100:.1f}% block bình thường",
                (peak["bin_start_hour"], peak["deeplog_fpr"] * 100), xytext=(12, -4), textcoords="offset points",
                fontsize=9, color=INK)
    ax.set_xlabel("Giờ kể từ đầu log (train chỉ dùng 9 phút đầu)", fontsize=9, color=INK_2)
    ax.set_ylabel("Tỉ lệ báo động giả (FPR, %)", fontsize=9, color=INK_2)
    ax.set_title("Drift theo thời gian: FPR của từng phương pháp trong mỗi khoảng 2 giờ", fontsize=10, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_latency(alerts: pd.DataFrame, out_path) -> None:
    """CDF độ trễ phát hiện (giây, thang log) cho các TP trên tập eval; PCA/IForest = chờ hết session."""
    ev = alerts[~alerts["is_tune"] & (alerts["Label"] == 1)]
    fig, ax = plt.subplots(figsize=(10, 4), facecolor=SURFACE)
    series = {name: ev[f"alert_{name}"] - ev["t0"] for name in ("deeplog", "rules", "hybrid")}
    series["end_of_session"] = ev["t_end"] - ev["t0"]
    labels = {**LABELS, "end_of_session": "PCA / IForest (chờ hết session)"}
    for name, delay in series.items():
        d = np.sort(delay[np.isfinite(delay)].to_numpy())
        d = np.maximum(d, 1)  # thang log: độ trễ 0 giây vẽ ở 1 giây
        frac = np.arange(1, len(d) + 1) / len(ev)  # mẫu số = mọi bất thường → đường dừng ở mức recall
        ax.step(d, frac * 100, where="post", color=COLORS[name], linewidth=2, label=labels[name])
    ax.set_xscale("log")
    ax.set_xticks([1, 10, 60, 600, 3600, 36000])
    ax.set_xticklabels(["≤1 s", "10 s", "1 phút", "10 phút", "1 giờ", "10 giờ"])
    ax.set_xlabel("Thời gian từ sự kiện đầu tiên của block tới lúc cảnh báo", fontsize=9, color=INK_2)
    ax.set_ylabel("% bất thường đã được phát hiện", fontsize=9, color=INK_2)
    ax.set_title("Độ trễ phát hiện (tập eval) — đường càng lên sớm càng tốt; độ cao cuối = recall",
                 fontsize=10, color=INK, loc="left")
    ax.set_ylim(0, 102)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Vẽ hình kết quả cho báo cáo")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    results_dir = cfg["paths"]["results_dir"]
    fig_dir = ensure_dir(PROJECT_ROOT / "outputs" / "figures" / results_dir.relative_to(PROJECT_ROOT / "outputs" / "results"))

    hybrid = load_json(results_dir / "hybrid_detection.json")
    tune_text = {"random": "10% ngẫu nhiên", "earliest": "10% sớm nhất"}[hybrid["tune_mode"]]
    split_text = {"chronological": "chia theo thời gian", "random": "chia ngẫu nhiên"}[cfg["split"]["strategy"]]
    plot_comparison(hybrid["eval"], f"So sánh trên tập eval — {split_text}, seed {cfg['seed']} (1 lần chạy), "
                    f"tham số chọn trên {tune_text} của test", fig_dir / "fig_comparison.png")
    plot_drift(pd.read_csv(results_dir / "hybrid_drift.csv"), fig_dir / "fig_drift.png")
    plot_latency(pd.read_pickle(results_dir / "hybrid_alerts.pkl"), fig_dir / "fig_latency.png")
    logger.info("Đã lưu hình vào %s", fig_dir)


if __name__ == "__main__":
    main()
