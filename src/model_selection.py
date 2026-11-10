"""
BƯỚC 7 — Chọn phương án cho PCA, Isolation Forest và DeepLog mà KHÔNG nhìn nhãn của phần test dùng để báo cáo.

Tập test (đã sắp theo StartTime) được tách làm 2:
  - tune : tune_ratio session — có nhãn, chỉ dùng để chấm điểm và chọn phương án
           tune_mode = "earliest": lấy các session SỚM NHẤT (chặt chẽ về thời gian: chọn trên quá khứ)
           tune_mode = "random"  : lấy ngẫu nhiên trong toàn bộ test (đại diện hơn khi phân bố đổi theo thời gian)
  - eval : phần còn lại — mỗi phương án đã chọn chỉ chạy trên đây ĐÚNG MỘT LẦN, số liệu này đưa vào báo cáo
Nếu chọn phương án bằng F1 trên chính tập test thì con số báo cáo bị thổi phồng (chỉnh theo nhãn test).

Mọi phương pháp cho ra một ĐIỂM bất thường liên tục cho từng session:
  - PCA    : sai số tái tạo
  - IForest: -score_samples
  - DeepLog: rank lớn nhất của sự kiện thật trong session (deeplog_ranks.pkl). Luật top-g của bài báo
             chính là đặt ngưỡng trên điểm này: session bất thường ⇔ max rank > g.
Ngưỡng của MỌI phương pháp được chọn cùng một cách: ngưỡng cho F1 cao nhất trên tập tune (duyệt mọi
ngưỡng trên đường precision–recall). Như vậy không phương pháp nào được thử nhiều ngưỡng hơn phương pháp khác.

Không gian thử của PCA / IForest (mục `select` trong config): biến đổi đặc trưng × n_components / n_estimators × fit_on
  fit_on = "train_normal"  : fit trên session Normal của train + val (cùng dữ liệu với DeepLog)
  fit_on = "unlabeled_tune": fit trên tập tune nhưng KHÔNG dùng nhãn (unsupervised, kiểu Loglizer).
                             Với tune_mode = "random", tập tune rải khắp thời gian test nên phương án này
                             dùng log (không nhãn) của tương lai — chỉ là cận trên, không triển khai được.

Input : data/processed/splits*/train.pkl, val.pkl, test.pkl, event_index.json
        outputs/results*/deeplog_ranks.pkl  (python -m src.deeplog_scores)
Output: outputs/results*/model_selection_<tune_mode>_grid.csv — F1 trên tập tune của mọi phương án (không có số eval)
        outputs/results*/model_selection_<tune_mode>.json:
          pca / iforest / deeplog — phương án có F1 tune cao nhất + kết quả trên eval (P/R/F1, ROC-AUC, AP,
                                    recall theo 2 nhóm bất thường: recall_structural / recall_sequence)
          by_fit_on               — phương án tốt nhất (theo tune) của PCA / IForest cho TỪNG fit_on,
                                    để so sánh với DeepLog khi cùng dữ liệu fit

Chạy:
    python -m src.model_selection --config configs/config.yaml --tune-mode earliest
"""
import argparse
import itertools

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import StandardScaler

from src.baselines import build_count_matrix, iforest_scores, pca_scores, tfidf_transform
from src.config import load_config
from src.evaluate import compute_metrics, recall_by_group, save_results, score_metrics, structural_anomaly_mask
from src.utils import get_logger, load_json

logger = get_logger(__name__)


def make_features(name: str, X_fit: np.ndarray, X_other: list[np.ndarray]) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    Biến đổi ma trận đếm; mọi thống kê (mean, std, idf) chỉ tính trên X_fit.

    Input : name    — "standard" | "raw" | "tfidf" | "presence" | "log1p"
            X_fit   — ma trận dùng để fit
            X_other — các ma trận cần biến đổi theo cùng thống kê
    Output: (F_fit, [F_other...])
    """
    if name == "standard":
        scaler = StandardScaler().fit(X_fit)
        return scaler.transform(X_fit), [scaler.transform(X) for X in X_other]
    if name == "tfidf":
        sizes = [len(X) for X in X_other]
        F_fit, F_rest = tfidf_transform(X_fit, np.vstack(X_other))
        return F_fit, np.split(F_rest, np.cumsum(sizes)[:-1])
    func = {"raw": lambda X: X, "presence": lambda X: (X > 0).astype(np.float64), "log1p": np.log1p}[name]
    return func(X_fit), [func(X) for X in X_other]


def run_config(method: str, params: dict, X: dict[str, np.ndarray], seed: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit một phương án và tính điểm bất thường.

    Input : method — "pca" | "iforest"
            params — dict có features, param (n_components / n_estimators), fit_on
            X      — dict ma trận đếm {"train_normal", "tune", "eval"}
    Output: (điểm trên tune, điểm trên eval)
    """
    X_fit = X["train_normal"] if params["fit_on"] == "train_normal" else X["tune"]
    F_fit, (F_tune, F_eval) = make_features(params["features"], X_fit, [X["tune"], X["eval"]])
    if method == "pca":
        _, (s_tune, s_eval) = pca_scores(F_fit, [F_tune, F_eval], params["param"])
    else:
        _, (s_tune, s_eval) = iforest_scores(F_fit, [F_tune, F_eval], params["param"], seed)
    return s_tune, s_eval


def best_threshold(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """
    Ngưỡng cho F1 cao nhất; session bị coi là bất thường khi score >= ngưỡng.

    Output: (ngưỡng, F1 tại ngưỡng đó). Khi bằng F1, lấy ngưỡng nhỏ nhất.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    best = int(np.argmax(f1[:-1]))  # phần tử cuối của precision/recall không ứng với ngưỡng nào
    return float(thresholds[best]), float(f1[best])


def choose_and_evaluate(params: dict, s_tune: np.ndarray, s_eval: np.ndarray, y: dict[str, np.ndarray]) -> dict:
    """
    Chọn ngưỡng trên tune rồi chấm trên eval: P/R/F1 tại ngưỡng đó + ROC-AUC/AP (không phụ thuộc ngưỡng).

    Nếu y có "eval_structural" (mặt nạ structural_anomaly_mask của tập eval) thì thêm recall theo 2 nhóm bất thường.
    """
    threshold, tune_f1 = best_threshold(y["tune"], s_tune)
    pred = (s_eval >= threshold).astype(int)
    eval_metrics = {**compute_metrics(y["eval"], pred), **score_metrics(y["eval"], s_eval)}
    if "eval_structural" in y:
        eval_metrics.update(recall_by_group(y["eval"], pred, y["eval_structural"]))
    return {"params": params, "threshold": threshold, "tune_f1": tune_f1, "eval": eval_metrics}


def deeplog_session_scores(results_dir, test: pd.DataFrame) -> np.ndarray | None:
    """Điểm DeepLog của từng session test = rank lớn nhất của sự kiện thật; None nếu chưa có deeplog_ranks.pkl."""
    path = results_dir / "deeplog_ranks.pkl"
    if not path.exists():
        return None
    ranks = test[["BlockId"]].merge(pd.read_pickle(path)[["BlockId", "Ranks"]], on="BlockId", how="left")["Ranks"]
    assert ranks.notna().all(), f"{path.name} thiếu rank cho một số session test"
    return np.array([r.max() if len(r) else 0 for r in ranks], dtype=np.float64)


def load_test_with_tune_mask(splits_dir, sel: dict, tune_mode: str | None = None) -> tuple[pd.DataFrame, np.ndarray, str]:
    """
    Đọc test.pkl (sắp theo StartTime) và tạo mặt nạ is_tune theo select.tune_ratio / tune_mode / tune_seed.

    Dùng chung cho model_selection.py và hybrid_detect.py để mọi phương pháp được chấm trên CÙNG tập eval.
    Output: (test, is_tune, tune_mode)
    """
    test = pd.read_pickle(splits_dir / "test.pkl").sort_values("StartTime", kind="stable").reset_index(drop=True)
    tune_mode = tune_mode or sel["tune_mode"]
    n_tune = round(len(test) * sel["tune_ratio"])
    if tune_mode == "earliest":
        is_tune = np.arange(len(test)) < n_tune
    else:
        is_tune = np.zeros(len(test), dtype=bool)
        is_tune[np.random.default_rng(sel["tune_seed"]).choice(len(test), n_tune, replace=False)] = True
    return test, is_tune, tune_mode


def main() -> None:
    parser = argparse.ArgumentParser(description="Chọn phương án trên tập tune, báo cáo trên tập eval")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--tune-mode", choices=["earliest", "random"], default=None,
                        help="ghi đè select.tune_mode trong config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths, sel = cfg["paths"], cfg["select"]
    num_keys = len(load_json(paths["event_index"])) + 1

    splits = {name: pd.read_pickle(paths["splits_dir"] / f"{name}.pkl") for name in ("train", "val")}
    test, is_tune, tune_mode = load_test_with_tune_mask(paths["splits_dir"], sel, args.tune_mode)

    train_seqs = splits["train"]["EventSequence"].tolist() + splits["val"]["EventSequence"].tolist()
    X_test = build_count_matrix(test["EventSequence"].tolist(), num_keys)
    X = {"train_normal": build_count_matrix(train_seqs, num_keys), "tune": X_test[is_tune], "eval": X_test[~is_tune]}
    y = {"tune": test["Label"].to_numpy()[is_tune], "eval": test["Label"].to_numpy()[~is_tune]}
    sizes = {name: {"total": int(len(labels)), "anomaly": int(labels.sum())} for name, labels in y.items()}
    y["eval_structural"] = structural_anomaly_mask(test["EventSequence"][~is_tune].tolist(), pd.read_pickle(paths["sessions"]))
    logger.info("tune_mode=%s | tập tune / eval: %s", tune_mode, sizes)

    grids = {
        "pca": (sel["pca_features"], sel["pca_n_components"]),
        "iforest": (sel["iforest_features"], sel["iforest_n_estimators"]),
    }
    candidates: dict[str, list[dict]] = {}
    for method, (features_grid, param_grid) in grids.items():
        for features, param, fit_on in itertools.product(features_grid, param_grid, sel["fit_on"]):
            params = {"features": features, "param": param, "fit_on": fit_on}
            candidates.setdefault(method, []).append(choose_and_evaluate(params, *run_config(method, params, X, cfg["seed"]), y))
            logger.info("%s %s: F1 tune = %.3f", method, params, candidates[method][-1]["tune_f1"])

    deeplog_scores = deeplog_session_scores(paths["results_dir"], test)
    if deeplog_scores is None:
        logger.warning("Không thấy deeplog_ranks.pkl trong %s — chạy deeplog_scores.py trước", paths["results_dir"])
    else:
        choice = choose_and_evaluate({}, deeplog_scores[is_tune], deeplog_scores[~is_tune], y)
        choice["params"] = {"top_g": int(choice["threshold"]) - 1}  # max rank >= ngưỡng ⇔ max rank > g
        candidates["deeplog"] = [choice]

    def pick(choices: list[dict]) -> dict:
        return max(choices, key=lambda c: c["tune_f1"])  # max() giữ phương án gặp trước khi bằng điểm

    summary = {"tune_mode": tune_mode, "tune_ratio": sel["tune_ratio"], "sizes": sizes,
               **{method: pick(choices) for method, choices in candidates.items()},
               "by_fit_on": {method: {fit_on: pick([c for c in candidates[method] if c["params"]["fit_on"] == fit_on])
                                      for fit_on in sel["fit_on"]} for method in grids}}
    rows = [{"method": method, **c["params"], "threshold": c["threshold"], "tune_f1": c["tune_f1"]}
            for method, choices in candidates.items() for c in choices]
    pd.DataFrame(rows).to_csv(paths["results_dir"] / f"model_selection_{tune_mode}_grid.csv", index=False)
    save_results(summary, paths["results_dir"] / f"model_selection_{tune_mode}.json")

    for name, res in [(m, summary[m]) for m in candidates] + [
            (f"{m}/{f}", r) for m, by_fit in summary["by_fit_on"].items() for f, r in by_fit.items()]:
        ev = res["eval"]
        logger.info("%-22s %s | F1 tune %.3f | eval: P %.3f R %.3f F1 %.3f ROC-AUC %.3f AP %.3f | recall cấu trúc %.3f, "
                    "thứ tự/số lần %.3f", name, res["params"], res["tune_f1"], ev["precision"], ev["recall"], ev["f1"],
                    ev["roc_auc"], ev["average_precision"], ev["recall_structural"], ev["recall_sequence"])


if __name__ == "__main__":
    main()
