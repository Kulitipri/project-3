"""
BƯỚC 6 — Baseline để so sánh với DeepLog: PCA và Isolation Forest.

Hai baseline này KHÔNG nhìn thứ tự sự kiện, chỉ đếm số lần mỗi sự kiện xuất hiện trong session
(ma trận đếm sự kiện — event count matrix). So sánh với DeepLog cho biết việc mô hình hoá
thứ tự sự kiện có thực sự giúp ích hay không.

Mỗi phương pháp được fit theo 2 cách:
  - "normal"   : fit trên session Normal của train + val (semi-supervised, cùng dữ liệu với DeepLog).
                 Ngưỡng = phân vị normal_threshold_percentile của điểm trên dữ liệu fit (~1% báo nhầm).
  - "unlabeled": fit trên unlabeled_ratio session SỚM NHẤT của test, BỎ nhãn (unsupervised, kiểu Loglizer).
                 Đây là log "quá khứ" ngay sau train: có lẫn bất thường nhưng chưa ai gán nhãn.
                 Ngưỡng = phân vị 100 × (1 - contamination), tức giả định ~contamination session là bất thường.
Cả 4 tổ hợp được chấm trên CÙNG tập eval = phần test còn lại, nên so sánh được với nhau.
Ngoài P/R/F1 ở ngưỡng mặc định còn có ROC-AUC và AP (không phụ thuộc ngưỡng, xem evaluate.score_metrics).
Đặc trưng / ngưỡng tốt nhất sẽ được chọn ở bước 7 (tuần 3) trên tập có nhãn, không chỉnh ở đây.

Vì sao cần cả 2 cách fit:
  Cây của Isolation Forest chỉ cắt trên cột có biến thiên trong dữ liệu fit. Session Normal của train
  không chứa phần lớn sự kiện lỗi, nên khi fit trên đó IForest bỏ qua chính các sự kiện báo lỗi
  (mục "diagnostics" trong file kết quả đo điều này). PCA không bị vì TF-IDF cho sự kiện chưa từng
  thấy trọng số rất lớn; ngược lại, khi fit trên dữ liệu không nhãn, sự kiện lỗi có trong dữ liệu fit
  nên idf của chúng nhỏ đi và PCA kém hơn.

Input : data/processed/splits/train.pkl, val.pkl, test.pkl
        data/processed/event_index.json
Output: outputs/results/baselines.json

Chạy:
    python -m src.baselines --config configs/config.yaml
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

from src.config import load_config
from src.evaluate import compute_metrics, save_results, score_metrics
from src.utils import get_logger, load_json

logger = get_logger(__name__)


def build_count_matrix(sequences: list[list[int]], num_keys: int) -> np.ndarray:
    """
    Tạo ma trận đếm sự kiện.

    Input : sequences — list session (mỗi session là list[int])
            num_keys  — len(event_index) + 1
    Output: np.ndarray kích thước (số session, num_keys);
            phần tử [i, k] = số lần sự kiện k xuất hiện trong session i.
    """
    X = np.zeros((len(sequences), num_keys), dtype=np.float64)
    for i, seq in enumerate(sequences):
        if seq:
            X[i] = np.bincount(seq, minlength=num_keys)[:num_keys]
    return X


def tfidf_transform(X_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Đánh trọng số TF-IDF rồi trừ trung bình — tiền xử lý chuẩn cho PCA của Xu et al. (SOSP 2009)
    và thư viện Loglizer (He et al., ISSRE 2016).

    Input : X_train, X_test — ma trận đếm sự kiện
    Output: (Z_train, Z_test) — đã nhân idf và trừ trung bình (cả hai đều fit trên X_train)
    idf[k] = log(N / df[k]), df[k] = số session train có sự kiện k:
      - sự kiện có trong MỌI session train → idf = 0, bị bỏ qua. Đây là các sự kiện vòng đời
        (allocate, receive, delete, verify...) mà số lần xuất hiện phụ thuộc block bị cắt bởi
        cuối log hay chưa, không phải dấu hiệu lỗi.
      - sự kiện chưa từng xuất hiện trong train → idf rất lớn, nên chỉ cần xuất hiện là bị chú ý.
    Không dùng StandardScaler: nó chia cho độ lệch chuẩn trên train, mà với chia chronological
    các cột này gần như hằng số trên train (std ≈ 0,1), nên lệch nhỏ ở test bị khuếch đại
    hàng chục lần và gần như mọi session Normal ở test bị báo bất thường.
    """
    df = (X_train > 0).sum(axis=0)
    idf = np.log(len(X_train) / (df + 1e-8))
    Z_train, Z_test = X_train * idf, X_test * idf
    mean = Z_train.mean(axis=0)
    return Z_train - mean, Z_test - mean


def pca_scores(F_fit: np.ndarray, F_other: list[np.ndarray], n_components: float) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    Điểm bất thường PCA = sai số tái tạo ||x - x_hat||^2 (PCA fit trên F_fit).

    Session Normal nằm gần không gian con chính nên sai số nhỏ; session Anomaly có sai số lớn.
    Output: (điểm trên F_fit, [điểm trên từng ma trận trong F_other])
    """
    pca = PCA(n_components=n_components, svd_solver="full").fit(F_fit)

    def reconstruction_error(F: np.ndarray) -> np.ndarray:
        return ((F - pca.inverse_transform(pca.transform(F))) ** 2).sum(axis=1)

    return reconstruction_error(F_fit), [reconstruction_error(F) for F in F_other]


def iforest_scores(F_fit: np.ndarray, F_other: list[np.ndarray], n_estimators: int,
                   seed: int) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    Điểm bất thường Isolation Forest = -score_samples (càng lớn càng bất thường).

    Không dùng IsolationForest.predict(): ngưỡng mặc định của sklearn không gắn với dữ liệu,
    ngưỡng được đặt bằng phân vị điểm trên dữ liệu fit (xem main).
    Output: (điểm trên F_fit, [điểm trên từng ma trận trong F_other])
    """
    model = IsolationForest(n_estimators=n_estimators, random_state=seed, n_jobs=-1).fit(F_fit)
    return -model.score_samples(F_fit), [-model.score_samples(F) for F in F_other]


def split_unlabeled(test: pd.DataFrame, ratio: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Tách test thành (phần không nhãn, phần eval) theo thời gian.

    Phần không nhãn = ratio session có StartTime SỚM NHẤT; chỉ dùng EventSequence, KHÔNG dùng Label.
    """
    test = test.sort_values("StartTime", kind="stable").reset_index(drop=True)
    n_unlabeled = round(len(test) * ratio)
    return test.iloc[:n_unlabeled], test.iloc[n_unlabeled:]


def blind_spots(X_fit: np.ndarray, X_eval: np.ndarray, y_eval: np.ndarray) -> dict:
    """
    Đo phần dữ liệu mà cây của Isolation Forest không "nhìn thấy" khi fit trên X_fit.

    Output: dict
      event_types                 — số loại sự kiện (cột 0 là PAD nên không tính)
      event_types_with_variance   — số loại sự kiện có biến thiên trong X_fit (cây chỉ cắt được trên các cột này)
      anomaly_with_unseen_event   — tỉ lệ session Anomaly ở eval chứa sự kiện không có trong X_fit
      normal_with_unseen_event    — tỉ lệ tương ứng ở session Normal
    """
    unseen = (X_eval[:, X_fit.sum(axis=0) == 0] > 0).any(axis=1)
    return {
        "event_types": int(X_fit.shape[1] - 1),
        "event_types_with_variance": int((X_fit[:, 1:].std(axis=0) > 0).sum()),
        "anomaly_with_unseen_event": float(unseen[y_eval == 1].mean()),
        "normal_with_unseen_event": float(unseen[y_eval == 0].mean()),
    }


def describe(df: pd.DataFrame) -> dict:
    """Kích thước, số Anomaly và khoảng thời gian của một tập session (để ghi vào file kết quả)."""
    return {
        "total": int(len(df)),
        "anomaly": int(df["Label"].sum()),
        "start": str(df["StartTime"].min()),
        "end": str(df["StartTime"].max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Bước 6: baseline PCA và Isolation Forest")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths, base = cfg["paths"], cfg["baseline"]
    num_keys = len(load_json(paths["event_index"])) + 1

    splits = {name: pd.read_pickle(paths["splits_dir"] / f"{name}.pkl") for name in ("train", "val", "test")}
    train_normal = pd.concat([splits["train"], splits["val"]])
    unlabeled, evaluation = split_unlabeled(splits["test"], base["unlabeled_ratio"])

    X_fit = {
        "normal": build_count_matrix(train_normal["EventSequence"].tolist(), num_keys),
        "unlabeled": build_count_matrix(unlabeled["EventSequence"].tolist(), num_keys),
    }
    X_eval = build_count_matrix(evaluation["EventSequence"].tolist(), num_keys)
    y_eval = evaluation["Label"].to_numpy()
    percentiles = {"normal": base["normal_threshold_percentile"],
                   "unlabeled": round(100 * (1 - base["contamination"]), 6)}

    results = {
        "sets": {"normal": describe(train_normal), "unlabeled": describe(unlabeled), "eval": describe(evaluation)},
        "pca": {}, "iforest": {}, "diagnostics": {},
    }
    for fit_on, X in X_fit.items():
        Z_fit, Z_eval = tfidf_transform(X, X_eval)
        scores = {
            "pca": pca_scores(Z_fit, [Z_eval], base["pca_n_components"]),
            "iforest": iforest_scores(X, [X_eval], base["iforest_n_estimators"], cfg["seed"]),
        }
        for method, (s_fit, (s_eval,)) in scores.items():
            threshold = np.percentile(s_fit, percentiles[fit_on])
            metrics = compute_metrics(y_eval, (s_eval > threshold).astype(int))
            results[method][fit_on] = {"threshold_percentile": percentiles[fit_on], **metrics,
                                       **score_metrics(y_eval, s_eval)}
            logger.info("%-7s fit=%-9s | P %.3f R %.3f F1 %.3f | ROC-AUC %.3f AP %.3f", method, fit_on,
                        metrics["precision"], metrics["recall"], metrics["f1"],
                        results[method][fit_on]["roc_auc"], results[method][fit_on]["average_precision"])
        results["diagnostics"][fit_on] = blind_spots(X, X_eval, y_eval)
        logger.info("fit=%s: %s", fit_on, results["diagnostics"][fit_on])

    save_results(results, paths["results_dir"] / "baselines.json")


if __name__ == "__main__":
    main()
