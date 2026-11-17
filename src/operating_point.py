"""
BƯỚC 11 — Chọn điểm vận hành theo NGÂN SÁCH cảnh báo giả, thay vì theo F1 cao nhất.

Vì sao: F1 cao nhất không phải điều người trực hệ thống cần. Họ có một ngân sách: "tôi xử lý được tối đa B cảnh báo
giả mỗi giờ". Câu hỏi đúng là: với ngân sách đó, mỗi phương pháp bắt được bao nhiêu bất thường?
Ở điểm F1 cao nhất (bước 7–8), DeepLog / hybrid báo nhầm ~70–100 lần mỗi giờ, 2 luật bất biến chỉ ~7–10.

Cách làm (cùng tập tune / eval với bước 7–8, không phương pháp nào được ưu ái):
  - Với mỗi ngân sách B (operating.budgets_fp_per_hour) và mỗi phương pháp, chọn trên tập tune tham số + ngưỡng
    cho NHIỀU bất thường bị bắt nhất (TP) mà số cảnh báo giả <= B × (số giờ tập tune đại diện);
    bằng TP thì lấy phương án ít cảnh báo giả hơn. Không có phương án nào thoả → không cảnh báo gì (recall 0).
  - Chấm đúng 1 lần trên tập eval: recall, precision, FP/giờ, và ngân sách có còn giữ được trên eval không.
Phương pháp: DeepLog (ngưỡng trên max rank ⇔ chọn g); PCA và Isolation Forest (lưới của mục `select`, tách riêng
theo dữ liệu học fit_on như bước 7); 2 luật bất biến (chọn margin); hybrid (chọn margin, g — có thêm lựa chọn
g = None, tức tắt tầng DeepLog, nên hybrid không bao giờ tệ hơn 2 luật trên tập tune). Luật giống hệt hybrid_detect.py.
"Rules" chính là tầng cảnh báo 1 (độ chính xác cao); hybrid = tầng 1 + cảnh báo chỉ do DeepLog (tầng 2).

Input : như hybrid_detect.py; DeepLog lấy deeplog_ranks.pkl trong --ranks-dir (mặc định results_dir)
        → chạy được cho model chính, từng seed (results_dir/seeds/seed<s>/<mode>) và ensemble (results_dir/ensemble/<mode>)
Output: <ranks-dir>/operating_points_<tune_mode>.csv (+ .json có meta)
        --collect: results_dir/operating_points_summary.csv — 1 model (trung bình ± độ lệch chuẩn qua seed) và ensemble

Chạy:
    python -m src.operating_point --config configs/config.yaml --tune-mode random
    python -m src.operating_point --config configs/config.yaml --tune-mode random --ranks-dir outputs/results/ensemble/random
    python -m src.operating_point --config configs/config.yaml --collect
"""
import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from src.baselines import build_count_matrix
from src.config import PROJECT_ROOT, load_config
from src.deeplog_scores import RANKS_FILE, load_ranks_source
from src.hybrid_detect import learn_normal_profile, time_to_kth, unseen_alert_times
from src.evaluate import compute_metrics, recall_by_unseen, unseen_event_mask
from src.model_selection import load_test_with_tune_mask, run_config
from src.utils import get_logger, load_json, run_metadata, save_json

logger = get_logger(__name__)

TUNE_MODES = ("earliest", "random")


def threshold_under_budget(y: np.ndarray, scores: np.ndarray, max_fp: float) -> tuple[float, int, int]:
    """
    Ngưỡng (cảnh báo khi score >= ngưỡng) cho TP lớn nhất mà FP <= max_fp; bằng TP → FP nhỏ hơn.

    Chỉ cắt ở ranh giới giữa 2 giá trị điểm khác nhau (các session cùng điểm luôn cùng kết quả).
    Output: (ngưỡng, TP, FP); ngưỡng = inf nếu không có ngưỡng nào thoả (không cảnh báo gì).
    """
    order = np.argsort(-scores, kind="stable")
    s, lab = scores[order], y[order]
    tp, fp = np.cumsum(lab), np.cumsum(1 - lab)
    is_cut = np.r_[s[1:] != s[:-1], True]
    valid = np.flatnonzero(is_cut & (fp <= max_fp))
    if not len(valid):
        return np.inf, 0, 0
    best = valid[np.argmax(tp[valid])]  # tp không giảm theo vị trí → vị trí đầu tiên đạt max có FP nhỏ nhất
    return float(s[best]), int(tp[best]), int(fp[best])


def pick_under_budget(candidates: dict, y: np.ndarray, max_fp: float):
    """Khoá (trong candidates: khoá → mảng 0/1 trên tune) cho TP lớn nhất mà FP <= max_fp; None nếu không có."""
    best_key, best = None, (-1, np.inf)
    for key, pred in candidates.items():
        tp, fp = int(((pred == 1) & (y == 1)).sum()), int(((pred == 1) & (y == 0)).sum())
        if fp <= max_fp and (tp > best[0] or (tp == best[0] and fp < best[1])):
            best_key, best = key, (tp, fp)
    return best_key


def eval_row(y: np.ndarray, pred: np.ndarray, hours: float, budget: float) -> dict:
    """Recall / precision / FP mỗi giờ trên eval, và ngân sách có còn giữ được không."""
    tp, fp = int(((pred == 1) & (y == 1)).sum()), int(((pred == 1) & (y == 0)).sum())
    return {**compute_metrics(y, pred), "recall": tp / max(int(y.sum()), 1), "precision": tp / (tp + fp) if tp + fp else float("nan"),
            "fp_per_hour": fp / hours, "within_budget": fp / hours <= budget, "tp": tp, "fp": fp}


def prepare(cfg: dict, tune_mode: str, ranks_dir: Path) -> dict:
    """Dựng mọi thứ cần cho việc chọn: nhãn, mặt nạ tune, điểm DeepLog, 2 luật, số giờ — giống hybrid_detect.py."""
    paths, hcfg, sel = cfg["paths"], cfg["hybrid"], cfg["select"]
    replica_event = load_json(paths["event_index"])[hcfg["replica_event"]]
    context = pd.read_pickle(paths["block_context"])
    train = pd.concat([pd.read_pickle(paths["splits_dir"] / f"{n}.pkl") for n in ("train", "val")])
    test, is_tune, tune_mode = load_test_with_tune_mask(paths["splits_dir"], sel, tune_mode)
    df = test.merge(context, on="BlockId", how="left").merge(pd.read_pickle(ranks_dir / RANKS_FILE), on="BlockId", how="left")
    assert len(df) == len(test) and df["EventTimes"].notna().all() and df["Ranks"].notna().all()

    t0 = df["EventTimes"].map(lambda t: t[0]).to_numpy(dtype=float)
    t_end = df["EventTimes"].map(lambda t: t[-1]).to_numpy(dtype=float)
    profile = learn_normal_profile(train.merge(context, on="BlockId"), replica_event)
    k, deadline = profile["replica_count"], profile["deadline_sec"]
    unseen = np.isfinite(unseen_alert_times(df, profile["seen_events"]))
    t_kth = np.array([time_to_kth(s, t, replica_event, k) for s, t in zip(df["EventSequence"], df["EventTimes"])])
    stream_end = t_end.max()
    rules = {m: unseen | ((t_kth > deadline * m) & (t0 + deadline * m <= stream_end)) for m in hcfg["deadline_margins"]}

    span_hours = (t_end.max() - t0.min()) / 3600
    num_keys = len(load_json(paths["event_index"])) + 1
    X_test = build_count_matrix(test["EventSequence"].tolist(), num_keys)
    train_seqs = train["EventSequence"].tolist()
    return {
        "tune_mode": tune_mode, "y": test["Label"].to_numpy(), "is_tune": is_tune,
        "max_rank": df["Ranks"].map(lambda r: int(r.max()) if len(r) else 0).to_numpy(dtype=float),
        "rules": rules, "g_values": list(range(1, hcfg["max_top_g"] + 1)),
        "unseen": unseen_event_mask(test["EventSequence"], pd.read_pickle(paths["splits_dir"] / "train.pkl")["EventSequence"]),
        "tune_hours": span_hours * sel["tune_ratio"], "eval_hours": span_hours * (1 - sel["tune_ratio"]),
        "X": {"train_normal": build_count_matrix(train_seqs, num_keys), "tune": X_test[is_tune], "eval": X_test[~is_tune]},
    }


def baseline_scores(cfg: dict, X: dict, methods=None) -> dict:
    """Điểm trên tune / eval của MỌI phương án PCA / IForest trong lưới `select` (như model_selection.py)."""
    sel = cfg["select"]
    grids = {"pca": (sel["pca_features"], sel["pca_n_components"]),
             "iforest": (sel["iforest_features"], sel["iforest_n_estimators"])}
    out = {}
    for method, (features_grid, param_grid) in grids.items():
        if methods is not None and method not in methods:
            continue
        for features, param, fit_on in itertools.product(features_grid, param_grid, sel["fit_on"]):
            params = {"features": features, "param": param, "fit_on": fit_on}
            out.setdefault(method, []).append((params, *run_config(method, params, X, cfg["seed"])))
    return out


def evaluate_budgets(cfg: dict, data: dict, baselines: dict) -> pd.DataFrame:
    """Một dòng cho mỗi ngân sách × phương pháp: tham số chọn trên tune + kết quả trên eval."""
    y, tune = data["y"], data["is_tune"]
    y_tune, y_eval = y[tune], y[~tune]
    rows = []
    variant_rows = []
    for budget in cfg["operating"]["budgets_fp_per_hour"]:
        max_fp = budget * data["tune_hours"]

        def add(method, params, pred_eval):
            rows.append({"budget_fp_per_hour": budget, "method": method, "params": str(params),
                         **eval_row(y_eval, pred_eval, data["eval_hours"], budget),
                         **recall_by_unseen(y_eval, pred_eval, data["unseen"][~tune])})

        thr, _, _ = threshold_under_budget(y_tune, data["max_rank"][tune], max_fp)
        add("deeplog", {"top_g": None if np.isinf(thr) else int(thr) - 1}, (data["max_rank"][~tune] >= thr).astype(int))

        for method, choices in baselines.items():
            for params, s_tune, s_eval in choices:
                threshold, tune_tp, tune_fp = threshold_under_budget(y_tune, s_tune, max_fp)
                pred_eval = (s_eval >= threshold).astype(int)
                variant_rows.append({'budget_fp_per_hour': budget, 'method': method, **params,
                                     'threshold': threshold, 'tune_tp': tune_tp, 'tune_fp': tune_fp,
                                     **eval_row(y_eval, pred_eval, data['eval_hours'], budget),
                                     **recall_by_unseen(y_eval, pred_eval, data['unseen'][~tune])})
            for fit_on in [None, *cfg["select"]["fit_on"]]:
                best = max(((threshold_under_budget(y_tune, s_tune, max_fp), params, s_eval)
                            for params, s_tune, s_eval in choices if fit_on is None or params["fit_on"] == fit_on),
                           key=lambda c: (c[0][1], -c[0][2]))
                (thr, _, _), params, s_eval = best
                add(f"{method}/{fit_on}" if fit_on else method, {**params, "threshold": thr}, (s_eval >= thr).astype(int))

        rules = data["rules"]
        m = pick_under_budget({m: r[tune].astype(int) for m, r in rules.items()}, y_tune, max_fp)
        add("rules", {"deadline_margin": m}, rules[m][~tune].astype(int) if m is not None else np.zeros(len(y_eval), int))

        # g = None: tắt tầng DeepLog (chỉ còn luật) — với ngân sách nhỏ, DeepLog ở g lớn nhất vẫn có thể vượt ngân sách
        hybrid = {(m, g): rules[m] | (data["max_rank"] > g) if g is not None else rules[m]
                  for m, g in itertools.product(rules, data["g_values"] + [None])}
        key = pick_under_budget({k: p[tune].astype(int) for k, p in hybrid.items()}, y_tune, max_fp)
        add("hybrid", {"deadline_margin": key[0], "top_g": key[1]} if key else None,
            hybrid[key][~tune].astype(int) if key else np.zeros(len(y_eval), int))
    table = pd.DataFrame(rows)
    table.attrs['baseline_variants'] = variant_rows
    return table


def collect(cfg: dict) -> pd.DataFrame:
    """Gom operating_points_<mode>.csv của mọi seed và của ensemble → trung bình ± độ lệch chuẩn theo seed."""
    results_dir = cfg["paths"]["results_dir"]
    frames = []
    for mode in TUNE_MODES:
        for path in sorted((results_dir / "seeds").glob(f"seed*/{mode}/operating_points_{mode}.csv")):
            frames.append(pd.read_csv(path).assign(tune_mode=mode, source="1 model", seed=path.parts[-3]))
        ens = results_dir / "ensemble" / mode / f"operating_points_{mode}.csv"
        if ens.exists():
            frames.append(pd.read_csv(ens).assign(tune_mode=mode, source="ensemble", seed="ensemble"))
    df = pd.concat(frames, ignore_index=True)
    summary = (df.groupby(["tune_mode", "budget_fp_per_hour", "method", "source"], sort=False)
               .agg(recall_mean=("recall", "mean"), recall_std=("recall", "std"), precision_mean=("precision", "mean"),
                    f1_mean=("f1", "mean"), f1_std=("f1", "std"),
                    recall_unseen_mean=("recall_unseen", "mean"), recall_unseen_std=("recall_unseen", "std"),
                    recall_known_mean=("recall_known", "mean"), recall_known_std=("recall_known", "std"),
                    fp_per_hour_mean=("fp_per_hour", "mean"), fp_per_hour_max=("fp_per_hour", "max"),
                    within_budget=("within_budget", "mean"), runs=("recall", "size"))
               .reset_index())
    return summary


def write_outputs(cfg, ranks_dir, data, baselines):
    table = evaluate_budgets(cfg, data, baselines)
    mode = data['tune_mode']
    table.to_csv(ranks_dir / f'operating_points_{mode}.csv', index=False)
    pd.DataFrame(table.attrs['baseline_variants']).to_csv(ranks_dir / f'operating_points_{mode}_grid.csv', index=False)
    save_json({'meta': run_metadata(cfg), 'deeplog_source': load_ranks_source(ranks_dir, cfg), 'tune_mode': mode,
               'tune_hours': data['tune_hours'], 'eval_hours': data['eval_hours'], 'rows': table.to_dict('records')},
              ranks_dir / f'operating_points_{mode}.json')
    return table


def all_runs(cfg, resume=False):
    """Share deterministic PCA scores across seeds and seed-42 baseline scores with ensemble."""
    base = cfg['paths']['results_dir']
    for mode in TUNE_MODES:
        directories = [base / 'seeds' / f'seed{seed}' / mode for seed in cfg['robustness']['seeds']]
        directories.append(base / 'ensemble' / mode)
        def complete(directory):
            from src.audit_results import differences
            if not all((directory / f'operating_points_{mode}{suffix}').exists()
                       for suffix in ('.csv', '.json', '_grid.csv')):
                return False
            value = load_json(directory / f'operating_points_{mode}.json')
            run_cfg = load_config(directory / 'config.yaml')
            return (not differences(value.get('meta', {}).get('config', {}), run_cfg)
                    and len(value['rows']) == len(cfg['operating']['budgets_fp_per_hour']) * 9
                    and all('recall_known' in row for row in value['rows']))

        if resume and all(complete(directory) for directory in directories):
            logger.info('Retaining completed operating mode: %s / %s', base, mode)
            continue
        first = base / 'seeds' / f"seed{cfg['robustness']['seeds'][0]}" / mode
        data = prepare(cfg, mode, first)
        test, _, _ = load_test_with_tune_mask(cfg['paths']['splits_dir'], cfg['select'], mode)
        pca = baseline_scores(cfg, data['X'], methods=['pca'])
        ensemble_baselines = None
        for seed in cfg['robustness']['seeds']:
            ranks_dir = base / 'seeds' / f'seed{seed}' / mode
            run_cfg = load_config(ranks_dir / 'config.yaml')
            baselines = {**pca, **baseline_scores(run_cfg, data['X'], methods=['iforest'])}
            ranks = pd.read_pickle(ranks_dir / RANKS_FILE).set_index('BlockId').reindex(test.BlockId)
            if ranks.Ranks.isna().any() or not ranks.index.is_unique:
                raise ValueError(f'Invalid ranks: {ranks_dir}')
            data['max_rank'] = ranks.Ranks.map(lambda r: int(r.max()) if len(r) else 0).to_numpy(dtype=float)
            write_outputs(run_cfg, ranks_dir, data, baselines)
            if seed == cfg['seed']:
                ensemble_baselines = baselines
            logger.info('Operating points done: %s seed %s', mode, seed)
        ranks_dir = base / 'ensemble' / mode
        ranks = pd.read_pickle(ranks_dir / RANKS_FILE).set_index('BlockId').reindex(test.BlockId)
        if ranks.Ranks.isna().any() or not ranks.index.is_unique:
            raise ValueError(f'Invalid ensemble ranks: {ranks_dir}')
        data['max_rank'] = ranks.Ranks.map(lambda r: int(r.max()) if len(r) else 0).to_numpy(dtype=float)
        if ensemble_baselines is None:
            ensemble_baselines = {**pca, **baseline_scores(cfg, data['X'], methods=['iforest'])}
        write_outputs(load_config(ranks_dir / 'config.yaml'), ranks_dir, data, ensemble_baselines)
    out = base / 'operating_points_summary.csv'
    collect(cfg).to_csv(out, index=False)
    save_json(run_metadata(cfg), out.with_suffix('.meta.json'))


def main() -> None:
    parser = argparse.ArgumentParser(description="Bước 11: điểm vận hành theo ngân sách cảnh báo giả")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--tune-mode", choices=TUNE_MODES, default=None, help="ghi đè select.tune_mode")
    parser.add_argument("--ranks-dir", default=None, help="thư mục chứa deeplog_ranks.pkl và nơi ghi kết quả")
    parser.add_argument("--collect", action="store_true", help="chỉ gom kết quả của các seed + ensemble")
    parser.add_argument('--all-runs', action='store_true', help='Evaluate both modes for every configured seed and ensemble')
    parser.add_argument('--resume', action='store_true', help='With --all-runs, retain already complete tuning modes')
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.all_runs:
        all_runs(cfg, resume=args.resume)
        return
    if args.collect:
        summary = collect(cfg)
        out = cfg["paths"]["results_dir"] / "operating_points_summary.csv"
        summary.to_csv(out, index=False)
        save_json(run_metadata(cfg), out.with_suffix(".meta.json"))
        logger.info("Đã lưu %s\n%s", out, summary.round(3).to_string())
        return

    ranks_dir = Path(args.ranks_dir) if args.ranks_dir else cfg["paths"]["results_dir"]
    ranks_dir = ranks_dir if ranks_dir.is_absolute() else PROJECT_ROOT / ranks_dir
    run_cfg_path = ranks_dir / "config.yaml"  # thư mục của seed_study / ensemble có config đã sinh cho lần chạy đó
    if ranks_dir != cfg["paths"]["results_dir"] and run_cfg_path.exists():
        cfg["seed"] = load_config(run_cfg_path)["seed"]  # cùng seed với DeepLog / IForest của lần chạy đó
        logger.info("Dùng seed %d theo %s", cfg["seed"], run_cfg_path)
    data = prepare(cfg, args.tune_mode, ranks_dir)
    table = write_outputs(cfg, ranks_dir, data, baseline_scores(cfg, data['X']))

    mode = data['tune_mode']
    logger.info("tune_mode=%s | %s\n%s", mode, ranks_dir,
                table[["budget_fp_per_hour", "method", "recall", "precision", "fp_per_hour", "within_budget"]].round(3).to_string())


if __name__ == "__main__":
    main()
