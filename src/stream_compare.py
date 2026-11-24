"""
BƯỚC 15 — So sánh cuối các bộ phát hiện theo luồng, tham số chỉ chọn trên tập tune, chốt MỘT LẦN trên phần chưa nhìn.

Ba phát hiện ở bước 12–14 dẫn tới bước này:
  - top-g bỏ độ lớn xác suất: sự kiện hiếm có trong vài session train lọt cả luật "chưa từng thấy" lẫn top-7 của DeepLog.
    → thay bằng ĐỘ BẤT NGỜ −log p, trộn với tần suất chung của sự kiện: p = (1 − eps)·p_LSTM + eps·p_tần_suất (eps, ngưỡng
    chọn trên tune). Phần trộn chống việc mạng quá tự tin ở ngữ cảnh mới (block bị đọc nhiều lần).
  - ngưỡng "chưa từng thấy" là c = 0 cố định; nay "hiếm" = có trong <= c session train, c chọn trên tune như mọi tham số khác.
  - phản hồi của người vận hành: so khớp mẫu chính xác / gần theo số đếm / gần theo biểu diễn LSTM.

Mỗi cấu hình dữ liệu cho một bảng; phần `holdout` là phiên bắt đầu sau streaming.holdout_hour, không dùng để phát triển.
Đối chứng không học sâu đi kèm: luật sự kiện hiếm, mô hình Markov (làm trơn Witten–Bell) cùng phép trộn tần suất.

Output: <results_dir>/stream/compare.csv + compare.json   (results_dir của config đang chạy)
Chạy (~15 phút):
    python -m src.stream_compare --config configs/config.yaml
"""
import argparse
import heapq
import itertools
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import torch

from src.config import load_config
from src.ensemble import load_seed_models
from src.hybrid_detect import learn_normal_profile, ops_metrics, select_on_tune, time_to_kth
from src.model import DeepLogEnsemble
from src.model_selection import load_test_with_tune_mask
from src.stream_offline import completion_times, shortest_equivalent
from src.utils import ensure_dir, get_logger, load_json, run_metadata, save_json

logger = get_logger(__name__)
NUM_EVENTS = 31  # 29 loại sự kiện + PAD + dự phòng; chỉ để cấp mảng đếm


def unigram_frequencies(sequences) -> np.ndarray:
    """Tần suất chung của từng sự kiện trong session train (cộng 0,1 cho sự kiện chưa gặp)."""
    counts = np.zeros(NUM_EVENTS)
    for s in sequences:
        np.add.at(counts, s, 1)
    return (counts + 0.1) / (counts.sum() + 0.1 * NUM_EVENTS)


class Stream:
    """Dữ liệu của một cấu hình: session test, cửa sổ duy nhất, các mảng thời gian. Không chứa quyết định nào."""

    def __init__(self, cfg: dict):
        paths, hcfg = cfg["paths"], cfg["hybrid"]
        self.cfg, self.window = cfg, cfg["model"]["window_size"]
        context = pd.read_pickle(paths["block_context"])
        train = pd.concat([pd.read_pickle(paths["splits_dir"] / f"{n}.pkl") for n in ("train", "val")])
        replica = load_json(paths["event_index"])[hcfg["replica_event"]]
        profile = learn_normal_profile(train.merge(context, on="BlockId"), replica)
        test, self.is_tune, _ = load_test_with_tune_mask(paths["splits_dir"], cfg["select"], cfg["streaming"]["tune_mode"])
        df = test.merge(context[["BlockId", "EventTimes"]], on="BlockId", how="left")
        self.seqs, self.times, self.y = df["EventSequence"].tolist(), df["EventTimes"].tolist(), df["Label"].to_numpy()
        self.n = len(df)
        self.t0 = np.array([t[0] for t in self.times], dtype=float)
        self.t_end = np.array([t[-1] for t in self.times], dtype=float)
        self.stream_end, self.deadline = float(self.t_end.max()), profile["deadline_sec"]
        self.t_kth = np.array([time_to_kth(s, t, replica, profile["replica_count"]) for s, t in zip(self.seqs, self.times)])
        self.train = train["EventSequence"].tolist()
        self.session_freq = Counter(e for s in self.train for e in set(s))
        self.unigram = unigram_frequencies(self.train)
        # cặp (cửa sổ 10 sự kiện, sự kiện đích) duy nhất và vị trí của mọi lần xuất hiện trong luồng
        ids: dict[tuple, int] = {}
        block, stamp, pair, pos = [], [], [], []
        for b, (s, t) in enumerate(zip(self.seqs, self.times)):
            for i in range(self.window, len(s)):
                block.append(b), stamp.append(t[i]), pos.append(i)
                pair.append(ids.setdefault(tuple(s[i - self.window:i + 1]), len(ids)))
        self.block, self.stamp, self.pair, self.pos = map(np.array, (block, stamp, pair, pos))
        self.pairs = torch.tensor(list(ids))
        self.first_row = np.full(self.n, -1)
        self.first_row[self.block[::-1]] = np.arange(len(self.block))[::-1]  # hàng đầu tiên của mỗi block

    # ---- thời điểm cảnh báo của từng thành phần
    def completion(self, margin: int) -> np.ndarray:
        return completion_times(self.t0, self.t_kth, self.deadline * margin, self.stream_end)

    def rare_times(self, c: int, low: int = 0) -> np.ndarray:
        """Thời điểm sự kiện đầu tiên có trong (low − 1, c] session train; low = 0, c = 0 là 'chưa từng thấy'."""
        out = np.full(self.n, np.inf)
        for b, (s, t) in enumerate(zip(self.seqs, self.times)):
            for e, tt in zip(s, t):
                if low <= self.session_freq.get(e, 0) <= c:
                    out[b] = tt
                    break
        return out

    def first_crossing(self, hot_pairs: np.ndarray) -> np.ndarray:
        """Thời điểm cặp 'nóng' đầu tiên của từng block."""
        out = np.full(self.n, np.inf)
        idx = np.flatnonzero(hot_pairs[self.pair])
        order = idx[np.argsort(-self.stamp[idx], kind="stable")]
        out[self.block[order]] = self.stamp[order]
        return out

    # ---- mô hình
    @torch.no_grad()
    def lstm(self, models: list) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(xác suất đích, hạng đích, biểu diễn ẩn của cửa sổ KẾT THÚC tại sự kiện đích) cho từng cặp duy nhất."""
        ens, p, r, h = DeepLogEnsemble(models).eval(), [], [], []
        for x in torch.split(self.pairs, 8192):
            probs = torch.exp(ens(x[:, :self.window]))
            target = probs.gather(1, x[:, self.window:])
            p.append(target[:, 0]), r.append((probs > target).sum(1) + 1)
            h.append(torch.cat([m.lstm(m.embedding(x[:, 1:]))[0][:, -1] for m in models], dim=1))
        return torch.cat(p).double().numpy(), torch.cat(r).numpy(), torch.cat(h).numpy()

    def markov(self, order: int) -> np.ndarray:
        """p(đích | order − 1 sự kiện trước), làm trơn Witten–Bell về bậc thấp hơn, ước lượng trên session train."""
        u, out = self.pairs.numpy(), np.full(len(self.pairs), 1.0 / NUM_EVENTS)
        for k in range(1, order + 1):
            full, ctx, kinds = Counter(), Counter(), defaultdict(set)
            for s in self.train:
                for i in range(k - 1, len(s)):
                    key = tuple(s[i - k + 1:i])
                    full[key, s[i]] += 1
                    ctx[key] += 1
                    kinds[key].add(s[i])
            for j in range(len(u)):
                key, e = tuple(u[j, self.window + 1 - k:self.window].tolist()), int(u[j, self.window])
                total = ctx.get(key, 0)
                if total:
                    lam = total / (total + len(kinds[key]))
                    out[j] = lam * full.get((key, e), 0) / total + (1 - lam) * out[j]
        return out

    def surprisal(self, p: np.ndarray, eps: float) -> np.ndarray:
        return -np.log(np.maximum((1 - eps) * p + eps * self.unigram[self.pairs[:, self.window].numpy()], 1e-30))


def feedback(s: Stream, triggers: dict[int, list[int]], immediate: np.ndarray, completion: np.ndarray, hidden: np.ndarray,
             kind: str, radius: float, k: int, wait: float, delay: float, flip: float, seed: int) -> tuple[np.ndarray, int]:
    """
    Chờ xác nhận + phản hồi (bước 14) với ba cách khái quát: exact (mẫu giống hệt), count (cosin trên log số đếm),
    lstm (cosin trên trung bình trạng thái ẩn của các cửa sổ tới lúc quyết định). flip = tỉ lệ nhãn phản hồi sai.
    Trả về (thời điểm cảnh báo, số lần hỏi).
    """
    rows, keys, vectors = [], [], []
    for b, positions in triggers.items():
        q, t = s.seqs[b], s.times[b]
        for i in positions:
            last = min(int(np.searchsorted(t, t[i] + wait, side="right")) - 1, i + k)
            at = float(t[last]) if last == i + k else float(t[i] + wait)
            if at > s.stream_end:
                continue
            rows.append((at, b, len(keys)))
            keys.append(tuple(sorted(Counter(q[:last + 1]).items())))
            if kind == "count":
                v = np.zeros(NUM_EVENTS)
                for e, n in keys[-1]:
                    v[e] = np.log1p(n)
                vectors.append(v)
            elif kind == "lstm":
                j0, n_win = s.first_row[b], last - s.window + 1
                vectors.append(hidden[s.pair[j0:j0 + n_win]].mean(0) if j0 >= 0 and n_win > 0 else np.zeros(hidden.shape[1]))
    vec = np.array(vectors) if vectors else np.zeros((1, 1))
    if kind == "lstm" and len(keys):
        on_tune = np.array([s.is_tune[b] for _, b, j in sorted(rows, key=lambda r: r[2])])
        vec = vec - (vec[on_tune].mean(0) if on_tune.any() else 0.0)  # tâm lấy ở đoạn tune, không nhìn phần sau
    vec = vec / np.maximum(np.linalg.norm(vec, axis=1, keepdims=True), 1e-9)
    rng, alert, asked = np.random.default_rng(seed), np.full(s.n, np.inf), 0
    memory, slot, bank, votes, pending = defaultdict(lambda: [0, 0]), {}, np.zeros((len(keys) + 1, vec.shape[1])), [], []
    for at, b, j in sorted(rows):
        while pending and pending[0][0] <= at:
            _, jj, label = heapq.heappop(pending)
            if kind == "exact":
                memory[keys[jj]][label] += 1
            else:
                if keys[jj] not in slot:
                    slot[keys[jj]] = len(votes)
                    bank[len(votes)] = vec[jj]
                    votes.append(0)
                votes[slot[keys[jj]]] += 1 if label else -1
        if np.isfinite(alert[b]) or immediate[b] <= at or completion[b] < at:
            continue
        if kind == "exact":
            n0, n1 = memory[keys[j]] if keys[j] in memory else (0, 0)
            suppressed = n0 > n1
        else:
            near = bank[:len(votes)] @ vec[j] >= 1 - radius if votes else np.zeros(0, dtype=bool)
            suppressed = bool(near.any()) and sum(v for v, hit in zip(votes, near) if hit) < 0
        if suppressed:
            continue
        alert[b], asked = at, asked + 1
        label = int(s.y[b]) if rng.random() >= flip else 1 - int(s.y[b])
        heapq.heappush(pending, (max(at, s.t_end[b]) + delay, j, label))
    return alert, asked


def pair_triggers(s: Stream, hot: np.ndarray) -> dict[int, list[int]]:
    out: dict[int, list[int]] = defaultdict(list)
    for n in np.flatnonzero(hot[s.pair]):
        out[int(s.block[n])].append(int(s.pos[n]))
    return out


def run_source(s: Stream, cfg: dict, name: str, models: list, rows: list, chosen: dict, subsets: dict, with_feedback: bool) -> None:
    """Chọn tham số trên tune và chấm mọi bộ phát hiện dùng model này (name = ensemble | seed<k>)."""
    scfg, y, tune = cfg["streaming"], s.y, s.is_tune
    margins = cfg["hybrid"]["deadline_margins"]
    comp = {m: s.completion(m) for m in margins}
    c_grid = [int(f * len(s.train)) for f in scfg["rare_fractions"]]
    rare = {c: s.rare_times(c) for c in dict.fromkeys(c_grid)}
    p, rank, hidden = s.lstm(models)
    seed = cfg["seed"]

    def add(variant, alert, asked=None, delay=None, source=name):
        for sub, (mask, hours) in subsets.items():
            rows.append({"source": source, "variant": variant, "label_delay_sec": delay, "subset": sub,
                         **ops_metrics(y[mask], alert[mask], s.t0[mask], hours), "labels_asked": asked})

    def pick(candidates: dict):
        """Khoá F1 tune cao nhất; hoà thì khoá xuất hiện trước."""
        return select_on_tune({k: np.isfinite(a).astype(int) for k, a in candidates.items()}, y, tune)[0]

    def shortest(best_m, preds):
        return shortest_equivalent(preds, best_m, y, tune, seed)

    # 1. hai luật cũ, 2. luật hiếm (c, m chọn trên tune)
    best_m = pick({m: np.minimum(rare[0], comp[m]) for m in margins})
    m_old = shortest(best_m, {m: np.isfinite(np.minimum(rare[0], comp[m])) for m in margins})
    add("hai luật cũ (c = 0)", np.minimum(rare[0], comp[m_old]), source="rules")
    c_best, m_best = pick({(c, m): np.minimum(rare[c], comp[m]) for c, m in itertools.product(rare, margins)})
    m_best = shortest(m_best, {m: np.isfinite(np.minimum(rare[c_best], comp[m])) for m in margins})
    add("luật hiếm + hạn chờ", np.minimum(rare[c_best], comp[m_best]), source="rules")

    # 3. DeepLog xếp hạng + hai luật cũ (cách của bước 12)
    g_grid = range(1, cfg["hybrid"]["max_top_g"] + 1)
    cross_rank = {g: s.first_crossing(rank > g) for g in g_grid}
    m_h, g_h = pick({(m, g): np.minimum.reduce([rare[0], comp[m], cross_rank[g]]) for m, g in itertools.product(margins, g_grid)})
    m_h = shortest(m_h, {m: np.isfinite(np.minimum.reduce([rare[0], comp[m], cross_rank[g_h]])) for m in margins})
    add("DeepLog xếp hạng + hai luật cũ", np.minimum.reduce([rare[0], comp[m_h], cross_rank[g_h]]))

    # 4. DeepLog độ bất ngờ trộn tần suất + hạn chờ (thay cả luật sự kiện lạ; vẫn có sự kiện chưa từng thấy ở 10 sự kiện đầu)
    def mixed(prob):
        best = None
        for eps in scfg["mix_eps"]:
            sur = s.surprisal(prob, eps)
            for th in scfg["surprisal_grid"]:
                cross = s.first_crossing(sur > th)
                for m in margins:
                    f1 = 2 * np.isfinite(np.minimum.reduce([rare[0], comp[m], cross]))[tune & (y == 1)].sum()
                    pred = np.isfinite(np.minimum.reduce([rare[0], comp[m], cross]))
                    f1 = f1 / max(pred[tune].sum() + (y[tune] == 1).sum(), 1)
                    if best is None or f1 > best[0]:
                        best = (f1, eps, th, m, cross)
        _, eps, th, m, cross = best
        m = shortest(m, {mm: np.isfinite(np.minimum.reduce([rare[0], comp[mm], cross])) for mm in margins})
        return eps, th, m, cross

    eps, th, m_d, cross = mixed(p)
    add("DeepLog độ bất ngờ trộn tần suất + hai luật cũ", np.minimum.reduce([rare[0], comp[m_d], cross]))
    add("chỉ DeepLog độ bất ngờ trộn tần suất", cross)
    g_only = pick({g: cross_rank[g] for g in g_grid})
    add("chỉ DeepLog xếp hạng", cross_rank[g_only])
    both = np.minimum.reduce([rare[c_best], comp[m_best], cross])
    add("luật hiếm + hạn chờ + DeepLog độ bất ngờ", both)
    chosen[name] = {"rules": {"m_old": m_old, "c": c_best, "m": m_best}, "rank_hybrid": {"g": g_h, "m": m_h},
                    "mixed": {"eps": eps, "threshold": th, "m": m_d}}

    if not with_feedback:
        return
    # 5. đối chứng Markov cùng phép trộn và chọn trên tune
    best = None
    for order in scfg["markov_orders"]:
        pm = s.markov(order)
        e_, t_, m_, cr = mixed(pm)
        f1 = compute_f1(np.minimum.reduce([rare[0], comp[m_], cr]), y, tune)
        if best is None or f1 > best[0]:
            best = (f1, order, e_, t_, m_, cr)
    add("Markov trộn tần suất + hai luật cũ", np.minimum.reduce([rare[0], comp[best[4]], best[5]]), source="markov")
    chosen["markov"] = {"order": best[1], "eps": best[2], "threshold": best[3], "m": best[4]}

    # 6. phản hồi (tham khảo): k = 20, H = 3600 cố định từ bước 14
    k, wait = scfg["feedback_k"], scfg["feedback_wait"]
    trig = {
        "luật hiếm": ({b: [i for i, e in enumerate(q) if 0 < s.session_freq.get(e, 0) <= c_best]
                       for b, q in enumerate(s.seqs) if np.isfinite(rare[c_best][b])}, m_best),
        "DeepLog trộn": (pair_triggers(s, s.surprisal(p, eps) > th), m_d),
        "DeepLog xếp hạng": (pair_triggers(s, rank > g_h), m_h),
    }
    for tname, (tr, mm) in trig.items():
        tr = {b: v for b, v in tr.items() if v}
        for kind, radius in (("exact", 0.0), ("count", scfg["feedback_radius_count"]), ("lstm", scfg["feedback_radius_lstm"])):
            for delay, flip in [(d, 0.0) for d in scfg["label_delays_sec"]] + [(scfg["label_delays_sec"][0], f) for f in scfg["label_flips"]]:
                alert, asked = feedback(s, tr, rare[0], comp[mm], hidden, kind, radius, k, wait, delay, flip, seed)
                add(f"{tname} + phản hồi {kind}" + (f", nhãn sai {flip:.0%}" if flip else ""),
                    np.minimum.reduce([rare[0], comp[mm], alert]), asked, delay)


def compute_f1(alert: np.ndarray, y: np.ndarray, mask: np.ndarray) -> float:
    pred = np.isfinite(alert[mask])
    return 2 * (pred & (y[mask] == 1)).sum() / max(pred.sum() + (y[mask] == 1).sum(), 1)


def verify_demo(s: Stream, cfg: dict, chosen: dict) -> int:
    """Phát lại toàn luồng bằng bộ phát hiện (cách chấm mixed) và so từng block với bản offline; trả về số block lệch."""
    from src.stream_demo import build_detector, by_second, replay_order  # tránh import vòng
    detector, _ = build_detector(cfg, "mixed")
    p, _, _ = s.lstm(load_seed_models(cfg, torch.device("cpu"))[0])
    reference = np.minimum.reduce([s.rare_times(0), s.completion(chosen["m"]),
                                   s.first_crossing(s.surprisal(p, chosen["eps"]) > chosen["threshold"])])
    got = np.full(s.n, np.inf)
    stream = replay_order(pd.Series(s.times), pd.Series(s.seqs))
    for second, batch in by_second(*stream):
        for a in detector.process(second, batch):
            got[a.block] = min(got[a.block], a.time)
    for a in detector.finish(float(stream[2][-1])):
        got[a.block] = min(got[a.block], a.time)
    return int((got != reference).sum())


def main() -> None:
    parser = argparse.ArgumentParser(description="Bước 15: so sánh cuối các bộ phát hiện theo luồng")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--base-config", default="configs/config.yaml", help="lấy mục streaming nếu --config không có")
    args = parser.parse_args()
    cfg = load_config(args.config)
    cfg.setdefault("streaming", load_config(args.base_config)["streaming"])
    s = Stream(cfg)
    hold = s.t0 > cfg["streaming"]["holdout_hour"] * 3600
    span = (s.t_end.max() - s.t0.min()) / 3600
    subsets = {"eval": (~s.is_tune, span * (1 - cfg["select"]["tune_ratio"])),
               "develop": (~s.is_tune & ~hold, max(cfg["streaming"]["holdout_hour"] - s.t0[~s.is_tune].min() / 3600, 1e-9)),
               "holdout": (hold, (s.stream_end / 3600 - cfg["streaming"]["holdout_hour"]))}
    subsets = {k: v for k, v in subsets.items() if v[0].any()}  # log ngắn hơn holdout_hour: không có holdout
    logger.info("%s", {k: (int(m.sum()), int(s.y[m].sum())) for k, (m, _) in subsets.items()})
    device = torch.device("cpu")
    models, _ = load_seed_models(cfg, device)
    rows, chosen = [], {}
    run_source(s, cfg, "ensemble", models, rows, chosen, subsets, with_feedback=True)
    for seed, model in zip(cfg["robustness"]["seeds"], models):
        run_source(s, cfg, f"seed{seed}", [model], rows, chosen, subsets, with_feedback=False)
    out = ensure_dir(cfg["paths"]["results_dir"] / "stream")  # mỗi config một thư mục, không ghi đè nhau
    table = pd.DataFrame(rows).drop_duplicates(["source", "variant", "label_delay_sec", "subset"])
    table.to_csv(out / "compare.csv", index=False)
    report = {"meta": run_metadata(cfg), "chosen": chosen}
    save_json(report, out / "compare.json")
    if (out / "offline.json").exists():  # demo cần kết quả bước 12 của cùng config
        report["demo_different_blocks"] = verify_demo(s, cfg, chosen["ensemble"]["mixed"])
        save_json(report, out / "compare.json")
        logger.info("Demo (độ bất ngờ) khác offline ở %d / %d block", report["demo_different_blocks"], s.n)
    view = table[(table["subset"] == "holdout") & table["source"].isin(["rules", "ensemble", "markov"])]
    logger.info("\n%s", view[["source", "variant", "label_delay_sec", "precision", "recall", "f1", "fp_per_hour"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
