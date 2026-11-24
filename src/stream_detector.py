"""
Bộ phát hiện theo luồng: nhận sự kiện theo thứ tự thời gian, phát cảnh báo ngay, KHÔNG nhìn sự kiện tương lai.

Mỗi block giữ window_size sự kiện gần nhất và số lần E5 đã thấy. Ba thành phần (khớp hybrid_stream của bước 12):
  - unseen     : sự kiện chưa từng có trong session Normal lúc train → báo ngay.
  - completion : quá deadline giây kể từ sự kiện đầu mà chưa đủ replica_count lần E5 → bộ hẹn giờ báo.
  - deeplog    : đã đủ window_size sự kiện đứng trước và sự kiện thật xếp hạng > top_g → báo ngay.

Chế độ phản hồi (feedback = True): khi DeepLog trượt, chưa báo ngay mà chờ thêm
confirm_events sự kiện của block hoặc confirm_seconds giây, rồi mới quyết định theo MẪU = số đếm sự kiện của block tới lúc
đó. Mẫu đã được người vận hành xác nhận phần lớn là bình thường thì không báo nữa. Lý do: tại lúc trượt, nhiều bất thường
có đúng cùng 10 sự kiện với phiên bình thường; thêm vài sự kiện mới tách được. Phản hồi chỉ có cho block đã bị báo.

Xem stream_demo.py để chạy; tests/test_stream.py để kiểm tra từng trường hợp biên.
"""
import heapq
from collections import OrderedDict, deque
from dataclasses import dataclass

import numpy as np
import torch

UNSEEN, COMPLETION, DEEPLOG = "unseen", "completion", "deeplog"
UNKNOWN_EVENT = -1  # dòng log không khớp template nào: luôn là sự kiện lạ


@dataclass
class Alert:
    block: object
    time: float        # giây, tính từ dòng log đầu tiên
    component: str     # unseen | completion | deeplog
    first: bool        # True nếu là cảnh báo đầu tiên của block (cảnh báo mà người vận hành thấy)
    event: int | None = None          # sự kiện gây ra cảnh báo (None với completion)
    rank: int | None = None           # thứ hạng của sự kiện trong dự đoán (deeplog)
    score: float | None = None        # điểm đem so với ngưỡng: hạng (RankScorer) hoặc độ bất ngờ (SurprisalScorer)
    expected: tuple[int, ...] = ()    # 3 sự kiện model chờ đợi nhất (deeplog)
    replicas: int | None = None       # số lần E5 đã thấy (completion)
    pattern: tuple | None = None      # mẫu dùng để nhận phản hồi (deeplog, chế độ chờ xác nhận)


class RankScorer:
    """Thứ hạng của mọi sự kiện theo model, cho từng cửa sổ; nhớ kết quả của các cửa sổ gần đây (LRU)."""

    def __init__(self, model: torch.nn.Module, cache_size: int, device: torch.device | str = "cpu"):
        self.model, self.device, self.cache_size = model, device, cache_size
        self.cache: OrderedDict[tuple, np.ndarray] = OrderedDict()
        self.requests = self.misses = 0

    @torch.no_grad()
    def _score(self, windows: list[tuple]) -> np.ndarray:
        logits = self.model(torch.tensor(windows, dtype=torch.long, device=self.device))
        # rank(k) = số sự kiện có điểm CAO HƠN HẲN sự kiện k, cộng 1 — cùng định nghĩa với deeplog_scores.py
        return ((logits.unsqueeze(1) > logits.unsqueeze(2)).sum(dim=2) + 1).cpu().numpy().astype(np.int16)

    def rank_tables(self, windows: list[tuple]) -> list[np.ndarray]:
        """Mỗi cửa sổ → mảng rank theo chỉ số sự kiện."""
        self.requests += len(windows)
        missing = [w for w in dict.fromkeys(windows) if w not in self.cache]
        self.misses += len(missing)
        fresh = dict(zip(missing, self._score(missing))) if missing else {}
        out = []
        for w in windows:
            table = fresh.get(w)
            if table is None:
                table = self.cache[w]
                self.cache.move_to_end(w)
            out.append(table)
        self.cache.update(fresh)
        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return out


class SurprisalScorer(RankScorer):
    """
    Như RankScorer nhưng điểm là ĐỘ BẤT NGỜ −log p, p = (1 − eps)·p_model + eps·tần_suất_chung của sự kiện (bước 15).

    Khác hạng ở chỗ giữ độ lớn xác suất: sự kiện hiếm vẫn bị báo dù lọt top-g; phần trộn chống mạng quá tự tin ở ngữ cảnh mới.
    """

    def __init__(self, model, cache_size: int, eps: float, unigram: np.ndarray, device: torch.device | str = "cpu"):
        super().__init__(model, cache_size, device)
        self.eps, self.unigram = eps, unigram

    @torch.no_grad()
    def _score(self, windows: list[tuple]) -> np.ndarray:
        x = torch.tensor(windows, dtype=torch.long, device=self.device)
        p = torch.exp(self.model(x)).double().cpu().numpy()
        return -np.log(np.maximum((1 - self.eps) * p + self.eps * self.unigram[:p.shape[1]], 1e-30))


class StreamDetector:
    def __init__(self, scorer: RankScorer, seen_events: set[int], replica_event: int, replica_count: int,
                 deadline_sec: float, top_g: float, window_size: int, feedback: bool = False, confirm_events: int = 0,
                 confirm_seconds: float = 0.0):
        self.scorer, self.seen, self.replica_event = scorer, seen_events, replica_event
        self.replica_count, self.deadline, self.top_g, self.window = replica_count, deadline_sec, top_g, window_size
        self.history: dict[object, deque] = {}
        self.replicas: dict[object, int] = {}       # chỉ giữ block chưa đủ replica_count lần E5
        self.timers: list[tuple[float, int, int, object]] = []  # (hạn, 0 = chờ xác nhận / 1 = nhân bản, số thứ tự, block)
        self.fired: dict[object, set[str]] = {}     # thành phần nào đã báo block nào (mỗi thành phần báo 1 lần)
        self.events = 0
        self.use_feedback, self.confirm_events, self.confirm_seconds = feedback, confirm_events, confirm_seconds
        self.counts: dict[object, dict[int, int]] = {}                 # số đếm từng loại sự kiện của block
        self.waiting: dict[object, list[list]] = {}                    # trượt đang chờ: [số sự kiện còn thiếu, id, chi tiết]
        self.memory: dict[tuple, list[int]] = {}                       # mẫu → [số lần xác nhận bình thường, bất thường]
        self.miss_id = 0

    def feedback(self, pattern: tuple, label: int) -> None:
        """Người vận hành xác nhận một cảnh báo đã phát: label 0 = báo nhầm, 1 = đúng."""
        self.memory.setdefault(pattern, [0, 0])[label] += 1

    def _pattern(self, block) -> tuple:
        return tuple(sorted(self.counts[block].items()))

    def _decide(self, out: list[Alert], block, time: float, details: dict) -> None:
        if block in self.fired:     # luật đã báo block này
            return
        pattern = self._pattern(block)
        normal, anomalous = self.memory.get(pattern, (0, 0))
        if normal <= anomalous:
            self._alert(out, block, time, DEEPLOG, pattern=pattern, **details)

    def _alert(self, out: list[Alert], block, time: float, component: str, **details) -> None:
        fired = self.fired.setdefault(block, set())
        if component not in fired:
            out.append(Alert(block, time, component, first=not fired, **details))
            fired.add(component)

    def _run_timers(self, out: list[Alert], now: float, inclusive: bool) -> None:
        """Báo các block đã quá hạn. Sự kiện đến ĐÚNG lúc hết hạn vẫn kịp, nên khi log còn chảy chỉ xét due < now."""
        while self.timers and (self.timers[0][0] < now or (inclusive and self.timers[0][0] == now)):
            due, kind, number, block = heapq.heappop(self.timers)
            if kind == 0:           # hết thời gian chờ xác nhận của một lần trượt
                for miss in [w for w in self.waiting.get(block, []) if w[1] == number]:
                    self.waiting[block].remove(miss)
                    self._decide(out, block, due, miss[2])
            elif block in self.replicas:
                self._alert(out, block, due, COMPLETION, replicas=self.replicas.pop(block))

    def process(self, time: float, batch: list[tuple[object, int]]) -> list[Alert]:
        """
        Xử lý các sự kiện (block, event) có CÙNG thời điểm, theo thứ tự xuất hiện trong log.

        Log HDFS có độ phân giải 1 giây nên gom theo giây không làm đổi thời điểm cảnh báo mà cho phép model chấm
        nhiều cửa sổ một lượt.
        """
        out: list[Alert] = []
        self._run_timers(out, time, inclusive=False)
        # lượt 1: cập nhật cửa sổ, luật sự kiện lạ, gom các cửa sổ cần chấm (chưa cần biết kết quả model)
        windows: list[tuple] = []
        scored: list[int] = []
        for order, (block, event) in enumerate(batch):
            self.events += 1
            past = self.history.get(block)
            if past is None:
                past = self.history[block] = deque(maxlen=self.window)
                self.replicas[block] = 0
                heapq.heappush(self.timers, (time + self.deadline, 1, self.events, block))
            if event not in self.seen:
                self._alert(out, block, time, UNSEEN, event=event)
            if event == self.replica_event and block in self.replicas:
                self.replicas[block] += 1
                if self.replicas[block] >= self.replica_count:
                    del self.replicas[block]
            if len(past) == self.window:
                windows.append(tuple(past))
                scored.append(order)
            # sự kiện không có trong từ vựng: model không biết nó, đưa vào cửa sổ dưới dạng PAD (0)
            past.append(event if event != UNKNOWN_EVENT else 0)
        tables = dict(zip(scored, self.scorer.rank_tables(windows))) if windows else {}

        # lượt 2: theo thứ tự trong log — DeepLog, và các lần trượt đang chờ xác nhận
        for order, (block, event) in enumerate(batch):
            if self.use_feedback:
                counts = self.counts.setdefault(block, {})
                counts[event] = counts.get(event, 0) + 1
                for miss in list(self.waiting.get(block, ())):
                    miss[0] -= 1
                    if miss[0] == 0:
                        self.waiting[block].remove(miss)
                        self._decide(out, block, time, miss[2])
            table = tables.get(order)
            if table is None:
                continue
            score = float("inf") if event == UNKNOWN_EVENT else float(table[event])
            if score <= self.top_g:
                continue
            # cả hạng lẫn độ bất ngờ: càng nhỏ càng được chờ đợi → hạng = số sự kiện được chờ đợi hơn + 1
            rank = len(table) if event == UNKNOWN_EVENT else int((table < score).sum()) + 1
            details = {"event": event, "rank": rank, "score": score,
                       "expected": tuple(int(i) for i in np.argsort(table, kind="stable")[:3])}
            if not self.use_feedback:
                self._alert(out, block, time, DEEPLOG, **details)
            elif self.confirm_events == 0:
                self._decide(out, block, time, details)
            else:
                self.miss_id += 1
                self.waiting.setdefault(block, []).append([self.confirm_events, self.miss_id, details])
                heapq.heappush(self.timers, (time + self.confirm_seconds, 0, self.miss_id, block))
        return out

    def advance(self, time: float) -> list[Alert]:
        """Chạy các hẹn giờ rơi vào TRƯỚC `time` (dùng khi có phản hồi đến giữa hai giây của log)."""
        out: list[Alert] = []
        self._run_timers(out, time, inclusive=False)
        return out

    def finish(self, time: float) -> list[Alert]:
        """Hết log: chỉ còn các hẹn giờ rơi vào trước hoặc đúng thời điểm cuối."""
        out: list[Alert] = []
        self._run_timers(out, time, inclusive=True)
        return out
