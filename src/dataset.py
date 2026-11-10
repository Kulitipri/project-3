"""
Biến session thành các cặp (cửa sổ w sự kiện trước, sự kiện kế tiếp) để train DeepLog.

Ví dụ với window_size = 3, session [5, 2, 2, 7, 1]:
    X = [5, 2, 2] → y = 7
    X = [2, 2, 7] → y = 1
"""
import torch
from torch.utils.data import Dataset

PAD_INDEX = 0


def make_windows(sequence: list[int], window_size: int) -> tuple[list[list[int]], list[int]]:
    """
    Cắt 1 session thành các cửa sổ trượt.

    Input : sequence    — list chỉ số sự kiện của 1 session, vd [5, 2, 2, 7, 1]
            window_size — độ dài cửa sổ w
    Output: (windows, targets)
            windows — list các list độ dài đúng bằng window_size
            targets — list sự kiện kế tiếp tương ứng; len(windows) == len(targets)
    Edge case:
      - Session có độ dài <= window_size (session HDFS ngắn nhất chỉ vài sự kiện):
        thêm PAD_INDEX vào ĐẦU session cho đủ window_size + 1 phần tử, rồi mới cắt cửa sổ.
        Như vậy mọi session đều sinh ra ít nhất 1 cặp (window, target).
      - Session rỗng → trả về ([], []).
    """
    if not sequence:
        return [], []
    if len(sequence) <= window_size:
        sequence = [PAD_INDEX] * (window_size + 1 - len(sequence)) + list(sequence)
    windows = [list(sequence[i:i + window_size]) for i in range(len(sequence) - window_size)]
    targets = [sequence[i + window_size] for i in range(len(sequence) - window_size)]
    return windows, targets


class DeepLogWindowDataset(Dataset):
    """Dataset PyTorch chứa toàn bộ cặp (window, target) của nhiều session."""

    def __init__(self, sequences: list[list[int]], window_size: int):
        """
        Input : sequences   — list các session (mỗi session là list[int])
                window_size — độ dài cửa sổ w
        Việc cần làm:
          - Gọi make_windows() cho từng session, nối tất cả kết quả lại.
          - Lưu vào self.X: torch.LongTensor kích thước (N, window_size)
                    self.y: torch.LongTensor kích thước (N,)
        """
        all_windows: list[list[int]] = []
        all_targets: list[int] = []
        for sequence in sequences:
            windows, targets = make_windows(sequence, window_size)
            all_windows.extend(windows)
            all_targets.extend(targets)
        self.X = torch.tensor(all_windows, dtype=torch.long).reshape(-1, window_size)
        self.y = torch.tensor(all_targets, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]
