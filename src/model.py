"""
Model DeepLog: dự đoán sự kiện log kế tiếp từ w sự kiện trước đó.

Kiến trúc: Embedding → LSTM nhiều lớp → Linear (num_keys lớp đầu ra)
Tham khảo: Du et al., "DeepLog: Anomaly Detection and Diagnosis from System Logs
through Deep Learning", ACM CCS 2017.
"""
import torch
from torch import nn


class DeepLog(nn.Module):
    def __init__(self, num_keys: int, embedding_dim: int, hidden_size: int, num_layers: int):
        """
        Input:
          num_keys      — số lớp đầu ra = len(event_index) + 1 (cộng 1 cho PAD = 0)
          embedding_dim — số chiều vector biểu diễn mỗi sự kiện
          hidden_size   — số chiều trạng thái ẩn của LSTM
          num_layers    — số lớp LSTM xếp chồng
        Việc cần làm:
          - Khai báo 3 lớp: embedding → LSTM → linear.
          - Embedding nên đặt padding_idx=0 để vector của PAD không được học.
          - LSTM nên đặt batch_first=True để input có dạng (batch, window, feature).
        """
        super().__init__()
        self.embedding = nn.Embedding(num_keys, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_keys)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input : x — LongTensor (batch, window_size), mỗi phần tử là chỉ số sự kiện
        Output: logits — FloatTensor (batch, num_keys), CHƯA qua softmax
                (nn.CrossEntropyLoss đã tự tính softmax bên trong)
        Chỉ lấy output của bước thời gian CUỐI CÙNG của LSTM để đưa vào lớp linear.
        """
        out, _ = self.lstm(self.embedding(x))  # (batch, window, hidden)
        return self.fc(out[:, -1, :])
