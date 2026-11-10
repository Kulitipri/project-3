#!/usr/bin/env bash
# Chạy pipeline theo đúng thứ tự (tuần 3: bước 1–7 + phân tích lỗi).
# Cách dùng (từ thư mục ~/project-3, đã activate venv):
#   bash scripts/run_pipeline.sh                      # dùng configs/config.yaml
#   bash scripts/run_pipeline.sh configs/khac.yaml    # dùng file config khác
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="${1:-configs/config.yaml}"

echo "=== [1/7] Parse log bằng Drain3 ===";             python -m src.parse_logs      --config "$CONFIG"
echo "=== [2/7] Chuỗi sự kiện theo block ===";          python -m src.build_sequences --config "$CONFIG"
echo "=== [3/7] Chia train / val / test ===";           python -m src.split_data      --config "$CONFIG"
echo "=== [4/7] Train DeepLog ===";                     python -m src.train           --config "$CONFIG"
echo "=== [5/7] Phát hiện bất thường (top-g) ===";      python -m src.detect          --config "$CONFIG"
echo "=== [6/7] Baseline PCA + Isolation Forest ===";   python -m src.baselines       --config "$CONFIG"
echo "=== [7/7] Chọn phương án + ngưỡng trên tập tune, báo cáo trên tập eval ==="
python -m src.deeplog_scores --config "$CONFIG"
for mode in earliest random; do python -m src.model_selection --config "$CONFIG" --tune-mode "$mode"; done
echo "=== Phân tích lỗi DeepLog (g = 7, giá trị chọn ở bước 7) ==="
python -m src.detect   --config "$CONFIG" --top-g 7
python -m src.evaluate --config "$CONFIG" --top-g 7
