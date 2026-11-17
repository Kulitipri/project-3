#!/usr/bin/env bash
# Chạy pipeline theo đúng thứ tự (10 bước + phân tích lỗi).
# Cách dùng (từ thư mục ~/project-3, đã activate venv):
#   bash scripts/run_pipeline.sh                      # dùng configs/config.yaml
#   bash scripts/run_pipeline.sh configs/khac.yaml    # dùng file config khác
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="${1:-configs/config.yaml}"

echo "=== [1/10] Parse log bằng Drain3 ===";             python -m src.parse_logs      --config "$CONFIG"
echo "=== [2/10] Chuỗi sự kiện theo block ===";          python -m src.build_sequences --config "$CONFIG"
echo "=== [3/10] Chia train / val / test ===";           python -m src.split_data      --config "$CONFIG"
echo "=== [4/10] Train DeepLog ===";                     python -m src.train           --config "$CONFIG"
echo "=== [5/10] Phát hiện bất thường (top-g) ===";      python -m src.detect          --config "$CONFIG"
echo "=== [6/10] Baseline PCA + Isolation Forest ===";   python -m src.baselines       --config "$CONFIG"
echo "=== [7/10] Chọn phương án + ngưỡng trên tập tune, báo cáo trên tập eval ==="
python -m src.deeplog_scores --config "$CONFIG"
for mode in earliest random; do python -m src.model_selection --config "$CONFIG" --tune-mode "$mode"; done
echo "=== Phân tích lỗi DeepLog (g = 7, giá trị chọn ở bước 7) ==="
python -m src.detect   --config "$CONFIG" --top-g 7
python -m src.evaluate --config "$CONFIG" --top-g 7
echo "=== [8/10] Luật bất biến + DeepLog, chỉ số vận hành (cảnh báo giả/giờ, độ trễ, drift) ==="
python -m src.block_context --config "$CONFIG"
python -m src.hybrid_detect --config "$CONFIG"
python -m src.plot_results  --config "$CONFIG"
echo "=== [9/10] Độ ổn định theo seed (~4 phút mỗi seed) ==="
python -m src.seed_study    --config "$CONFIG"
echo "=== [10/10] Ghép 5 model DeepLog của bước 9 ==="
python -m src.ensemble      --config "$CONFIG"
