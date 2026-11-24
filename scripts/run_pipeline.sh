#!/usr/bin/env bash
# Chạy pipeline theo đúng thứ tự (15 bước + phân tích lỗi).
# Cách dùng (từ thư mục ~/project-3, đã activate venv):
#   bash scripts/run_pipeline.sh                      # dùng configs/config.yaml
#   bash scripts/run_pipeline.sh configs/khac.yaml    # dùng file config khác
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="${1:-configs/config.yaml}"

echo "=== [1/15] Parse log bằng Drain3 ===";             python -m src.parse_logs      --config "$CONFIG"
echo "=== [2/15] Chuỗi sự kiện theo block ===";          python -m src.build_sequences --config "$CONFIG"
echo "=== [3/15] Chia train / val / test ===";           python -m src.split_data      --config "$CONFIG"
echo "=== [4/15] Train DeepLog ===";                     python -m src.train           --config "$CONFIG"
echo "=== [5/15] Phát hiện bất thường (top-g) ===";      python -m src.detect          --config "$CONFIG"
echo "=== [6/15] Baseline PCA + Isolation Forest ===";   python -m src.baselines       --config "$CONFIG"
echo "=== [7/15] Chọn phương án + ngưỡng trên tập tune, báo cáo trên tập eval ==="
python -m src.deeplog_scores --config "$CONFIG"
for mode in earliest random; do python -m src.model_selection --config "$CONFIG" --tune-mode "$mode"; done
echo "=== Phân tích lỗi DeepLog (g = 7, giá trị chọn ở bước 7) ==="
python -m src.detect   --config "$CONFIG" --top-g 7
python -m src.evaluate --config "$CONFIG" --top-g 7
echo "=== [8/15] Luật bất biến + DeepLog, chỉ số vận hành (cảnh báo giả/giờ, độ trễ, drift) ==="
python -m src.block_context --config "$CONFIG"
python -m src.hybrid_detect --config "$CONFIG"
python -m src.plot_results  --config "$CONFIG"
echo "=== [9/15] Độ ổn định theo seed (~4 phút mỗi seed) ==="
python -m src.seed_study    --config "$CONFIG"
echo "=== [10/15] Ghép 5 model DeepLog của bước 9 ==="
python -m src.ensemble      --config "$CONFIG"
echo "=== [11/15] Điểm vận hành theo ngân sách cảnh báo giả (model chính, từng seed, ensemble) ==="
RESULTS_DIR=$(python -c "from src.config import load_config; print(load_config('$CONFIG')['paths']['results_dir'])")
for mode in earliest random; do
  python -m src.operating_point --config "$CONFIG" --tune-mode "$mode"
  for dir in "$RESULTS_DIR"/seeds/seed*/"$mode" "$RESULTS_DIR"/ensemble/"$mode"; do
    python -m src.operating_point --config "$CONFIG" --tune-mode "$mode" --ranks-dir "$dir"
  done
done
python -m src.operating_point --config "$CONFIG" --collect
echo "=== [12/15] Kết quả offline theo quy tắc chạy được theo luồng ==="
python -m src.stream_offline --config "$CONFIG"
echo "=== [13/15] Phát lại theo luồng, kiểm tra trùng kết quả bước 12 ==="
python -m src.stream_demo --config "$CONFIG" --verify
echo "=== [14/15] Chờ xác nhận + phản hồi của người vận hành, đối chứng n-gram ==="
python -m src.stream_feedback --config "$CONFIG"
echo "=== [15/15] So sánh cuối các bộ phát hiện theo luồng ==="
python -m src.stream_compare --config "$CONFIG"
