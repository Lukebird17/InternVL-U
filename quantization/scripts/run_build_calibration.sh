#!/bin/bash

# 构建校准数据集 (Flickr8K → 多类型 VQA)
#
# 用法:
#   bash scripts/run_build_calibration.sh                  # 默认 1000 条
#   NUM_SAMPLES=500 bash scripts/run_build_calibration.sh  # 自定义条数

FLICKR8K_ROOT="/data/14thdd/users/yongsencheng/Bagel/data/flickr8k"
QUANT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUTPUT_DIR="${QUANT_ROOT}/quantization_outputs/calibration_data"

NUM_SAMPLES="${NUM_SAMPLES:-1000}"
SEED="${SEED:-42}"
RUN_DATE="${RUN_DATE:-$(date +%Y%m%d)}"

echo "=================================================="
echo "Build Calibration Dataset (Flickr8K → Multi-Type VQA)"
echo "=================================================="
echo "Source:    $FLICKR8K_ROOT"
echo "Output:   $OUTPUT_DIR"
echo "Samples:  $NUM_SAMPLES"
echo "Run date: $RUN_DATE"
echo ""

if [ ! -d "$FLICKR8K_ROOT" ]; then
    echo "Error: Flickr8K not found: $FLICKR8K_ROOT"
    exit 1
fi

CALIB_FILE="${OUTPUT_DIR}/calibration_dataset_${NUM_SAMPLES}samples_${RUN_DATE}.json"

if [ -f "$CALIB_FILE" ]; then
    echo "Already exists: $CALIB_FILE"
    echo "Delete it first if you want to rebuild."
    exit 0
fi

python -u "${QUANT_ROOT}/utils/build_calibration_dataset.py" \
    --flickr8k_root "$FLICKR8K_ROOT" \
    --output_dir "$OUTPUT_DIR" \
    --num_samples "$NUM_SAMPLES" \
    --seed "$SEED" \
    --run_date "$RUN_DATE"

echo ""
echo "Done! Output: $CALIB_FILE"
