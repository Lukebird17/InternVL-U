#!/bin/bash

# Stage 0 (InternVL-U): 使用外部校准数据集收集激活值
#
# 与原始 run_stage0_collect_activations.sh 的区别：
#   原版用内置的 8 UND + 8 GEN 小校准集
#   本脚本用 build_calibration_dataset.py 构建的大校准集（如 1000 条 Flickr8K VQA）
#
# 输出: stage0_full_activation_Xsamples_latest.pt
#   该文件供 Stage 20 搜索时的 GPTQ/AWQ/SmoothQuant 算法使用
#
# 用法:
#   bash scripts/run_stage0_with_calibration.sh
#   CALIB_DATASET=/path/to/dataset.json GPU_IDS=0,1 bash scripts/run_stage0_with_calibration.sh

# ===================== 配置 =====================

MODEL_PATH="/data/14thdd/users/honglianglu/InternVL-U/model"

QUANT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUTPUT_DIR="${QUANT_ROOT}/quantization_outputs/stage0_full_activation"

GPU_IDS="${GPU_IDS:-6}"
MAX_MEM_PER_GPU="40GiB"

# 不再需要 MAX_SAMPLES：所有 token 的 Hessian/channel stats 在线积累，不存原始 activation
SEED="${SEED:-42}"

# 校准数据集路径（默认用 latest 软链接）
CALIB_DATA_DIR="${QUANT_ROOT}/quantization_outputs/calibration_data"
CALIB_DATASET="${CALIB_DATASET:-${CALIB_DATA_DIR}/calibration_dataset_latest.json}"

# ===================== 运行 =====================

echo "=================================================="
echo "Stage 0 (InternVL-U): Activation Collection"
echo "  Using EXTERNAL calibration dataset"
echo "=================================================="
echo "Model:       $MODEL_PATH"
echo "Output:      $OUTPUT_DIR"
echo "GPU IDs:     $GPU_IDS"
echo "Calib data:  $CALIB_DATASET"
echo "Mode: Online Hessian (GPTQ) + Channel Stats (AWQ/SmoothQuant)"
echo ""

if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model path not found: $MODEL_PATH"
    exit 1
fi

if [ ! -f "$CALIB_DATASET" ]; then
    echo "Error: Calibration dataset not found: $CALIB_DATASET"
    echo "  Run: bash scripts/run_build_calibration.sh first"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

CMD="python -u ${QUANT_ROOT}/stages/stage0_collect_activations.py \
    --model_path $MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    --gpu_ids $GPU_IDS \
    --max_mem_per_gpu $MAX_MEM_PER_GPU \
    --calibration_dataset $CALIB_DATASET \
    --seed $SEED"

echo ""
eval $CMD 2>&1 | tee "$OUTPUT_DIR/stage0_log_$(date +%Y%m%d_%H%M%S).txt"

echo ""
echo "Stage 0 completed!"
echo "  GPTQ Hessian:  $OUTPUT_DIR/gptq_hessian/"
echo "  SmoothQuant:   $OUTPUT_DIR/smoothquant_stats_latest.pt"
echo "  AWQ:           $OUTPUT_DIR/awq_stats_latest.pt"
