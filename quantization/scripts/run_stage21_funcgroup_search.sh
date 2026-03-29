#!/bin/bash

# Stage 21 (InternVL-U): Large-Calibration Functional-Group CKA Search (W4A4)
#
# = Stage 20 的大校准集 + Stage 18 的 attn/mlp 分组搜索
#
# 前置步骤（需提前完成）:
#   1. bash scripts/run_build_calibration.sh          → 构建校准数据集
#   2. bash scripts/run_stage0_with_calibration.sh    → 收集 GPTQ/AWQ 激活
#
# 用法:
#   bash scripts/run_stage21_funcgroup_search.sh
#   GPU_IDS=4,5 bash scripts/run_stage21_funcgroup_search.sh

# ===================== 配置 =====================

MODEL_PATH="/data/14thdd/users/honglianglu/InternVL-U/model"

QUANT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUTPUT_DIR="${QUANT_ROOT}/quantization_outputs/stage21_funcgroup"

# 校准数据集
CALIB_DATA_DIR="${QUANT_ROOT}/quantization_outputs/calibration_data"
CALIB_DATASET="${CALIB_DATASET:-${CALIB_DATA_DIR}/calibration_dataset_latest.json}"

# Stage 0 输出
STAGE0_DIR="${QUANT_ROOT}/quantization_outputs/stage0_full_activation"
GPTQ_HESSIAN_INDEX="${STAGE0_DIR}/gptq_hessian_index_latest.json"
SMOOTHQUANT_STATS="${STAGE0_DIR}/smoothquant_stats_latest.pt"
AWQ_STATS="${STAGE0_DIR}/awq_stats_latest.pt"

# 检查 Stage 0 文件
HAS_STAGE0=true
if [ ! -f "$GPTQ_HESSIAN_INDEX" ]; then
    echo "WARNING: GPTQ Hessian not found: $GPTQ_HESSIAN_INDEX"
    HAS_STAGE0=false
fi
if [ ! -f "$SMOOTHQUANT_STATS" ]; then
    echo "WARNING: SmoothQuant stats not found: $SMOOTHQUANT_STATS"
fi
if [ ! -f "$AWQ_STATS" ]; then
    echo "WARNING: AWQ stats not found: $AWQ_STATS"
fi
if [ "$HAS_STAGE0" = false ]; then
    echo "  Run: bash scripts/run_stage0_with_calibration.sh first"
fi

GPU_IDS="${GPU_IDS:-0,1}"
MAX_MEM_PER_GPU="40GiB"

# CKA 搜索子采样
# 200 条 × 9 算法 × 2 组 × 28 层 ≈ 100k 次前向
CKA_NUM_SAMPLES="${CKA_NUM_SAMPLES:-200}"

SUBSAMPLE_STEP=5
SEED=42

RUN_DATE="${RUN_DATE:-$(date +%Y%m%d)}"

# ===================== 运行 =====================

echo "=================================================="
echo "Stage 21: Functional-Group CKA Search (W4A4)"
echo "=================================================="
echo "Model:    $MODEL_PATH"
echo "Output:   $OUTPUT_DIR"
echo "GPU IDs:  $GPU_IDS"
echo "Run date: $RUN_DATE"
echo "Calib:    $CALIB_DATASET"
echo "GPTQ Hessian: ${GPTQ_HESSIAN_INDEX}"
echo "SmoothQuant:  ${SMOOTHQUANT_STATS}"
echo "AWQ stats:    ${AWQ_STATS}"
echo "CKA samples:  $CKA_NUM_SAMPLES"
echo "Search:   attn / mlp independent"
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

CMD="python -u ${QUANT_ROOT}/stages/stage21_largecalib_funcgroup_search.py \
    --model_path $MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    --calibration_dataset $CALIB_DATASET \
    --gpu_ids $GPU_IDS \
    --max_mem_per_gpu $MAX_MEM_PER_GPU \
    --cka_num_samples $CKA_NUM_SAMPLES \
    --subsample_step $SUBSAMPLE_STEP \
    --seed $SEED \
    --run_date $RUN_DATE"

if [ -f "$GPTQ_HESSIAN_INDEX" ]; then
    CMD="$CMD --gptq_hessian_index $GPTQ_HESSIAN_INDEX"
fi
if [ -f "$SMOOTHQUANT_STATS" ]; then
    CMD="$CMD --smoothquant_stats $SMOOTHQUANT_STATS"
fi
if [ -f "$AWQ_STATS" ]; then
    CMD="$CMD --awq_stats $AWQ_STATS"
fi

if [ -n "${TARGET_LAYERS:-}" ]; then
    CMD="$CMD --target_layers $TARGET_LAYERS"
fi

if [ -n "${MAX_CALIB_SAMPLES:-}" ]; then
    CMD="$CMD --max_calib_samples $MAX_CALIB_SAMPLES"
fi

echo ""
LOG_FILE="$OUTPUT_DIR/stage21_log_${RUN_DATE}.txt"
eval $CMD 2>&1 | tee "$LOG_FILE"

echo ""
echo "Stage 21 completed! (run_date=${RUN_DATE})"
echo "  Results: $OUTPUT_DIR"
echo "  Config:  ${QUANT_ROOT}/quantization_outputs/configs/stage21_funcgroup_w4a4_${RUN_DATE}.json"
