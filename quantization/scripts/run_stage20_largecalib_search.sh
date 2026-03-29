#!/bin/bash

# Stage 20 (InternVL-U): Large-Calibration CKA Layer-Wise Search (W4A4 only)
#
# 前置步骤（需提前完成）:
#   1. bash scripts/run_build_calibration.sh          → 构建校准数据集
#   2. bash scripts/run_stage0_with_calibration.sh    → 收集 GPTQ/AWQ 激活
#
# 本脚本仅执行 CKA 搜索：
#   - 读取校准数据集计算 CKA
#   - 读取 Stage 0 激活数据用于 GPTQ/AWQ 权重量化
#   - 精简 5 算法池, 仅搜索 W4A4
#
# 用法:
#   bash scripts/run_stage20_largecalib_search.sh
#   GPU_IDS=2,3 bash scripts/run_stage20_largecalib_search.sh

# ===================== 配置 =====================

MODEL_PATH="/data/14thdd/users/honglianglu/InternVL-U/model"

QUANT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUTPUT_DIR="${QUANT_ROOT}/quantization_outputs/stage20_largecalib"

# 校准数据集（默认用 latest 软链接）
CALIB_DATA_DIR="${QUANT_ROOT}/quantization_outputs/calibration_data"
CALIB_DATASET="${CALIB_DATASET:-${CALIB_DATA_DIR}/calibration_dataset_latest.json}"

# Stage 0 输出（三类文件）
STAGE0_DIR="${QUANT_ROOT}/quantization_outputs/stage0_full_activation"

GPTQ_HESSIAN_INDEX="${STAGE0_DIR}/gptq_hessian_index_latest.json"
SMOOTHQUANT_STATS="${STAGE0_DIR}/smoothquant_stats_latest.pt"
AWQ_STATS="${STAGE0_DIR}/awq_stats_latest.pt"

# 检查是否存在
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
    echo "  GPTQ/AWQ/SmoothQuant algorithms will be excluded"
    echo "  Run: bash scripts/run_stage0_with_calibration.sh first"
fi

GPU_IDS="${GPU_IDS:-0,1}"
MAX_MEM_PER_GPU="40GiB"

# CKA 搜索子采样：从完整校准集中抽取多少条样本用于每层 CKA 比较
# 200 条 × 9 算法 × 28 层 ≈ 50k 次前向，可接受
# 可通过环境变量调大，如 CKA_NUM_SAMPLES=500
CKA_NUM_SAMPLES="${CKA_NUM_SAMPLES:-200}"

SUBSAMPLE_STEP=5
SEED=42

RUN_DATE="${RUN_DATE:-$(date +%Y%m%d)}"

# 快速测试 (取消注释)
# TARGET_LAYERS="0,1,2"
# MAX_CALIB_SAMPLES=20

# ===================== 运行 =====================

echo "=================================================="
echo "Stage 20: CKA Layer-Wise Search (W4A4 only)"
echo "=================================================="
echo "Model:    $MODEL_PATH"
echo "Output:   $OUTPUT_DIR"
echo "GPU IDs:  $GPU_IDS"
echo "Run date: $RUN_DATE"
echo "Calib:    $CALIB_DATASET"
echo "GPTQ Hessian: ${GPTQ_HESSIAN_INDEX}"
echo "SmoothQuant:  ${SMOOTHQUANT_STATS}"
echo "AWQ stats:    ${AWQ_STATS}"
echo "CKA samples:  $CKA_NUM_SAMPLES (subsampled from calib for search)"
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

CMD="python -u ${QUANT_ROOT}/stages/stage20_largecalib_search.py \
    --model_path $MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    --calibration_dataset $CALIB_DATASET \
    --gpu_ids $GPU_IDS \
    --max_mem_per_gpu $MAX_MEM_PER_GPU \
    --cka_num_samples $CKA_NUM_SAMPLES \
    --subsample_step $SUBSAMPLE_STEP \
    --seed $SEED \
    --run_date $RUN_DATE"

# Stage 0 新格式（三类文件）
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
LOG_FILE="$OUTPUT_DIR/stage20_log_${RUN_DATE}.txt"
eval $CMD 2>&1 | tee "$LOG_FILE"

echo ""
echo "Stage 20 completed! (run_date=${RUN_DATE})"
echo "  Results: $OUTPUT_DIR"
echo "  Config:  ${QUANT_ROOT}/quantization_outputs/configs/stage20_largecalib_w4a4_${RUN_DATE}.json"
