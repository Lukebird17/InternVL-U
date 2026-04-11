#!/bin/bash

# Stage 22 (InternVL-U): Modality-Specific Outlier Analysis
#
# 方向 1 实验 1：分析视觉 Token 与文本 Token 的激活分布差异
# 输出：逐层统计 JSON + CKA 敏感度分析
#
# 前置步骤（需提前完成）:
#   1. bash scripts/run_build_calibration.sh          → 构建校准数据集 (1000 条)
#
# 用法:
#   bash scripts/run_stage22_modality_analysis.sh
#   GPU_IDS=4,5 bash scripts/run_stage22_modality_analysis.sh
#   GPU_IDS=0 NUM_SAMPLES=100 bash scripts/run_stage22_modality_analysis.sh

# ===================== 配置 =====================

MODEL_PATH="/home/honglianglu/data/InternVL-U/model"

QUANT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUTPUT_DIR="${QUANT_ROOT}/quantization_outputs/stage22_modality_analysis"

# 校准数据集 (1000 条 Flickr8K VQA)
CALIB_DATA_DIR="${QUANT_ROOT}/quantization_outputs/calibration_data"
CALIB_DATASET="${CALIB_DATASET:-${CALIB_DATA_DIR}/calibration_dataset_latest.json}"

# Stage 0 激活统计 (GPTQ/SmoothQuant/AWQ)
STAGE0_DIR="${QUANT_ROOT}/quantization_outputs/stage0_activations"
GPTQ_HESSIAN="${GPTQ_HESSIAN:-${STAGE0_DIR}/gptq_hessian_index_latest.json}"
SMOOTH_STATS="${SMOOTH_STATS:-${STAGE0_DIR}/smoothquant_stats_latest.pt}"
AWQ_STATS="${AWQ_STATS:-${STAGE0_DIR}/awq_stats_latest.pt}"

# GPU 控制
GPU_IDS="${GPU_IDS:-0}"

# ========== 可调参数 ==========
# NUM_SAMPLES: 用于分析的校准样本数 (默认 200, 越多越准但越慢)
#   - 推荐 100~500, 需要有图样本才能分析视觉 token
NUM_SAMPLES="${NUM_SAMPLES:-1000}"

# SUBSAMPLE_STEP: CKA 计算时序列维下采样步长 (默认 5)
#   - 越小越精确但越慢, 推荐 3~10
SUBSAMPLE_STEP="${SUBSAMPLE_STEP:-5}"

# QUANT_METHODS: 控制 Phase 3 测试哪些量化方法 (默认全部 7 种)
#   - 可选值(逗号分隔): rtn_w4a4,gptq_w4a4,smooth_a50_gptq_w4a4,
#     smooth_a70_gptq_w4a4,svdquant_a50_w4a4,svdquant_a70_w4a4,awq_svd_rtn_w4a4
#   - 快速版: QUANT_METHODS="rtn_w4a4,gptq_w4a4,svdquant_a50_w4a4" (3 种)
#   - 留空 = 全部 7 种
QUANT_METHODS="rtn_w4a4,gptq_w4a4,smooth_a50_gptq_w4a4,smooth_a70_gptq_w4a4,svdquant_a50_w4a4,svdquant_a70_w4a4,awq_svd_rtn_w4a4"
# TARGET_LAYERS: 指定分析哪些层 (默认全部)
#   - 例: TARGET_LAYERS="0,5,10,15,20,27" 只分析 6 层, 大幅加速
#   - 留空 = 分析所有层

SEED="${SEED:-42}"
RUN_DATE="${RUN_DATE:-$(date +%Y%m%d)}"

# ===================== 运行 =====================

echo "=================================================="
echo "Stage 22: Modality-Specific Outlier Analysis"
echo "=================================================="
echo "Model:       $MODEL_PATH"
echo "Output:      $OUTPUT_DIR"
echo "GPU IDs:     $GPU_IDS"
echo "Num samples: $NUM_SAMPLES"
echo "Calib:       $CALIB_DATASET"
echo "Run date:    $RUN_DATE"
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

CMD="python -u ${QUANT_ROOT}/stages/stage22_modality_outlier_analysis.py \
    --model_path $MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    --calibration_dataset $CALIB_DATASET \
    --gpu_ids $GPU_IDS \
    --num_samples $NUM_SAMPLES \
    --subsample_step $SUBSAMPLE_STEP \
    --seed $SEED \
    --run_date $RUN_DATE"

if [ -f "$GPTQ_HESSIAN" ]; then
    CMD="$CMD --gptq_hessian_index $GPTQ_HESSIAN"
fi
if [ -f "$SMOOTH_STATS" ]; then
    CMD="$CMD --smoothquant_stats $SMOOTH_STATS"
fi
if [ -f "$AWQ_STATS" ]; then
    CMD="$CMD --awq_stats $AWQ_STATS"
fi
if [ -n "${TARGET_LAYERS:-}" ]; then
    CMD="$CMD --target_layers $TARGET_LAYERS"
fi
if [ -n "${QUANT_METHODS:-}" ]; then
    CMD="$CMD --quant_methods $QUANT_METHODS"
    echo "Quant methods: $QUANT_METHODS"
fi

echo ""
LOG_FILE="$OUTPUT_DIR/stage22_log_${RUN_DATE}.txt"
eval $CMD 2>&1 | tee "$LOG_FILE"

echo ""
echo "Stage 22 completed! (run_date=${RUN_DATE})"
echo "  Results: $OUTPUT_DIR/stage22_modality_analysis_${RUN_DATE}.json"
