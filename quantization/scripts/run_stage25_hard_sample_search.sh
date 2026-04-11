#!/bin/bash

# Stage 25 (InternVL-U): Hard-Sample Aware Calibration Search
#
# 方向 3：硬样本校准搜索
# Phase 0: 用 FP16 模型给 1000 条校准样本打困惑度分
# Phase 1: 用随机 100 条跑 functional group 搜索 → baseline 策略
# Phase 2: 用 top-100 困惑度硬样本跑搜索 → hard 策略
# Phase 3: 对比两种策略的差异
#
# 前置步骤:
#   1. bash scripts/run_build_calibration.sh
#   2. bash scripts/run_stage0_with_calibration.sh
#
# 用法:
#   bash scripts/run_stage25_hard_sample_search.sh
#   GPU_IDS=4,5 bash scripts/run_stage25_hard_sample_search.sh
#   GPU_IDS=0,1 NUM_HARD=200 NUM_SEARCH=150 bash scripts/run_stage25_hard_sample_search.sh

# ===================== 配置 =====================

MODEL_PATH="/home/honglianglu/data/InternVL-U/model"

QUANT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUTPUT_DIR="${QUANT_ROOT}/quantization_outputs/stage25_hard_sample"

# 校准数据集 (完整 1000 条, 全部用于 difficulty scoring)
CALIB_DATA_DIR="${QUANT_ROOT}/quantization_outputs/calibration_data"
CALIB_DATASET="${CALIB_DATASET:-${CALIB_DATA_DIR}/calibration_dataset_latest.json}"

# Stage 0 输出
STAGE0_DIR="${QUANT_ROOT}/quantization_outputs/stage0_full_activation"
GPTQ_HESSIAN_INDEX="${STAGE0_DIR}/gptq_hessian_index_latest.json"
SMOOTHQUANT_STATS="${STAGE0_DIR}/smoothquant_stats_latest.pt"
AWQ_STATS="${STAGE0_DIR}/awq_stats_latest.pt"

HAS_STAGE0=true
if [ ! -f "$GPTQ_HESSIAN_INDEX" ]; then
    echo "WARNING: GPTQ Hessian not found: $GPTQ_HESSIAN_INDEX"
    HAS_STAGE0=false
fi
if [ "$HAS_STAGE0" = false ]; then
    echo "  Run: bash scripts/run_stage0_with_calibration.sh first"
fi

# GPU 控制
GPU_IDS="${GPU_IDS:-6}"

# ========== 可调参数 ==========
# NUM_HARD: 筛选多少条硬样本 (默认 100)
#   - 从 1000 条中按 perplexity 排序取 top-K
#   - 推荐 50~200
NUM_HARD="${NUM_HARD:-100}"

# NUM_SEARCH: 每次搜索用多少条样本 (默认 100)
#   - random 搜索和 hard 搜索各用这么多条
#   - 越多越稳定, 建议 ≤ NUM_HARD
NUM_SEARCH="${NUM_SEARCH:-100}"

# MAX_CALIB_SAMPLES: 限制加载的校准样本总数 (默认 1000 全部加载并评分)
#   - Phase 0 会对所有加载的样本计算 perplexity
#   - 1000 条 × 1 次前向 ≈ 30-60 分钟
MAX_CALIB_SAMPLES="${MAX_CALIB_SAMPLES:-1000}"

SUBSAMPLE_STEP="${SUBSAMPLE_STEP:-5}"
SEED="${SEED:-42}"
RUN_DATE="${RUN_DATE:-$(date +%Y%m%d)}"

# ===================== 运行 =====================

echo "=================================================="
echo "Stage 25: Hard-Sample Aware Calibration Search"
echo "=================================================="
echo "Model:         $MODEL_PATH"
echo "Output:        $OUTPUT_DIR"
echo "GPU IDs:       $GPU_IDS"
echo "Calib dataset: $CALIB_DATASET"
echo "Max calib:     $MAX_CALIB_SAMPLES (scored for difficulty)"
echo "Hard samples:  $NUM_HARD (top perplexity)"
echo "Search each:   $NUM_SEARCH samples"
echo "GPTQ Hessian:  $GPTQ_HESSIAN_INDEX"
echo "Run date:      $RUN_DATE"
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

CMD="python -u ${QUANT_ROOT}/stages/stage25_hard_sample_search.py \
    --model_path $MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    --calibration_dataset $CALIB_DATASET \
    --gpu_ids $GPU_IDS \
    --num_hard_samples $NUM_HARD \
    --num_search_samples $NUM_SEARCH \
    --max_calib_samples $MAX_CALIB_SAMPLES \
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

echo ""
LOG_FILE="$OUTPUT_DIR/stage25_log_${RUN_DATE}.txt"
eval $CMD 2>&1 | tee "$LOG_FILE"

echo ""
echo "Stage 25 completed! (run_date=${RUN_DATE})"
echo "  Results:      $OUTPUT_DIR/stage25_results_${RUN_DATE}.json"
echo "  Difficulty:   $OUTPUT_DIR/difficulty_scores_${RUN_DATE}.json"
echo "  Random config: ${QUANT_ROOT}/quantization_outputs/configs/stage25_random_w4a4_${RUN_DATE}.json"
echo "  Hard config:   ${QUANT_ROOT}/quantization_outputs/configs/stage25_hard_w4a4_${RUN_DATE}.json"
