#!/bin/bash

# Stage 15 (InternVL-U): CALM — CKA-Guided Adaptive Layer-Wise Modularization

# ===================== 配置 =====================

MODEL_PATH="/home/honglianglu/data/InternVL-U/model"

# GPTQ/AWQ 激活数据（由 stage0 生成，无则自动排除 GPTQ/AWQ 算法）
# 自动检测 stage0 输出
QUANT_ROOT_TMP="$(cd "$(dirname "$0")/.." && pwd)"
ACTIVATION_DATA="${QUANT_ROOT_TMP}/quantization_outputs/stage0_full_activation/stage0_full_activation_50samples_latest.pt"
if [ ! -f "$ACTIVATION_DATA" ]; then
    ACTIVATION_DATA=""
    echo "NOTE: No stage0 activation data found, GPTQ/AWQ algorithms will be excluded"
    echo "  Run scripts/run_stage0_collect_activations.sh first for full algorithm coverage"
fi

MME_DATA_ROOT="/home/honglianglu/data/Bagel/data/mme/MME_Benchmark_release_version"

QUANT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUTPUT_DIR="${QUANT_ROOT}/quantization_outputs/stage15_calm"

GPU_IDS="${GPU_IDS:-2,3}"
MAX_MEM_PER_GPU="40GiB"

NUM_UND_SAMPLES=16
NUM_GEN_SAMPLES=16
SUBSAMPLE_STEP=5
SEED=42

RUN_DATE="${RUN_DATE:-$(date +%Y%m%d)}"

# 快速测试
# TARGET_LAYERS="0,1,2"
# ALGORITHMS="rtn_w4a4,smooth_rtn_w4a4"
# NUM_UND_SAMPLES=4
# NUM_GEN_SAMPLES=4

# ===================== 运行 =====================

echo "=================================================="
echo "Stage 15 (InternVL-U): CALM — CKA Layer-Wise Search"
echo "=================================================="
echo "Model:    $MODEL_PATH"
echo "Output:   $OUTPUT_DIR"
echo "GPU IDs:  $GPU_IDS"
echo "Run date: $RUN_DATE"
echo "Calib:    ${NUM_UND_SAMPLES} und + ${NUM_GEN_SAMPLES} gen"
echo ""

if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model path not found: $MODEL_PATH"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

CMD="python -u ${QUANT_ROOT}/stages/stage15_calm_search.py \
    --model_path $MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    --gpu_ids $GPU_IDS \
    --max_mem_per_gpu $MAX_MEM_PER_GPU \
    --num_und_samples $NUM_UND_SAMPLES \
    --num_gen_samples $NUM_GEN_SAMPLES \
    --subsample_step $SUBSAMPLE_STEP \
    --seed $SEED \
    --run_date $RUN_DATE"

if [ -n "$ACTIVATION_DATA" ] && [ -f "$ACTIVATION_DATA" ]; then
    CMD="$CMD --activation_data $ACTIVATION_DATA"
    echo "Using activation data: $ACTIVATION_DATA"
fi

if [ -d "$MME_DATA_ROOT" ]; then
    CMD="$CMD --mme_data_root $MME_DATA_ROOT"
    echo "Using MME calibration: $MME_DATA_ROOT"
fi

if [ -n "${TARGET_LAYERS:-}" ]; then
    CMD="$CMD --target_layers $TARGET_LAYERS"
fi

if [ -n "${ALGORITHMS:-}" ]; then
    CMD="$CMD --algorithms $ALGORITHMS"
fi

echo ""
LOG_FILE="$OUTPUT_DIR/stage15_log_${RUN_DATE}.txt"
eval $CMD 2>&1 | tee "$LOG_FILE"

echo ""
echo "Stage 15 completed! (run_date=${RUN_DATE})"
echo "  Results: $OUTPUT_DIR"
