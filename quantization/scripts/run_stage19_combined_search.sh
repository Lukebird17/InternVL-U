#!/bin/bash

# Stage 19 (InternVL-U): Combined-Metric Functional-Group Greedy Search
#
# 改进点：
#   1. CombinedSimilarity (CKA + Cosine + MSE) 替代纯 CKA
#   2. UND + GEN 混合校准（16 + 16 = 32 条样本）
#   3. 仅搜索 W4A4 / W3A4（跳过 W2A4）
#   4. 全程日期命名
#
# 用法：
#   bash quantization/scripts/run_stage19_combined_search.sh
#   RUN_DATE=20260323 bash quantization/scripts/run_stage19_combined_search.sh

set -euo pipefail

# ===================== 配置 =====================

MODEL_PATH="/data/14thdd/users/honglianglu/InternVL-U/model"

QUANT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ACTIVATION_DATA="${QUANT_ROOT}/quantization_outputs/stage0_full_activation/stage0_full_activation_50samples_latest.pt"
if [ ! -f "$ACTIVATION_DATA" ]; then
    ACTIVATION_DATA=""
    echo "NOTE: No stage0 activation data found, GPTQ/AWQ algorithms will be excluded"
fi

MME_DATA_ROOT="/home/honglianglu/data/Bagel/data/mme/MME_Benchmark_release_version"

OUTPUT_DIR="${QUANT_ROOT}/quantization_outputs/stage19_combined"

GPU_IDS="${GPU_IDS:-0,1,2,3}"
MAX_MEM_PER_GPU="40GiB"

NUM_UND_SAMPLES=16
NUM_GEN_SAMPLES=16
SUBSAMPLE_STEP=5
SEED=42

# 组合指标权重
CKA_WEIGHT=0.4
COSINE_WEIGHT=0.3
MSE_WEIGHT=0.3

# 日期（默认今天，可通过环境变量覆盖）
RUN_DATE="${RUN_DATE:-$(date +%Y%m%d)}"

# 可选覆盖（快速测试时取消注释）
# TARGET_LAYERS="0,1,2"
# ALGORITHMS="rtn_w4a4,smooth_rtn_w4a4,gptq_w4a4"
# NUM_UND_SAMPLES=4
# NUM_GEN_SAMPLES=4

# ===================== 运行 =====================

echo "=================================================="
echo "Stage 19 (InternVL-U): Combined-Metric FuncGroup Search"
echo "=================================================="
echo "Model:      $MODEL_PATH"
echo "Output:     $OUTPUT_DIR"
echo "GPU IDs:    $GPU_IDS"
echo "Run date:   $RUN_DATE"
echo "Calib:      ${NUM_UND_SAMPLES} und + ${NUM_GEN_SAMPLES} gen"
echo "Weights:    CKA=${CKA_WEIGHT} Cos=${COSINE_WEIGHT} MSE=${MSE_WEIGHT}"
echo ""

if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model path not found: $MODEL_PATH"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

CMD="python -u ${QUANT_ROOT}/stages/stage19_combined_funcgroup_search.py search \
    --model_path $MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    --gpu_ids $GPU_IDS \
    --max_mem_per_gpu $MAX_MEM_PER_GPU \
    --num_und_samples $NUM_UND_SAMPLES \
    --num_gen_samples $NUM_GEN_SAMPLES \
    --subsample_step $SUBSAMPLE_STEP \
    --seed $SEED \
    --cka_weight $CKA_WEIGHT \
    --cosine_weight $COSINE_WEIGHT \
    --mse_weight $MSE_WEIGHT \
    --run_date $RUN_DATE"

if [ -n "$ACTIVATION_DATA" ] && [ -f "$ACTIVATION_DATA" ]; then
    CMD="$CMD --activation_data $ACTIVATION_DATA"
    echo "Using activation data: $ACTIVATION_DATA"
else
    echo "No activation data, GPTQ/AWQ will be excluded"
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

if [ -n "${GROUP_ORDER:-}" ]; then
    CMD="$CMD --group_order $GROUP_ORDER"
fi

echo ""
LOG_FILE="$OUTPUT_DIR/stage19_log_${RUN_DATE}.txt"
eval $CMD 2>&1 | tee "$LOG_FILE"

echo ""
echo "Stage 19 completed! (run_date=${RUN_DATE})"
echo "  Results:  $OUTPUT_DIR/combined_search_results_${RUN_DATE}.json"
echo "  Configs:  ${QUANT_ROOT}/quantization_outputs/configs/combined_funcgroup_w*a4_${RUN_DATE}.json"
echo "  Log:      $LOG_FILE"
