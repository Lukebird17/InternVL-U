#!/bin/bash

# Stage 0 (InternVL-U): 收集完整激活值（用于 GPTQ/AWQ）
#
# 校准策略：混合理解 + 生成样本
#   - UND: 使用 MME benchmark 的 VQA 图文对（或 fallback 到纯文本 prompt）
#   - GEN: 使用 T2I 文本 prompt（走 generation_mode="image" 路径）
#
# 输出: stage0_full_activation_Xsamples_latest.pt

# ===================== 配置 =====================

MODEL_PATH="/home/honglianglu/data/InternVL-U/model"

MME_DATA_ROOT="/home/honglianglu/data/Bagel/data/mme/MME_Benchmark_release_version"

QUANT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUTPUT_DIR="${QUANT_ROOT}/quantization_outputs/stage0_full_activation"

GPU_IDS="0,1,2,3"
MAX_MEM_PER_GPU="40GiB"

# 每层最大激活样本数（GPTQ Hessian 计算用）
MAX_SAMPLES=50

# 校准数据配比
NUM_UND=8       # 理解任务样本数
NUM_GEN=8       # 生成任务样本数

# 或者用比例方式：
# TOTAL_SAMPLES=16
# UND_RATIO=0.5     # 50% 理解 + 50% 生成

SEED=42

# ===================== 运行 =====================

echo "=================================================="
echo "Stage 0 (InternVL-U): Full Activation Collection"
echo "=================================================="
echo "Model: $MODEL_PATH"
echo "Output: $OUTPUT_DIR"
echo "GPU IDs: $GPU_IDS"
echo "Calibration: ${NUM_UND} UND + ${NUM_GEN} GEN samples"
echo ""

if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model path not found: $MODEL_PATH"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

CMD="python -u ${QUANT_ROOT}/stages/stage0_collect_activations.py \
    --model_path $MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    --gpu_ids $GPU_IDS \
    --max_mem_per_gpu $MAX_MEM_PER_GPU \
    --max_samples $MAX_SAMPLES \
    --num_und $NUM_UND \
    --num_gen $NUM_GEN \
    --seed $SEED"

if [ -d "$MME_DATA_ROOT" ]; then
    CMD="$CMD --mme_data_root $MME_DATA_ROOT"
    echo "Using MME images: $MME_DATA_ROOT"
fi

if [ -n "${TOTAL_SAMPLES:-}" ] && [ -n "${UND_RATIO:-}" ]; then
    CMD="$CMD --total_samples $TOTAL_SAMPLES --und_ratio $UND_RATIO"
    echo "Override: total=$TOTAL_SAMPLES, und_ratio=$UND_RATIO"
fi

echo ""
eval $CMD 2>&1 | tee "$OUTPUT_DIR/stage0_log_$(date +%Y%m%d_%H%M%S).txt"

echo ""
echo "Stage 0 completed! Output: $OUTPUT_DIR"
echo "Activation file: $OUTPUT_DIR/stage0_full_activation_${MAX_SAMPLES}samples_latest.pt"
