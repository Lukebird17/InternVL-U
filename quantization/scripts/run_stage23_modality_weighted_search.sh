#!/bin/bash

# Stage 23 (InternVL-U): Modality-Weighted CKA Functional-Group Search
#
# 方向 1 实验 2：使用 w_v * CKA_vision + w_l * CKA_text 替代统一 CKA
# 在 functional group 粒度 (attn/mlp) 搜索最优 W4A4 量化算法
#
# 前置步骤（需提前完成）:
#   1. bash scripts/run_build_calibration.sh          → 构建校准数据集 (1000 条)
#   2. bash scripts/run_stage0_with_calibration.sh    → 收集 GPTQ/AWQ 激活
#
# 用法:
#   bash scripts/run_stage23_modality_weighted_search.sh
#   GPU_IDS=4,5 bash scripts/run_stage23_modality_weighted_search.sh
#   GPU_IDS=0,1 VISION_WEIGHT=0.8 TEXT_WEIGHT=0.2 bash scripts/run_stage23_modality_weighted_search.sh

# ===================== 配置 =====================

MODEL_PATH="/home/honglianglu/data/InternVL-U/model"

QUANT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUTPUT_DIR="${QUANT_ROOT}/quantization_outputs/stage23_modality_weighted"

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
if [ "$HAS_STAGE0" = false ]; then
    echo "  Run: bash scripts/run_stage0_with_calibration.sh first"
fi

# GPU 控制
GPU_IDS="${GPU_IDS:-4,5}"

# ========== 可调参数 ==========
# VISION_WEIGHT / TEXT_WEIGHT: 模态加权系数
#   - 默认 0.7/0.3, 意味着视觉 token 保真度更重要
#   - 论文对比: 0.5/0.5 (等权), 0.7/0.3, 0.8/0.2, 0.9/0.1
#   - 调大 VISION_WEIGHT → 在视觉判别任务 (检测/VQA) 上更好
VISION_WEIGHT="${VISION_WEIGHT:-0.7}"
TEXT_WEIGHT="${TEXT_WEIGHT:-0.3}"

# CKA_NUM_SAMPLES: 每层搜索用的样本数 (从 1000 条中子采样)
#   - 默认 200, 越多越稳定但越慢
#   - 200 条 × 9 算法 × 2 组 × 28 层 ≈ 100k 次前向
CKA_NUM_SAMPLES="${CKA_NUM_SAMPLES:-200}"

# SUBSAMPLE_STEP: CKA 计算时序列维下采样步长
SUBSAMPLE_STEP="${SUBSAMPLE_STEP:-5}"

# TARGET_LAYERS: 指定搜索哪些层 (默认全部)
# MAX_CALIB_SAMPLES: 限制加载的校准样本总数

SEED="${SEED:-42}"
RUN_DATE="${RUN_DATE:-$(date +%Y%m%d)}"

# ===================== 运行 =====================

echo "=================================================="
echo "Stage 23: Modality-Weighted CKA Functional-Group Search"
echo "=================================================="
echo "Model:         $MODEL_PATH"
echo "Output:        $OUTPUT_DIR"
echo "GPU IDs:       $GPU_IDS"
echo "Vision weight: $VISION_WEIGHT"
echo "Text weight:   $TEXT_WEIGHT"
echo "CKA samples:   $CKA_NUM_SAMPLES"
echo "Calib:         $CALIB_DATASET"
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

CMD="python -u ${QUANT_ROOT}/stages/stage23_modality_weighted_search.py \
    --model_path $MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    --calibration_dataset $CALIB_DATASET \
    --gpu_ids $GPU_IDS \
    --cka_num_samples $CKA_NUM_SAMPLES \
    --subsample_step $SUBSAMPLE_STEP \
    --vision_weight $VISION_WEIGHT \
    --text_weight $TEXT_WEIGHT \
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
LOG_FILE="$OUTPUT_DIR/stage23_log_${RUN_DATE}.txt"
eval $CMD 2>&1 | tee "$LOG_FILE"

echo ""
echo "Stage 23 completed! (run_date=${RUN_DATE})"
echo "  Results: $OUTPUT_DIR/stage23_search_results_${RUN_DATE}.json"
echo "  Config:  ${QUANT_ROOT}/quantization_outputs/configs/stage23_modality_weighted_w4a4_${RUN_DATE}.json"
