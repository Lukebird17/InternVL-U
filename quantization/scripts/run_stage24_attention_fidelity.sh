#!/bin/bash

# Stage 24 (InternVL-U): Attention Fidelity Analysis & Search
#
# 方向 2：Attention Map 保真度保护
# Phase A: 测量 FP16 vs Int4 的 attention divergence (KL/JS/cosine)
# Phase B: 联合 CKA + attention fidelity 搜索
# Phase C: 生成 saliency map
#
# ⚠️  本 Stage 使用 eager attention (非 flash_attention_2)
#     显存需求比 Stage 21/23 高约 2-3 倍
#     建议: CKA_NUM_SAMPLES ≤ 100, 或使用更大显存的 GPU
#
# 前置步骤:
#   1. bash scripts/run_build_calibration.sh
#   2. bash scripts/run_stage0_with_calibration.sh
#
# 用法:
#   bash scripts/run_stage24_attention_fidelity.sh
#   GPU_IDS=4,5 bash scripts/run_stage24_attention_fidelity.sh
#   GPU_IDS=0,1 CKA_WEIGHT=0.5 ATTN_WEIGHT=0.5 bash scripts/run_stage24_attention_fidelity.sh

# ===================== 配置 =====================

MODEL_PATH="/home/honglianglu/data/InternVL-U/model"

QUANT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUTPUT_DIR="${QUANT_ROOT}/quantization_outputs/stage24_attention_fidelity"

# 校准数据集
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
GPU_IDS="${GPU_IDS:-5,6}"

# ========== 可调参数 ==========
# CKA_WEIGHT / ATTN_WEIGHT: 联合目标的权重
#   - 默认 0.6/0.4, CKA 仍为主要指标
#   - ATTN_WEIGHT 仅对 attn 组生效 (mlp 组只用 CKA)
#   - 论文对比: 0.7/0.3, 0.6/0.4, 0.5/0.5, 纯CKA (1.0/0.0)
CKA_WEIGHT="${CKA_WEIGHT:-0.6}"
ATTN_WEIGHT="${ATTN_WEIGHT:-0.4}"

# CKA_NUM_SAMPLES: 搜索用样本数
#   - 默认 200, 与 Stage 21/23 保持一致
#   - 如果 OOM, 可降低到 100
CKA_NUM_SAMPLES="${CKA_NUM_SAMPLES:-200}"

# MAX_CALIB_SAMPLES: 从 1000 条中加载多少条 (默认全部)
MAX_CALIB_SAMPLES="${MAX_CALIB_SAMPLES:-1000}"

SUBSAMPLE_STEP="${SUBSAMPLE_STEP:-5}"
SEED="${SEED:-42}"
RUN_DATE="${RUN_DATE:-$(date +%Y%m%d)}"

# ===================== 运行 =====================

echo "=================================================="
echo "Stage 24: Attention Fidelity Analysis & Search"
echo "=================================================="
echo "Model:       $MODEL_PATH"
echo "Output:      $OUTPUT_DIR"
echo "GPU IDs:     $GPU_IDS"
echo "CKA weight:  $CKA_WEIGHT"
echo "Attn weight: $ATTN_WEIGHT"
echo "CKA samples: $CKA_NUM_SAMPLES"
echo "Calib:       $CALIB_DATASET"
echo "Attention:   eager (not flash)"
echo "Run date:    $RUN_DATE"
echo ""

if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model path not found: $MODEL_PATH"
    exit 1
fi

if [ ! -f "$CALIB_DATASET" ]; then
    echo "Error: Calibration dataset not found: $CALIB_DATASET"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

CMD="python -u ${QUANT_ROOT}/stages/stage24_attention_fidelity.py \
    --model_path $MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    --calibration_dataset $CALIB_DATASET \
    --gpu_ids $GPU_IDS \
    --cka_num_samples $CKA_NUM_SAMPLES \
    --max_calib_samples $MAX_CALIB_SAMPLES \
    --cka_weight $CKA_WEIGHT \
    --attn_weight $ATTN_WEIGHT \
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

# Phase C only mode: skip Phase A/B, only generate saliency maps
PHASE_C_ONLY="${PHASE_C_ONLY:-false}"
QUANT_CONFIG="${QUANT_CONFIG:-}"

if [ "$PHASE_C_ONLY" = true ] || [ "$1" = "--phase_c_only" ] || [ "$1" = "phase_c" ]; then
    CMD="$CMD --phase_c_only"
    if [ -n "$QUANT_CONFIG" ]; then
        CMD="$CMD --quant_config $QUANT_CONFIG"
    fi
    echo "[Phase C only] Skipping Phase A/B, generating saliency maps"
    echo ""
fi

LOG_FILE="$OUTPUT_DIR/stage24_log_${RUN_DATE}.txt"
eval $CMD 2>&1 | tee "$LOG_FILE"

echo ""
echo "Stage 24 completed! (run_date=${RUN_DATE})"
echo "  Results:  $OUTPUT_DIR/stage24_results_${RUN_DATE}.json"
echo "  Config:   ${QUANT_ROOT}/quantization_outputs/configs/stage24_attn_fidelity_w4a4_${RUN_DATE}.json"
echo "  Saliency: $OUTPUT_DIR/saliency_maps/"
