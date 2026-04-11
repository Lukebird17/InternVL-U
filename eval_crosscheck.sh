#!/bin/bash
# 2x2 交叉实验：解开 config vs activation stats 的贡献
#
# 实验 A: S25R config + Stage0 全量 stats
# 实验 B: S21 config + Random 子集 stats
#
# 用法:
#   bash eval_crosscheck.sh a     # 只跑实验 A (GPU 5)
#   bash eval_crosscheck.sh b     # 只跑实验 B (GPU 6)
#   bash eval_crosscheck.sh both  # 并行跑 A 和 B

set -euo pipefail

# 使用 internvl conda 环境的 Python
PYTHON="/home/honglianglu/data/.conda/envs/internvl/bin/python"

# HuggingFace 镜像
export HF_ENDPOINT="https://hf-mirror.com"

PROJECT_ROOT="/home/honglianglu/data/InternVL-U"
QUANT_ROOT="${PROJECT_ROOT}/quantization"
MODEL_PATH="/home/honglianglu/data/InternVL-U/model"
OUTPUT_BASE="${QUANT_ROOT}/quantization_outputs/eval_results"

# Configs
S21_CONFIG="${QUANT_ROOT}/quantization_outputs/configs/stage21_funcgroup_w4a4_20260330.json"
S25R_CONFIG="${QUANT_ROOT}/quantization_outputs/configs/stage25_random_w4a4_20260405.json"

# Stage0 full stats (1000 samples)
FULL_HESSIAN="${QUANT_ROOT}/quantization_outputs/stage0_full_activation/gptq_hessian_index_latest.json"
FULL_SMOOTH="${QUANT_ROOT}/quantization_outputs/stage0_full_activation/smoothquant_stats_latest.pt"
FULL_AWQ="${QUANT_ROOT}/quantization_outputs/stage0_full_activation/awq_stats_latest.pt"

# Random subset stats (100 samples)
RAND_STATS="${QUANT_ROOT}/quantization_outputs/stage25_hard_sample/activation_stats_random_20260405"
RAND_HESSIAN="${RAND_STATS}/gptq_hessian_index.json"
RAND_SMOOTH="${RAND_STATS}/smoothquant_stats.pt"
RAND_AWQ="${RAND_STATS}/awq_stats.pt"

GPU_A="${GPU_A:-5}"
GPU_B="${GPU_B:-6}"

run_experiment() {
    local name="$1"
    local config="$2"
    local hessian="$3"
    local smooth="$4"
    local awq="$5"
    local gpu="$6"
    local outdir="${OUTPUT_BASE}/${name}"

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo " ${name}"
    echo " Config: $(basename ${config})"
    echo " Stats:  $(dirname ${hessian} | xargs basename)"
    echo " GPU:    ${gpu}"
    echo " Output: ${outdir}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    CUDA_VISIBLE_DEVICES="${gpu}" ${PYTHON} -u "${PROJECT_ROOT}/eval_internvlu.py" \
        --model_path "${MODEL_PATH}" \
        --quant_config "${config}" \
        --gptq_hessian_index "${hessian}" \
        --smoothquant_stats "${smooth}" \
        --awq_stats "${awq}" \
        --output_dir "${outdir}" \
        --benchmarks mme geneval \
        2>&1 | tee "${outdir}/eval_log.txt"

    echo ""
    echo "✓ ${name} 完成"
    echo ""
}

MODE="${1:-both}"

case "$MODE" in
    a|A)
        echo "=== 实验 A: S25R config + Stage0 全量 stats ==="
        run_experiment \
            "crosscheck_A_s25r_config_full_stats" \
            "$S25R_CONFIG" "$FULL_HESSIAN" "$FULL_SMOOTH" "$FULL_AWQ" "$GPU_A"
        ;;
    b|B)
        echo "=== 实验 B: S21 config + Random 子集 stats ==="
        run_experiment \
            "crosscheck_B_s21_config_rand_stats" \
            "$S21_CONFIG" "$RAND_HESSIAN" "$RAND_SMOOTH" "$RAND_AWQ" "$GPU_B"
        ;;
    both)
        echo "=== 并行启动实验 A 和 B ==="
        mkdir -p "${OUTPUT_BASE}/crosscheck_A_s25r_config_full_stats"
        mkdir -p "${OUTPUT_BASE}/crosscheck_B_s21_config_rand_stats"

        # 实验 A in background
        (
            run_experiment \
                "crosscheck_A_s25r_config_full_stats" \
                "$S25R_CONFIG" "$FULL_HESSIAN" "$FULL_SMOOTH" "$FULL_AWQ" "$GPU_A"
        ) &
        PID_A=$!

        # 实验 B in background
        (
            run_experiment \
                "crosscheck_B_s21_config_rand_stats" \
                "$S21_CONFIG" "$RAND_HESSIAN" "$RAND_SMOOTH" "$RAND_AWQ" "$GPU_B"
        ) &
        PID_B=$!

        echo "实验 A PID: ${PID_A} (GPU ${GPU_A})"
        echo "实验 B PID: ${PID_B} (GPU ${GPU_B})"
        echo ""
        echo "等待两个实验完成..."

        wait $PID_A
        STATUS_A=$?
        wait $PID_B
        STATUS_B=$?

        echo ""
        echo "══════════════════════════════════════════════"
        echo " 交叉实验结果汇总"
        echo "══════════════════════════════════════════════"

        # Print results
        for exp in "crosscheck_A_s25r_config_full_stats" "crosscheck_B_s21_config_rand_stats"; do
            mme_file="${OUTPUT_BASE}/${exp}/mme_results/results.txt"
            echo ""
            echo "--- ${exp} ---"
            if [ -f "$mme_file" ]; then
                cat "$mme_file"
            else
                echo "  MME results not found"
            fi
        done

        echo ""
        echo "实验 A 退出码: ${STATUS_A}"
        echo "实验 B 退出码: ${STATUS_B}"
        ;;
    *)
        echo "用法: bash eval_crosscheck.sh [a|b|both]"
        exit 1
        ;;
esac
