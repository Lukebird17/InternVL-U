#!/bin/bash
#
# InternVL-U 量化配置评测脚本
#
# 可自由选择要评测的量化配置和 benchmark
#
# === 用法 ===
#
#   bash eval_quant.sh                         # 交互式：列出所有配置让你选
#   bash eval_quant.sh --list                  # 仅列出可用配置
#   bash eval_quant.sh --config stage23        # 模糊匹配 "stage23" 的配置
#   bash eval_quant.sh --config 1              # 按编号选（编号见 --list）
#   bash eval_quant.sh --config 1,3            # 多选：编号 1 和 3
#   bash eval_quant.sh --config all            # 全部配置
#   bash eval_quant.sh --config stage23 --bench mme       # 只跑 MME
#   bash eval_quant.sh --config stage23 --bench geneval   # 只跑 GenEval
#   bash eval_quant.sh --config baseline                  # FP16 基线（不加量化）
#   bash eval_quant.sh --config baseline,stage23 --bench mme  # 基线 + stage23，仅 MME
#
# === 环境变量 ===
#
#   GPU_IDS=4,5          使用的 GPU（默认 4,5）
#   MODEL_PATH=...       模型路径
#

set -euo pipefail

# ===================== 默认配置 =====================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
QUANT_ROOT="${SCRIPT_DIR}/quantization"
CONFIG_DIR="${QUANT_ROOT}/quantization_outputs/configs"
STAGE0_DIR="${QUANT_ROOT}/quantization_outputs/stage0_full_activation"
OUTPUT_BASE="${QUANT_ROOT}/quantization_outputs/eval_results"

MODEL_PATH="${MODEL_PATH:-/home/honglianglu/data/InternVL-U/model}"
GPU_IDS="${GPU_IDS:-4,5}"
SEED="${SEED:-42}"
GENEVAL_RESOLUTION="${GENEVAL_RESOLUTION:-1024}"
GENEVAL_CFG_SCALE="${GENEVAL_CFG_SCALE:-4.0}"
GENEVAL_STEPS="${GENEVAL_STEPS:-20}"

# 校准数据
GPTQ_HESSIAN_INDEX="${STAGE0_DIR}/gptq_hessian_index_latest.json"
SMOOTHQUANT_STATS="${STAGE0_DIR}/smoothquant_stats_latest.pt"
AWQ_STATS="${STAGE0_DIR}/awq_stats_latest.pt"

# ===================== 自动发现配置 =====================

declare -a CONFIG_NAMES=()
declare -a CONFIG_PATHS=()

discover_configs() {
    local idx=0
    for f in "${CONFIG_DIR}"/*.json; do
        [ -f "$f" ] || continue
        local name
        name="$(basename "$f" .json)"
        CONFIG_NAMES+=("$name")
        CONFIG_PATHS+=("$f")
        idx=$((idx + 1))
    done
}

discover_configs

# ===================== 显示可用配置 =====================

print_configs() {
    echo ""
    echo "可用的量化配置 (${CONFIG_DIR}):"
    echo "────────────────────────────────────────────────────"
    printf "  %3s  %-50s\n" "#" "配置名称"
    echo "  ---  --------------------------------------------------"
    printf "  %3s  %-50s  %s\n" "0" "baseline (FP16, 无量化)" ""
    for i in "${!CONFIG_NAMES[@]}"; do
        printf "  %3d  %-50s\n" "$((i + 1))" "${CONFIG_NAMES[$i]}"
    done
    echo "────────────────────────────────────────────────────"
    echo ""
}

# ===================== 解析参数 =====================

ARG_CONFIG=""
ARG_BENCH="both"
ARG_LIST=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --list|-l)
            ARG_LIST=true; shift ;;
        --config|-c)
            ARG_CONFIG="$2"; shift 2 ;;
        --bench|-b)
            ARG_BENCH="$2"; shift 2 ;;
        --gpu|-g)
            GPU_IDS="$2"; shift 2 ;;
        --help|-h)
            echo "用法: bash eval_quant.sh [选项]"
            echo ""
            echo "选项:"
            echo "  --list,   -l            列出所有可用配置"
            echo "  --config, -c CONFIG     选择配置（编号/名称/模糊匹配/all/baseline）"
            echo "                          多选用逗号分隔: 1,3 或 stage23,stage25"
            echo "  --bench,  -b BENCH      benchmark: mme / geneval / both (默认 both)"
            echo "  --gpu,    -g GPU_IDS    GPU 编号 (默认 ${GPU_IDS})"
            echo "  --help,   -h            显示帮助"
            echo ""
            echo "示例:"
            echo "  bash eval_quant.sh --list"
            echo "  bash eval_quant.sh --config stage23 --bench mme"
            echo "  bash eval_quant.sh --config 1,3 --bench geneval"
            echo "  bash eval_quant.sh --config baseline,stage23"
            echo "  bash eval_quant.sh --config all"
            exit 0 ;;
        *)
            echo "未知参数: $1 (用 --help 查看用法)"; exit 1 ;;
    esac
done

if $ARG_LIST; then
    print_configs
    exit 0
fi

# ===================== 选择配置 =====================

declare -a SELECTED_NAMES=()
declare -a SELECTED_PATHS=()

resolve_selection() {
    local sel="$1"

    # baseline
    if [[ "$sel" == "baseline" || "$sel" == "0" ]]; then
        SELECTED_NAMES+=("fp_baseline")
        SELECTED_PATHS+=("")
        return
    fi

    # all
    if [[ "$sel" == "all" ]]; then
        for i in "${!CONFIG_NAMES[@]}"; do
            SELECTED_NAMES+=("${CONFIG_NAMES[$i]}")
            SELECTED_PATHS+=("${CONFIG_PATHS[$i]}")
        done
        return
    fi

    # 按编号
    if [[ "$sel" =~ ^[0-9]+$ ]]; then
        local idx=$((sel - 1))
        if [ "$idx" -ge 0 ] && [ "$idx" -lt "${#CONFIG_NAMES[@]}" ]; then
            SELECTED_NAMES+=("${CONFIG_NAMES[$idx]}")
            SELECTED_PATHS+=("${CONFIG_PATHS[$idx]}")
            return
        else
            echo "[错误] 编号 ${sel} 超出范围 (1-${#CONFIG_NAMES[@]})"
            exit 1
        fi
    fi

    # 模糊匹配
    local matched=false
    for i in "${!CONFIG_NAMES[@]}"; do
        if [[ "${CONFIG_NAMES[$i]}" == *"$sel"* ]]; then
            SELECTED_NAMES+=("${CONFIG_NAMES[$i]}")
            SELECTED_PATHS+=("${CONFIG_PATHS[$i]}")
            matched=true
        fi
    done
    if ! $matched; then
        echo "[错误] 没有匹配 \"${sel}\" 的配置"
        print_configs
        exit 1
    fi
}

if [ -n "$ARG_CONFIG" ]; then
    IFS=',' read -ra SELECTIONS <<< "$ARG_CONFIG"
    for sel in "${SELECTIONS[@]}"; do
        resolve_selection "$sel"
    done
else
    # 交互式选择
    print_configs
    echo "请输入要评测的配置（编号/名称/逗号分隔多选/all/baseline）:"
    read -rp "> " user_input

    if [ -z "$user_input" ]; then
        echo "未选择任何配置，退出。"
        exit 0
    fi

    IFS=',' read -ra SELECTIONS <<< "$user_input"
    for sel in "${SELECTIONS[@]}"; do
        sel="$(echo "$sel" | xargs)"  # trim whitespace
        resolve_selection "$sel"
    done
fi

if [ ${#SELECTED_NAMES[@]} -eq 0 ]; then
    echo "未选择任何配置，退出。"
    exit 0
fi

# ===================== 确认 =====================

BENCHMARKS=""
case "$ARG_BENCH" in
    mme)     BENCHMARKS="mme" ;;
    geneval) BENCHMARKS="geneval" ;;
    both)    BENCHMARKS="mme geneval" ;;
    *)       echo "[错误] --bench 只支持 mme / geneval / both"; exit 1 ;;
esac

echo ""
echo "══════════════════════════════════════════════════════════"
echo " InternVL-U 量化评测"
echo "══════════════════════════════════════════════════════════"
echo " 模型:       ${MODEL_PATH}"
echo " GPU:        ${GPU_IDS}"
echo " Benchmark:  ${BENCHMARKS}"
echo " 输出目录:   ${OUTPUT_BASE}"
echo ""
echo " 选中的配置 (${#SELECTED_NAMES[@]} 个):"
for i in "${!SELECTED_NAMES[@]}"; do
    if [ -z "${SELECTED_PATHS[$i]}" ]; then
        echo "   [$((i+1))] ${SELECTED_NAMES[$i]}  (FP16 无量化)"
    else
        echo "   [$((i+1))] ${SELECTED_NAMES[$i]}"
    fi
done
echo "══════════════════════════════════════════════════════════"
echo ""

read -rp "确认开始评测? [Y/n] " confirm
confirm="${confirm:-Y}"
if [[ ! "$confirm" =~ ^[Yy] ]]; then
    echo "已取消。"
    exit 0
fi

# ===================== 构建校准数据参数 =====================

build_calib_args() {
    local args=""
    if [ -f "$GPTQ_HESSIAN_INDEX" ]; then
        args="${args} --gptq_hessian_index ${GPTQ_HESSIAN_INDEX}"
        args="${args} --smoothquant_stats ${SMOOTHQUANT_STATS}"
        args="${args} --awq_stats ${AWQ_STATS}"
    fi
    echo "$args"
}

# ===================== 执行评测 =====================

TOTAL=${#SELECTED_NAMES[@]}
PASSED=0
FAILED=0

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"

for i in "${!SELECTED_NAMES[@]}"; do
    task_name="${SELECTED_NAMES[$i]}"
    config_path="${SELECTED_PATHS[$i]}"
    task_output="${OUTPUT_BASE}/${task_name}"
    log_file="${task_output}/eval_log.txt"

    mkdir -p "$task_output"

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo " [$((i+1))/${TOTAL}] ${task_name}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    CMD="python -u ${SCRIPT_DIR}/eval_internvlu.py \
        --model_path ${MODEL_PATH} \
        --output_dir ${task_output} \
        --benchmarks ${BENCHMARKS} \
        --seed ${SEED} \
        --geneval_resolution ${GENEVAL_RESOLUTION} \
        --geneval_cfg_scale ${GENEVAL_CFG_SCALE} \
        --geneval_steps ${GENEVAL_STEPS}"

    if [ -n "$config_path" ]; then
        CMD="${CMD} --quant_config ${config_path}"
        CMD="${CMD} $(build_calib_args)"
    fi

    echo "  输出: ${task_output}"
    echo "  日志: ${log_file}"
    echo ""

    if eval "$CMD" 2>&1 | tee "$log_file"; then
        PASSED=$((PASSED + 1))
        echo ""
        echo "  ✓ ${task_name} 完成"
    else
        FAILED=$((FAILED + 1))
        echo ""
        echo "  ✗ ${task_name} 失败 (详见 ${log_file})"
    fi
done

# ===================== 汇总 =====================

echo ""
echo "══════════════════════════════════════════════════════════"
echo " 评测汇总"
echo "══════════════════════════════════════════════════════════"
echo ""

printf "  %-45s  %-15s  %s\n" "配置" "MME" "GenEval"
printf "  %-45s  %-15s  %s\n" "---------------------------------------------" "---------------" "-------"

for i in "${!SELECTED_NAMES[@]}"; do
    task_name="${SELECTED_NAMES[$i]}"
    task_dir="${OUTPUT_BASE}/${task_name}"

    mme_status="--"
    geneval_status="--"

    mme_results="${task_dir}/mme_results/results.txt"
    if [ -f "$mme_results" ]; then
        mme_score=$(grep -oP 'Total score: \K[\d.]+' "$mme_results" 2>/dev/null || echo "done")
        mme_status="$mme_score"
    fi

    geneval_summary="${task_dir}/geneval_results/summary.txt"
    if [ -f "$geneval_summary" ]; then
        geneval_status="done"
    elif [ -f "${task_dir}/geneval_results/results.jsonl" ]; then
        geneval_status="partial"
    fi

    printf "  %-45s  %-15s  %s\n" "$task_name" "$mme_status" "$geneval_status"
done

echo ""
echo " 通过: ${PASSED}  失败: ${FAILED}  总计: ${TOTAL}"
echo " 结果目录: ${OUTPUT_BASE}"
echo "══════════════════════════════════════════════════════════"
