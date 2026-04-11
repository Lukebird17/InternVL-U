#!/bin/bash
#
# 并行评测所有量化配置 (Stage15/17/18/19 × W4/W3)
#
# === 两阶段调度策略 ===
#
#   Phase 1 — MME (理解，轻量)：1 卡/任务，最多 8 路并行
#     模型 bfloat16 全放一张卡 (~15GB)，VQA 推理峰值 ~20GB
#     一张 4090 48GB 绰绰有余
#
#   Phase 2 — GenEval (生成，重量)：2 卡/任务，最多 4 路并行
#     Diffusion 生成需要额外显存，用 2 卡更安全
#
# === 用法 ===
#
#   bash quantization/scripts/run_all_evals.sh                     # 全部配置
#   bash quantization/scripts/run_all_evals.sh w4a4                # 仅 W4A4
#   bash quantization/scripts/run_all_evals.sh baseline            # FP 基线
#   bash quantization/scripts/run_all_evals.sh all mme             # 全部，只跑 MME
#   bash quantization/scripts/run_all_evals.sh all geneval         # 全部，只跑 GenEval
#   bash quantization/scripts/run_all_evals.sh stage19             # 仅 stage19
#   bash quantization/scripts/run_all_evals.sh stage19 both 20260323  # stage19 指定日期
#
# === 可调环境变量 ===
#
#   GPUS_PER_TASK_MME=1        MME 阶段每任务 GPU 数
#   GPUS_PER_TASK_GENEVAL=2    GenEval 阶段每任务 GPU 数
#   TOTAL_GPUS=8               可用 GPU 总数
#   GPU_OFFSET=0               起始 GPU 编号

set -euo pipefail

# ===================== 配置 =====================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
QUANT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

MODEL_PATH="/home/honglianglu/data/InternVL-U/model"
CONFIG_DIR="${QUANT_ROOT}/quantization_outputs/configs"
EVAL_SCRIPT="${PROJECT_ROOT}/eval_internvlu.py"
OUTPUT_BASE="${QUANT_ROOT}/quantization_outputs/eval_results"

# 校准数据（新格式：三类文件，Stage 0 产出）
STAGE0_DIR="${QUANT_ROOT}/quantization_outputs/stage0_full_activation"
GPTQ_HESSIAN_INDEX="${STAGE0_DIR}/gptq_hessian_index_latest.json"
SMOOTHQUANT_STATS="${STAGE0_DIR}/smoothquant_stats_latest.pt"
AWQ_STATS="${STAGE0_DIR}/awq_stats_latest.pt"

# 旧格式兼容（如果新格式不存在则回退）
LEGACY_ACTIVATION_DATA="${STAGE0_DIR}/stage0_full_activation_50samples_latest.pt"

# GPU 参数（可通过环境变量覆盖）
TOTAL_GPUS="${TOTAL_GPUS:-2}"
GPU_OFFSET="${GPU_OFFSET:-4}"
GPUS_PER_TASK_MME="${GPUS_PER_TASK_MME:-1}"
GPUS_PER_TASK_GENEVAL="${GPUS_PER_TASK_GENEVAL:-1}"

# 评测参数
SEED=42
GENEVAL_RESOLUTION=1024
GENEVAL_CFG_SCALE=4.0
GENEVAL_STEPS=20

# ===================== 解析参数 =====================

FILTER="${1:-all}"          # all / w4a4 / w3a4 / baseline / stage19 / stage20 / stage21 / stage23 / stage24 / stage25
BENCH_FILTER="${2:-both}"   # both / mme / geneval
CONFIG_DATE="${3:-}"        # 可选：指定配置日期 (YYYYMMDD)，stage19/20 专用

# ===================== 构造任务列表 =====================

declare -a TASKS=()

add_task() {
    local name="$1" path="$2"
    if [ -n "$path" ] && [ ! -f "$path" ]; then
        echo "[SKIP] $name: config not found ($path)"
        return
    fi
    TASKS+=("${name}:${path}")
}

# Stage19 日期配置自动发现：找到最新的 combined_funcgroup_w*a4_*.json
find_stage19_date() {
    local latest=""
    for f in "${CONFIG_DIR}"/combined_funcgroup_w4a4_*.json; do
        [ -f "$f" ] && latest="$f"
    done
    if [ -n "$latest" ]; then
        local base="$(basename "$latest" .json)"
        echo "${base##*_}"
    fi
}

# Stage20 日期配置自动发现
find_stage20_date() {
    local latest=""
    for f in "${CONFIG_DIR}"/stage20_largecalib_w4a4_*.json; do
        [ -f "$f" ] && latest="$f"
    done
    if [ -n "$latest" ]; then
        local base="$(basename "$latest" .json)"
        echo "${base##*_}"
    fi
}

# Stage21 日期配置自动发现
find_stage21_date() {
    local latest=""
    for f in "${CONFIG_DIR}"/stage21_funcgroup_w4a4_*.json; do
        [ -f "$f" ] && latest="$f"
    done
    if [ -n "$latest" ]; then
        local base="$(basename "$latest" .json)"
        echo "${base##*_}"
    fi
}

# Stage23 日期配置自动发现
find_stage23_date() {
    local latest=""
    for f in "${CONFIG_DIR}"/stage23_modality_weighted_w4a4_*.json; do
        [ -f "$f" ] && latest="$f"
    done
    if [ -n "$latest" ]; then
        local base="$(basename "$latest" .json)"
        echo "${base##*_}"
    fi
}

# Stage24 日期配置自动发现
find_stage24_date() {
    local latest=""
    for f in "${CONFIG_DIR}"/stage24_attn_fidelity_w4a4_*.json; do
        [ -f "$f" ] && latest="$f"
    done
    if [ -n "$latest" ]; then
        local base="$(basename "$latest" .json)"
        echo "${base##*_}"
    fi
}

# Stage25 日期配置自动发现
find_stage25_date() {
    local latest=""
    for f in "${CONFIG_DIR}"/stage25_hard_w4a4_*.json; do
        [ -f "$f" ] && latest="$f"
    done
    if [ -n "$latest" ]; then
        local base="$(basename "$latest" .json)"
        echo "${base##*_}"
    fi
}

if [ "$FILTER" = "baseline" ]; then
    add_task "fp_baseline" ""
elif [ "$FILTER" = "stage21" ]; then
    S21_DATE="${CONFIG_DATE:-$(find_stage21_date)}"
    if [ -z "$S21_DATE" ]; then
        echo "Error: No stage21 config found. Run stage21 search first or pass date as 3rd arg."
        exit 1
    fi
    cfg_file="${CONFIG_DIR}/stage21_funcgroup_w4a4_${S21_DATE}.json"
    add_task "stage21_funcgroup_w4a4_${S21_DATE}" "$cfg_file"
elif [ "$FILTER" = "stage20" ]; then
    S20_DATE="${CONFIG_DATE:-$(find_stage20_date)}"
    if [ -z "$S20_DATE" ]; then
        echo "Error: No stage20 config found. Run stage20 search first or pass date as 3rd arg."
        exit 1
    fi
    cfg_file="${CONFIG_DIR}/stage20_largecalib_w4a4_${S20_DATE}.json"
    add_task "stage20_largecalib_w4a4_${S20_DATE}" "$cfg_file"
elif [ "$FILTER" = "stage23" ]; then
    S23_DATE="${CONFIG_DATE:-$(find_stage23_date)}"
    if [ -z "$S23_DATE" ]; then
        echo "Error: No stage23 config found. Run stage23 search first or pass date as 3rd arg."
        exit 1
    fi
    cfg_file="${CONFIG_DIR}/stage23_modality_weighted_w4a4_${S23_DATE}.json"
    add_task "stage23_modality_weighted_w4a4_${S23_DATE}" "$cfg_file"
elif [ "$FILTER" = "stage24" ]; then
    S24_DATE="${CONFIG_DATE:-$(find_stage24_date)}"
    if [ -z "$S24_DATE" ]; then
        echo "Error: No stage24 config found. Run stage24 search first or pass date as 3rd arg."
        exit 1
    fi
    cfg_file="${CONFIG_DIR}/stage24_attn_fidelity_w4a4_${S24_DATE}.json"
    add_task "stage24_attn_fidelity_w4a4_${S24_DATE}" "$cfg_file"
elif [ "$FILTER" = "stage25" ]; then
    S25_DATE="${CONFIG_DATE:-$(find_stage25_date)}"
    if [ -z "$S25_DATE" ]; then
        echo "Error: No stage25 config found. Run stage25 search first or pass date as 3rd arg."
        exit 1
    fi
    cfg_file="${CONFIG_DIR}/stage25_hard_w4a4_${S25_DATE}.json"
    add_task "stage25_hard_w4a4_${S25_DATE}" "$cfg_file"
    cfg_file="${CONFIG_DIR}/stage25_random_w4a4_${S25_DATE}.json"
    add_task "stage25_random_w4a4_${S25_DATE}" "$cfg_file"
elif [ "$FILTER" = "stage19" ]; then
    # 仅 stage19 配置
    S19_DATE="${CONFIG_DATE:-$(find_stage19_date)}"
    if [ -z "$S19_DATE" ]; then
        echo "Error: No stage19 config date found. Run stage19 search first or pass date as 3rd arg."
        exit 1
    fi
    for wb in 4 3; do
        cfg_file="${CONFIG_DIR}/combined_funcgroup_w${wb}a4_${S19_DATE}.json"
        add_task "stage19_combined_w${wb}a4_${S19_DATE}" "$cfg_file"
    done
else
    # Legacy stages (15/17/18)
    STAGES=("calm_layerwise" "exhaustive_sublayer" "funcgroup")
    STAGE_LABELS=("stage15_calm" "stage17_exhaustive" "stage18_funcgroup")

    WBITS=("4" "3")
    if [ "$FILTER" != "all" ]; then
        WBITS=()
        for w in $FILTER; do
            w="${w#w}"; w="${w%%a*}"
            WBITS+=("$w")
        done
    fi

    for si in "${!STAGES[@]}"; do
        stage="${STAGES[$si]}"
        label="${STAGE_LABELS[$si]}"
        for wb in "${WBITS[@]}"; do
            cfg_file="${CONFIG_DIR}/${stage}_w${wb}a4.json"
            add_task "${label}_w${wb}a4" "$cfg_file"
        done
    done

    # 如果 all，也加上 stage19（如果配置存在）
    if [ "$FILTER" = "all" ]; then
        S19_DATE="${CONFIG_DATE:-$(find_stage19_date)}"
        if [ -n "$S19_DATE" ]; then
            for wb in 4 3; do
                cfg_file="${CONFIG_DIR}/combined_funcgroup_w${wb}a4_${S19_DATE}.json"
                add_task "stage19_combined_w${wb}a4_${S19_DATE}" "$cfg_file"
            done
        fi
        # stage20 (仅 W4A4)
        S20_DATE="${CONFIG_DATE:-$(find_stage20_date)}"
        if [ -n "$S20_DATE" ]; then
            cfg_file="${CONFIG_DIR}/stage20_largecalib_w4a4_${S20_DATE}.json"
            add_task "stage20_largecalib_w4a4_${S20_DATE}" "$cfg_file"
        fi
        # stage21 (仅 W4A4, funcgroup)
        S21_DATE="${CONFIG_DATE:-$(find_stage21_date)}"
        if [ -n "$S21_DATE" ]; then
            cfg_file="${CONFIG_DIR}/stage21_funcgroup_w4a4_${S21_DATE}.json"
            add_task "stage21_funcgroup_w4a4_${S21_DATE}" "$cfg_file"
        fi
        # stage23 (模态加权搜索)
        S23_DATE="${CONFIG_DATE:-$(find_stage23_date)}"
        if [ -n "$S23_DATE" ]; then
            cfg_file="${CONFIG_DIR}/stage23_modality_weighted_w4a4_${S23_DATE}.json"
            add_task "stage23_modality_weighted_w4a4_${S23_DATE}" "$cfg_file"
        fi
        # stage24 (attention fidelity)
        S24_DATE="${CONFIG_DATE:-$(find_stage24_date)}"
        if [ -n "$S24_DATE" ]; then
            cfg_file="${CONFIG_DIR}/stage24_attn_fidelity_w4a4_${S24_DATE}.json"
            add_task "stage24_attn_fidelity_w4a4_${S24_DATE}" "$cfg_file"
        fi
        # stage25 (hard/random 两组)
        S25_DATE="${CONFIG_DATE:-$(find_stage25_date)}"
        if [ -n "$S25_DATE" ]; then
            cfg_file="${CONFIG_DIR}/stage25_hard_w4a4_${S25_DATE}.json"
            add_task "stage25_hard_w4a4_${S25_DATE}" "$cfg_file"
            cfg_file="${CONFIG_DIR}/stage25_random_w4a4_${S25_DATE}.json"
            add_task "stage25_random_w4a4_${S25_DATE}" "$cfg_file"
        fi
    fi
fi

if [ ${#TASKS[@]} -eq 0 ]; then
    echo "No tasks to run. Check config files in $CONFIG_DIR"
    exit 1
fi

# ===================== GPU Slot 构建 =====================

build_gpu_slots() {
    local gpus_per_task=$1
    local -n _out_slots=$2
    _out_slots=()
    local max_slots=$(( TOTAL_GPUS / gpus_per_task ))
    for (( i=0; i < max_slots; i++ )); do
        local start=$(( GPU_OFFSET + i * gpus_per_task ))
        local slot=""
        for (( g=start; g < start + gpus_per_task; g++ )); do
            [ -n "$slot" ] && slot="${slot},"
            slot="${slot}${g}"
        done
        _out_slots+=("$slot")
    done
}

# ===================== 通用并行调度器 =====================

run_phase() {
    local phase_name="$1"
    local benchmark="$2"
    shift 2
    local slots=("$@")

    local n_slots=${#slots[@]}
    local n_tasks=${#TASKS[@]}

    echo ""
    echo "=========================================="
    echo " Phase: ${phase_name}"
    echo "   Benchmark:  ${benchmark}"
    echo "   Tasks:      ${n_tasks}"
    echo "   Parallel:   ${n_slots} slots  [${slots[0]}] ..."
    echo "=========================================="

    # slot -> pid 映射
    declare -A _slot_pid=()
    local submitted=0 completed=0 failed=0

    _get_free_slot() {
        for (( s=0; s < n_slots; s++ )); do
            local pid="${_slot_pid[$s]:-}"
            if [ -z "$pid" ]; then echo "$s"; return 0; fi
            if ! kill -0 "$pid" 2>/dev/null; then
                wait "$pid" 2>/dev/null; local rc=$?
                completed=$((completed + 1))
                [ $rc -ne 0 ] && failed=$((failed + 1))
                unset "_slot_pid[$s]"
                echo "$s"; return 0
            fi
        done
        return 1
    }

    _wait_any_slot() {
        while true; do
            for (( s=0; s < n_slots; s++ )); do
                local pid="${_slot_pid[$s]:-}"
                [ -z "$pid" ] && continue
                if ! kill -0 "$pid" 2>/dev/null; then
                    wait "$pid" 2>/dev/null; local rc=$?
                    completed=$((completed + 1))
                    [ $rc -ne 0 ] && failed=$((failed + 1))
                    unset "_slot_pid[$s]"
                    echo "$s"; return 0
                fi
            done
            sleep 3
        done
    }

    for task in "${TASKS[@]}"; do
        local task_name="${task%%:*}"
        local config_path="${task#*:}"
        local task_output="${OUTPUT_BASE}/${task_name}"
        local log_file="${task_output}/${benchmark}_log.txt"

        mkdir -p "$task_output"

        local slot=""
        slot=$(_get_free_slot 2>/dev/null) || slot=$(_wait_any_slot)
        local gpu_ids="${slots[$slot]}"

        local CMD="CUDA_VISIBLE_DEVICES=${gpu_ids} python -u ${EVAL_SCRIPT} \
            --model_path ${MODEL_PATH} \
            --output_dir ${task_output} \
            --benchmarks ${benchmark} \
            --seed ${SEED} \
            --geneval_resolution ${GENEVAL_RESOLUTION} \
            --geneval_cfg_scale ${GENEVAL_CFG_SCALE} \
            --geneval_steps ${GENEVAL_STEPS}"

        [ -n "$config_path" ] && CMD="${CMD} --quant_config ${config_path}"

        # 校准数据：优先新格式（三文件），兼容旧格式
        if [ -n "$config_path" ]; then
            if [ -f "$GPTQ_HESSIAN_INDEX" ]; then
                CMD="${CMD} --gptq_hessian_index ${GPTQ_HESSIAN_INDEX}"
                CMD="${CMD} --smoothquant_stats ${SMOOTHQUANT_STATS}"
                CMD="${CMD} --awq_stats ${AWQ_STATS}"
            elif [ -f "$LEGACY_ACTIVATION_DATA" ]; then
                CMD="${CMD} --activation_data ${LEGACY_ACTIVATION_DATA}"
            fi
        fi

        (
            echo "[START] ${task_name} / ${benchmark}  (GPU: ${gpu_ids})"
            eval "$CMD" > "$log_file" 2>&1
            local rc=$?
            if [ $rc -eq 0 ]; then
                echo "[DONE]  ${task_name} / ${benchmark}"
            else
                echo "[FAIL]  ${task_name} / ${benchmark} (exit=${rc}) — see ${log_file}"
            fi
            exit $rc
        ) &

        _slot_pid[$slot]=$!
        submitted=$((submitted + 1))
        echo "  [${submitted}/${n_tasks}] ${task_name} -> GPU ${gpu_ids} (PID $!)"
        sleep 1
    done

    # 等待本阶段全部完成
    for (( s=0; s < n_slots; s++ )); do
        local pid="${_slot_pid[$s]:-}"
        [ -z "$pid" ] && continue
        wait "$pid" 2>/dev/null; local rc=$?
        completed=$((completed + 1))
        [ $rc -ne 0 ] && failed=$((failed + 1))
    done

    echo "  ${phase_name} done: ${completed} completed, ${failed} failed"
}

# ===================== Banner =====================

echo ""
echo "=================================================="
echo " InternVL-U Batch Evaluation (Two-Phase Scheduler)"
echo "=================================================="
echo " Tasks:  ${#TASKS[@]}"
for t in "${TASKS[@]}"; do echo "   - ${t%%:*}"; done
echo " Filter: config=${FILTER}  bench=${BENCH_FILTER}"
echo " GPUs:   ${TOTAL_GPUS} (offset ${GPU_OFFSET})"
echo "   MME phase:     ${GPUS_PER_TASK_MME} GPU/task -> $(( TOTAL_GPUS / GPUS_PER_TASK_MME )) parallel"
echo "   GenEval phase:  ${GPUS_PER_TASK_GENEVAL} GPU/task -> $(( TOTAL_GPUS / GPUS_PER_TASK_GENEVAL )) parallel"
echo " Output: ${OUTPUT_BASE}"
echo "=================================================="

mkdir -p "$OUTPUT_BASE"

# ===================== 执行 =====================

if [ "$BENCH_FILTER" = "both" ] || [ "$BENCH_FILTER" = "mme" ]; then
    declare -a MME_SLOTS=()
    build_gpu_slots "$GPUS_PER_TASK_MME" MME_SLOTS
    run_phase "MME (Understanding)" "mme" "${MME_SLOTS[@]}"
fi

if [ "$BENCH_FILTER" = "both" ] || [ "$BENCH_FILTER" = "geneval" ]; then
    declare -a GENEVAL_SLOTS=()
    build_gpu_slots "$GPUS_PER_TASK_GENEVAL" GENEVAL_SLOTS
    run_phase "GenEval (Generation)" "geneval" "${GENEVAL_SLOTS[@]}"
fi

# ===================== 汇总 =====================

echo ""
echo "=================================================="
echo " Evaluation Summary"
echo "=================================================="

printf "   %-30s  %-12s  %s\n" "Config" "MME" "GenEval"
printf "   %-30s  %-12s  %s\n" "------------------------------" "------------" "-------"

for task in "${TASKS[@]}"; do
    task_name="${task%%:*}"
    task_dir="${OUTPUT_BASE}/${task_name}"
    mme_status="--"
    geneval_status="--"

    mme_results="${task_dir}/mme_results/results.txt"
    if [ -f "$mme_results" ]; then
        mme_score=$(grep -oP 'Total score: \K[\d.]+' "$mme_results" 2>/dev/null || echo "done")
        mme_status="$mme_score"
    fi

    geneval_summary="${task_dir}/geneval_results/summary.txt"
    geneval_results="${task_dir}/geneval_results/results.jsonl"
    if [ -f "$geneval_summary" ]; then
        geneval_status="done"
    elif [ -f "$geneval_results" ]; then
        geneval_status="partial"
    fi

    printf "   %-30s  %-12s  %s\n" "$task_name" "$mme_status" "$geneval_status"
done

echo ""
echo " Output: ${OUTPUT_BASE}"
echo "=================================================="
