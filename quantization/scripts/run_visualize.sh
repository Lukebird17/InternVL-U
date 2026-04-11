#!/bin/bash

# 可视化脚本 — 为 Stage 22/23/24/25 的 JSON 结果生成图表
#
# 用法:
#   bash scripts/run_visualize.sh 22          # 只可视化 Stage 22
#   bash scripts/run_visualize.sh 23          # 只可视化 Stage 23
#   bash scripts/run_visualize.sh all         # 全部可视化 + 跨 stage 对比
#   bash scripts/run_visualize.sh cross       # 只做跨 stage 配置对比
#
# 可选环境变量:
#   RESULTS_DIR=xxx bash scripts/run_visualize.sh 22   # 指定结果目录

QUANT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-${QUANT_ROOT}/quantization_outputs}"

STAGE="${1:-all}"

echo "=================================================="
echo "Visualization: Stage ${STAGE}"
echo "=================================================="
echo "Output root: ${OUTPUT_ROOT}"
echo ""

CMD="python -u ${QUANT_ROOT}/utils/visualize_results.py --stage ${STAGE} --root ${OUTPUT_ROOT}"

if [ -n "${RESULTS_DIR:-}" ]; then
    CMD="${CMD} --results_dir ${RESULTS_DIR}"
fi

eval $CMD
