# 根目录（与 scripts/run_all_evals.sh 一致时）
QUANT_ROOT="/home/honglianglu/data/InternVL-U/quantization"
PROJECT_ROOT="/home/honglianglu/data/InternVL-U"
RUN_DATE="20260405"

RANDOM_STATS="${QUANT_ROOT}/quantization_outputs/stage25_hard_sample/activation_stats_random_${RUN_DATE}"
HARD_STATS="${QUANT_ROOT}/quantization_outputs/stage25_hard_sample/activation_stats_hard_${RUN_DATE}"

CFG_RANDOM="${QUANT_ROOT}/quantization_outputs/configs/stage25_random_w4a4_${RUN_DATE}.json"
CFG_HARD="${QUANT_ROOT}/quantization_outputs/configs/stage25_hard_w4a4_${RUN_DATE}.json"

# Random 配置 + Random 子集统计
CUDA_VISIBLE_DEVICES=7 python -u "${PROJECT_ROOT}/eval_internvlu.py" \
  --model_path "/home/honglianglu/data/InternVL-U/model" \
  --quant_config "$CFG_RANDOM" \
  --gptq_hessian_index "${RANDOM_STATS}/gptq_hessian_index.json" \
  --smoothquant_stats "${RANDOM_STATS}/smoothquant_stats.pt" \
  --awq_stats "${RANDOM_STATS}/awq_stats.pt" \
  --output_dir "${QUANT_ROOT}/quantization_outputs/eval_results/stage25_random_w4a4_${RUN_DATE}_subsetcalib" \
  --benchmarks mme geneval

# Hard 配置 + Hard 子集统计
CUDA_VISIBLE_DEVICES=7 python -u "${PROJECT_ROOT}/eval_internvlu.py" \
  --model_path "/home/honglianglu/data/InternVL-U/model" \
  --quant_config "$CFG_HARD" \
  --gptq_hessian_index "${HARD_STATS}/gptq_hessian_index.json" \
  --smoothquant_stats "${HARD_STATS}/smoothquant_stats.pt" \
  --awq_stats "${HARD_STATS}/awq_stats.pt" \
  --output_dir "${QUANT_ROOT}/quantization_outputs/eval_results/stage25_hard_w4a4_${RUN_DATE}_subsetcalib" \
  --benchmarks mme geneval