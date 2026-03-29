#!/bin/bash

# InternVL-U Evaluation: MME + GenEval
# Usage: bash eval.sh

# ===================== 配置 =====================

MODEL_PATH="/data/14thdd/users/honglianglu/InternVL-U/model"
OUTPUT_DIR="./eval_outputs"
GPU_IDS="2,3"
BENCHMARKS="geneval"
SEED=42

# GenEval 参数
GENEVAL_RESOLUTION=1024
GENEVAL_CFG_SCALE=4.0
GENEVAL_STEPS=20

# ===================== 下载模型（首次运行） =====================

if [ ! -d "$MODEL_PATH" ]; then
    echo "Model not found at $MODEL_PATH, downloading..."
    echo ""
    echo "Option 1: huggingface-cli (recommended)"
    echo "  huggingface-cli download InternVL-U/InternVL-U --local-dir $MODEL_PATH"
    echo ""
    echo "Option 2: If HuggingFace is slow, use mirror"
    echo "  HF_ENDPOINT=https://hf-mirror.com huggingface-cli download InternVL-U/InternVL-U --local-dir $MODEL_PATH"
    echo ""

    # Uncomment one of the following to auto-download:
    # huggingface-cli download InternVL-U/InternVL-U --local-dir "$MODEL_PATH"
    HF_ENDPOINT=https://hf-mirror.com huggingface-cli download InternVL-U/InternVL-U --local-dir "$MODEL_PATH"

    if [ $? -ne 0 ]; then
        echo "Download failed. Please download manually and set MODEL_PATH."
        exit 1
    fi
fi

# ===================== 运行评测 =====================

export CUDA_VISIBLE_DEVICES="$GPU_IDS"

echo "=================================================="
echo "InternVL-U Evaluation"
echo "=================================================="
echo "Model:      $MODEL_PATH"
echo "Output:     $OUTPUT_DIR"
echo "GPU IDs:    $GPU_IDS (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
echo "Benchmarks: $BENCHMARKS"
echo "=================================================="

python eval_internvlu.py \
    --model_path "$MODEL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --benchmarks $BENCHMARKS \
    --seed $SEED \
    --geneval_resolution $GENEVAL_RESOLUTION \
    --geneval_cfg_scale $GENEVAL_CFG_SCALE \
    --geneval_steps $GENEVAL_STEPS

echo ""
echo "=================================================="
echo "Evaluation complete!"
echo "=================================================="
echo "Results: $OUTPUT_DIR"
