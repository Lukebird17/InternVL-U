CUDA_VISIBLE_DEVICES=0,1 python eval_internvlu.py \
    --model_path /data/14thdd/users/honglianglu/InternVL-U/model \
    --quant_config quantization/quantization_outputs/configs/smoothquant_uniform_w3a4.json \
    --activation_data quantization/quantization_outputs/stage0_full_activation/stage0_full_activation_50samples_latest.pt \
    --output_dir quantization/quantization_outputs/eval_results/svdquant_uniform_w3a4 \
    --benchmarks mme geneval

CUDA_VISIBLE_DEVICES=2,4 python eval_internvlu.py \
    --model_path /data/14thdd/users/honglianglu/InternVL-U/model \
    --quant_config quantization/quantization_outputs/configs/gptq_uniform_w4a4.json \
    --activation_data quantization/quantization_outputs/stage0_full_activation/stage0_full_activation_50samples_latest.pt \
    --output_dir quantization/quantization_outputs/eval_results/gptq_uniform_w4a4 \
    --benchmarks mme geneval

CUDA_VISIBLE_DEVICES=0,1 python eval_internvlu.py \
    --model_path /data/14thdd/users/honglianglu/InternVL-U/model \
    --quant_config quantization/quantization_outputs/configs/smoothquant_uniform_w3a4.json \
    --output_dir quantization/quantization_outputs/eval_results/smoothquant_uniform_w3a4 \
    --benchmarks mme geneval

# AWQ
CUDA_VISIBLE_DEVICES=1 python eval_internvlu.py \
    --model_path /data/14thdd/users/honglianglu/InternVL-U/model \
    --quant_config quantization/quantization_outputs/configs/awq_uniform_w3a4.json \
    --activation_data quantization/quantization_outputs/stage0_full_activation/stage0_full_activation_50samples_latest.pt \
    --output_dir quantization/quantization_outputs/eval_results/awq_uniform_w3a4 \
    --benchmarks mme geneval