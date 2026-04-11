"""
InternVL-U Evaluation Script: MME + GenEval

Mirrors the evaluation pipeline from Bagel's stage3_test.py,
adapted for InternVL-U's InternVLUPipeline interface.

Usage:
    # Run both MME and GenEval
    python eval_internvlu.py --model_path /path/to/InternVL-U \
        --benchmarks mme geneval --gpu_ids 0,1,4,5

    # Run only MME
    python eval_internvlu.py --model_path /path/to/InternVL-U \
        --benchmarks mme

    # Run only GenEval
    python eval_internvlu.py --model_path /path/to/InternVL-U \
        --benchmarks geneval --geneval_resolution 1024
"""

import os
import sys
import re
import json
import random
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

BAGEL_DIR = Path("/data/user/honglianglu/Bagel")
SCRIPT_DIR = Path(__file__).resolve().parent

# GenEval 目标检测用的 mmdetection config（mask2former）
# 可通过环境变量 MMDETECTION_DIR 指定 mmdetection 源码根目录，否则默认 BAGEL_DIR 同级的 mmdetection
GENEVAL_MMDET_CONFIG_NAME = "mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py"


def _geneval_mmdet_config_path() -> Path:
    base = os.environ.get("MMDETECTION_DIR")
    if base:
        return Path(base) / "configs" / "mask2former" / GENEVAL_MMDET_CONFIG_NAME
    return BAGEL_DIR.parent / "mmdetection" / "configs" / "mask2former" / GENEVAL_MMDET_CONFIG_NAME


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def post_processing(response: str) -> str:
    """Post-process MME responses (same logic as Bagel)."""
    response = response.replace('\n', '')
    response = response.replace('不是', 'No').replace('是', 'Yes').replace('否', 'No')
    response = response.lower().replace('true', 'yes').replace('false', 'no')
    response = re.sub(r'[\u4e00-\u9fa5]', '', response)
    return response


def load_model(model_path: str, gpu_ids: str = None):
    """Load InternVL-U pipeline."""
    if gpu_ids and 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids

    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
    sys.path.insert(0, str(SCRIPT_DIR))
    from internvlu import InternVLUPipeline

    print(f"Loading InternVL-U from {model_path} ...")
    pipeline = InternVLUPipeline.from_pretrained(
        model_path, torch_dtype=torch.bfloat16,
    )
    pipeline.to("cuda")
    print(f"Model loaded on {pipeline.vlm.device}")
    return pipeline


# ============================================================
# Quantization Config Application
# ============================================================

def apply_quant_config(
    pipeline,
    quant_config_file: str,
    activation_data_file: str = None,
    gptq_hessian_index: str = None,
    smoothquant_stats: str = None,
    awq_stats: str = None,
):
    """Apply a quantization config JSON to the pipeline's language_model.

    支持两种校准数据格式：
      1. 旧格式: 单个 .pt 文件 (activation_data_file)
      2. 新格式: gptq_hessian_index + smoothquant_stats + awq_stats (三文件)
    新格式优先。两者都没有时，GPTQ/AWQ/SmoothQuant 会缺数据。
    """
    import gc
    import torch.nn as nn
    from collections import OrderedDict

    quant_root = SCRIPT_DIR / "quantization"
    if str(quant_root) not in sys.path:
        sys.path.insert(0, str(quant_root))
    from layers.hybrid_quant_linear import HybridQuantLinear

    print(f"\nApplying quantization config: {quant_config_file}")
    with open(quant_config_file) as f:
        layer_configs = json.load(f)
    print(f"  {len(layer_configs)} sublayer configs loaded")

    # ---- 加载校准数据 ----
    hessian_index = None
    smooth_data = None
    awq_data = None
    legacy_data = None
    use_new_format = False

    if gptq_hessian_index and Path(gptq_hessian_index).exists():
        with open(gptq_hessian_index, "r") as f:
            idx = json.load(f)
        hessian_index = idx.get("layers", {})
        use_new_format = True
        print(f"  [GPTQ] Hessian index: {len(hessian_index)} layers")

    if smoothquant_stats and Path(smoothquant_stats).exists():
        smooth_data = torch.load(smoothquant_stats, map_location="cpu")
        use_new_format = True
        print(f"  [SmoothQuant] Stats: {len(smooth_data)} layers")

    if awq_stats and Path(awq_stats).exists():
        awq_data = torch.load(awq_stats, map_location="cpu")
        use_new_format = True
        print(f"  [AWQ] Stats: {len(awq_data)} layers")

    if not use_new_format and activation_data_file and Path(activation_data_file).exists():
        legacy_data = torch.load(activation_data_file, map_location="cpu")
        print(f"  [Legacy] Activation data: {len(legacy_data)} layers")

    if not use_new_format and legacy_data is None:
        print("  [WARN] No calibration data found — GPTQ/AWQ/SmoothQuant may degrade")

    # ---- Hessian 重建辅助函数 ----
    _hessian_cache = OrderedDict()
    _HESSIAN_CACHE_SIZE = 10
    _DAMP_RATIO = 0.01

    def _reconstruct_activation(layer_name: str):
        """从 Hessian 通过 Cholesky 分解重建等价 activation 矩阵。"""
        if layer_name in _hessian_cache:
            _hessian_cache.move_to_end(layer_name)
            return _hessian_cache[layer_name]

        if hessian_index is None or layer_name not in hessian_index:
            return None

        info = hessian_index[layer_name]
        pt_path = info["path"]
        if not Path(pt_path).exists():
            return None
        data = torch.load(pt_path, map_location="cpu", weights_only=True)
        H_sum = data["hessian_sum"].double()
        D = H_sum.shape[0]
        damp = _DAMP_RATIO * H_sum.diagonal().mean()
        H_reg = H_sum + damp * torch.eye(D, dtype=H_sum.dtype)

        try:
            L = torch.linalg.cholesky(H_reg)
            X_equiv = L.t().float()
        except torch.linalg.LinAlgError:
            eigvals, eigvecs = torch.linalg.eigh(H_reg)
            eigvals = eigvals.clamp(min=0)
            X_equiv = (eigvecs * eigvals.sqrt().unsqueeze(0)).t().float()

        _hessian_cache[layer_name] = X_equiv
        if len(_hessian_cache) > _HESSIAN_CACHE_SIZE:
            _hessian_cache.popitem(last=False)
        return X_equiv

    def _get_channel_max(layer_name: str):
        """从 SmoothQuant/AWQ stats 获取 activation channel max。"""
        if smooth_data and layer_name in smooth_data:
            return smooth_data[layer_name].get("act_channel_max")
        if awq_data and layer_name in awq_data:
            return awq_data[layer_name].get("channel_max")
        return None

    # ---- 逐层量化 ----
    model = pipeline.vlm
    replaced = 0
    failed = 0

    for layer_name, cfg in layer_configs.items():
        parts = layer_name.split(".")
        parent = model
        try:
            for p in parts[:-1]:
                parent = getattr(parent, p)
            child_name = parts[-1]
            layer_module = getattr(parent, child_name)
        except AttributeError:
            failed += 1
            continue

        if not isinstance(layer_module, nn.Linear) or isinstance(layer_module, HybridQuantLinear):
            continue

        original_device = layer_module.weight.device
        has_bias = layer_module.bias is not None

        quant_layer = HybridQuantLinear(
            in_features=layer_module.in_features,
            out_features=layer_module.out_features,
            bias=has_bias,
            weight_bit=cfg.get("weight_bit", 8),
            act_bit=cfg.get("act_bit", 8),
            quant_percentile=cfg.get("quant_percentile", 0.999),
            act_unsigned=cfg.get("act_unsigned", False),
            use_sparse=cfg.get("use_sparse", False),
            use_smoothquant=cfg.get("use_smoothquant", False),
            use_svd=cfg.get("use_svd", False),
            use_block_quant=cfg.get("use_block_quant", False),
            use_block_quant_act=cfg.get("use_block_quant_act", False),
            sparse_ratio=cfg.get("sparse_ratio", 0.0),
            smoothquant_alpha=cfg.get("smoothquant_alpha", 0.5),
            svd_rank=cfg.get("svd_rank", 0),
            block_size_weight=cfg.get("block_size_weight", 256),
            block_size_act=cfg.get("block_size_act", 256),
            use_gptq=cfg.get("use_gptq", False),
            gptq_group_size=cfg.get("gptq_group_size", 64),
            gptq_damp_percentage=cfg.get("gptq_damp_percentage", 0.01),
            gptq_block_size=cfg.get("gptq_block_size", 128),
            gptq_num_inv_tries=cfg.get("gptq_num_inv_tries", 250),
            gptq_hessian_block_size=cfg.get("gptq_hessian_block_size", 512),
            use_awq=cfg.get("use_awq", False),
            awq_alpha=cfg.get("awq_alpha", 0.5),
            awq_n_grid=cfg.get("awq_n_grid", 20),
        )

        try:
            quant_layer.weight.data.copy_(layer_module.weight.data)
        except RuntimeError:
            quant_layer.weight.data = layer_module.weight.data
        if has_bias:
            try:
                quant_layer.bias.data.copy_(layer_module.bias.data)
            except RuntimeError:
                quant_layer.bias.data = layer_module.bias.data

        quant_layer = quant_layer.to(original_device)

        # 获取该层的校准数据
        act_data = None
        act_max = None

        if use_new_format:
            needs_gptq = cfg.get("use_gptq", False)
            needs_awq = cfg.get("use_awq", False)
            needs_smooth = cfg.get("use_smoothquant", False)

            if needs_gptq or needs_awq:
                act_data = _reconstruct_activation(layer_name)
            if needs_smooth:
                act_max = _get_channel_max(layer_name)
                if act_max is not None:
                    act_max = act_max.to(original_device, dtype=quant_layer.weight.dtype)
        elif legacy_data is not None and layer_name in legacy_data:
            act_data = legacy_data[layer_name]

        try:
            quant_layer.prepare_weight(
                activation_data=act_data,
                activation_max=act_max,
                layer_name=layer_name,
                verbose=False,
            )
        except Exception as e:
            print(f"  [WARN] prepare_weight failed for {layer_name}: {e}")

        setattr(parent, child_name, quant_layer)
        replaced += 1

        del layer_module
        if replaced % 50 == 0:
            _hessian_cache.clear()
            gc.collect()
            torch.cuda.empty_cache()

    _hessian_cache.clear()
    print(f"  Quantized {replaced}/{len(layer_configs)} layers"
          + (f" ({failed} not found)" if failed else ""))
    gc.collect()
    torch.cuda.empty_cache()


# ============================================================
# MME Evaluation
# ============================================================

def run_mme(pipeline, output_dir: Path, mme_data_root: Path, mme_questions_root: Path):
    """Run MME evaluation using InternVL-U."""
    print("\n" + "=" * 60)
    print("MME Evaluation")
    print("=" * 60)

    if not mme_data_root.exists():
        print(f"MME data not found at {mme_data_root}")
        return None
    if not mme_questions_root.exists():
        print(f"MME questions not found at {mme_questions_root}")
        return None

    mme_output_dir = output_dir / "mme_results"
    mme_output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = pipeline.processor.tokenizer
    prompt_suffix = "Answer the question using a single word or phrase."

    categories_to_process = []
    categories_completed = []

    for question_file in sorted(mme_questions_root.iterdir()):
        if not question_file.name.endswith('.txt'):
            continue
        category = question_file.stem
        output_file = mme_output_dir / question_file.name

        if output_file.exists():
            with open(question_file, 'r') as f:
                num_questions = len(f.readlines())
            with open(output_file, 'r') as f:
                num_answers = len(f.readlines())
            if num_questions == num_answers:
                categories_completed.append(category)
                print(f"  [skip] {category} ({num_answers}/{num_questions})")
                continue
            else:
                print(f"  [redo] {category} ({num_answers}/{num_questions})")
                output_file.unlink()
        categories_to_process.append(question_file)

    if not categories_to_process:
        print(f"\nAll {len(categories_completed)} MME categories complete, skipping.")
    else:
        print(f"\n{len(categories_completed)} complete, {len(categories_to_process)} to process")

    for question_file in categories_to_process:
        category = question_file.stem
        output_file = mme_output_dir / question_file.name

        with open(question_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        print(f"\nProcessing {category} ({len(lines)} questions)...")
        with open(output_file, 'w', encoding='utf-8') as fout:
            for line in tqdm(lines, desc=f"  {category}"):
                img_name, question, gt = line.strip().split('\t')
                question_with_prompt = question + ' ' + prompt_suffix

                img_path = mme_data_root / category / img_name
                if not img_path.exists():
                    img_path = mme_data_root / category / "images" / img_name

                image = Image.open(img_path).convert('RGB')

                with torch.no_grad():
                    output = pipeline(
                        prompt=question_with_prompt,
                        image=image,
                        generation_mode="text",
                        max_new_tokens=20,
                    ).generate_output[0]

                response = tokenizer.decode(output, skip_special_tokens=True)
                response = post_processing(response)

                print(img_name, question, gt, response, sep='\t', file=fout)

    print(f"\nMME predictions saved to: {mme_output_dir}")

    # Calculate scores
    calculation_file = mme_output_dir / "calculation.txt"
    results_file = mme_output_dir / "results.txt"
    if results_file.exists():
        print(f"MME scores already exist: {results_file}")
    else:
        print("Calculating MME scores...")
        subprocess.run([
            sys.executable, "-m", "eval.vlm.eval.mme.calculation",
            "--out-dir", str(mme_output_dir.resolve())
        ], cwd=str(BAGEL_DIR), check=True)
        print("MME scores calculated.")

    if results_file.exists():
        print("\n" + results_file.read_text())

    return {"output_dir": str(mme_output_dir)}


# ============================================================
# GenEval Evaluation
# ============================================================

def run_geneval(pipeline, output_dir: Path, resolution: int = 1024,
                cfg_scale: float = 4.5, num_inference_steps: int = 20,
                num_images_per_prompt: int = 1, seed: int = 42):
    """Run GenEval evaluation using InternVL-U."""
    print("\n" + "=" * 60)
    print("GenEval Evaluation")
    print("=" * 60)

    prompts_file = BAGEL_DIR / "eval" / "gen" / "geneval" / "prompts" / "evaluation_metadata.jsonl"
    geneval_model_path = BAGEL_DIR / "eval" / "gen" / "geneval" / "model"

    if not prompts_file.exists():
        print(f"GenEval prompts not found: {prompts_file}")
        return None

    geneval_output_dir = output_dir / "geneval_results"
    geneval_output_dir.mkdir(parents=True, exist_ok=True)

    with open(prompts_file, 'r') as f:
        metadatas = [json.loads(line) for line in f]

    total_prompts = len(metadatas)
    print(f"Total prompts: {total_prompts}")
    print(f"Resolution: {resolution}x{resolution}")
    print(f"CFG scale: {cfg_scale}, Steps: {num_inference_steps}")

    # Check completeness
    prompts_to_process = []
    for idx in range(total_prompts):
        sample_path = geneval_output_dir / f"{idx:05d}" / "samples"
        all_exist = all(
            (sample_path / f"{i:05d}.png").exists()
            for i in range(num_images_per_prompt)
        )
        if not all_exist:
            prompts_to_process.append(idx)

    print(f"{total_prompts - len(prompts_to_process)} complete, {len(prompts_to_process)} to generate")

    for idx in prompts_to_process:
        metadata = metadatas[idx]
        prompt = metadata['prompt']

        outpath = geneval_output_dir / f"{idx:05d}"
        outpath.mkdir(parents=True, exist_ok=True)
        sample_path = outpath / "samples"
        sample_path.mkdir(parents=True, exist_ok=True)

        with open(outpath / "metadata.jsonl", "w") as f:
            json.dump(metadata, f)

        generator = torch.Generator(device="cuda").manual_seed(seed + idx)

        with torch.no_grad():
            result = pipeline(
                prompt=prompt,
                generation_mode="image",
                height=resolution,
                width=resolution,
                all_cfg_scale=cfg_scale,
                num_inference_steps=num_inference_steps,
                num_images_per_prompt=num_images_per_prompt,
                generator=generator,
            )

        for img_idx, image in enumerate(result.images):
            bbox = image.getbbox()
            if bbox:
                image = image.crop(bbox)
            image.save(sample_path / f"{img_idx:05d}.png")

        print(f"  [{idx+1}/{total_prompts}] '{prompt[:60]}...'")

    print(f"\nGenEval images saved to: {geneval_output_dir}")

    # Phase 2: Evaluate images
    results_file = geneval_output_dir / "results.jsonl"
    if results_file.exists():
        print(f"GenEval results already exist: {results_file}")
    else:
        print("Running GenEval evaluation (object detection)...")
        eval_script = BAGEL_DIR / "eval" / "gen" / "geneval" / "evaluation" / "evaluate_images_mp.py"
        mmdet_config = _geneval_mmdet_config_path()
        if eval_script.exists() and geneval_model_path.exists():
            if not mmdet_config.exists():
                print(f"[WARN] mmdetection config not found: {mmdet_config}, set MMDETECTION_DIR or place mmdetection next to Bagel.")
            nproc = min(2, torch.cuda.device_count())
            torchrun_bin = str(Path(sys.executable).parent / "torchrun")
            if not Path(torchrun_bin).exists():
                torchrun_bin = "torchrun"
            master_port = int(os.environ.get("GENEVAL_MASTER_PORT", "29502"))
            cmd = [
                torchrun_bin,
                "--nnodes=1", "--node_rank=0",
                f"--nproc_per_node={nproc}",
                "--master_addr=127.0.0.1",
                f"--master_port={master_port}",
                str(eval_script),
                str(geneval_output_dir.resolve()),
                "--outfile", str(results_file.resolve()),
                "--model-path", str(geneval_model_path.resolve()),
            ]
            if mmdet_config.exists():
                cmd.extend(["--model-config", str(mmdet_config.resolve())])
            subprocess.run(cmd, check=True)
            print("GenEval evaluation completed.")
        else:
            print(f"Evaluation script or model not found, skipping.")

    # Phase 3: Summary
    summary_file = geneval_output_dir / "summary.txt"
    if summary_file.exists():
        print(f"GenEval summary already exists: {summary_file}")
    else:
        summary_script = BAGEL_DIR / "eval" / "gen" / "geneval" / "evaluation" / "summary_scores.py"
        if results_file.exists() and summary_script.exists():
            print("Generating GenEval summary...")
            subprocess.run([
                sys.executable, str(summary_script),
                str(results_file.resolve()),
            ], check=True)
            print("GenEval summary generated.")

    if summary_file.exists():
        print("\n" + summary_file.read_text())

    return {"output_dir": str(geneval_output_dir), "total_prompts": total_prompts}


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="InternVL-U Evaluation: MME + GenEval")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to InternVL-U checkpoint")
    parser.add_argument("--output_dir", type=str, default="./eval_outputs",
                        help="Output directory")
    parser.add_argument("--gpu_ids", type=str, default=None,
                        help="GPU IDs (e.g. '0,1,4,5')")
    parser.add_argument("--benchmarks", nargs="+", default=["mme", "geneval"],
                        choices=["mme", "geneval"],
                        help="Benchmarks to run")
    parser.add_argument("--seed", type=int, default=42)

    # GenEval params
    parser.add_argument("--geneval_resolution", type=int, default=1024)
    parser.add_argument("--geneval_cfg_scale", type=float, default=4.5)
    parser.add_argument("--geneval_steps", type=int, default=20)
    parser.add_argument("--geneval_num_images", type=int, default=1)

    # Quantization config
    parser.add_argument("--quant_config", type=str, default=None,
                        help="Path to quantization config JSON (from stage15/17/18/19/20)")
    parser.add_argument("--activation_data", type=str, default=None,
                        help="Path to legacy activation data .pt file (旧格式兼容)")
    parser.add_argument("--gptq_hessian_index", type=str, default=None,
                        help="Path to GPTQ Hessian index JSON (新格式, from Stage 0)")
    parser.add_argument("--smoothquant_stats", type=str, default=None,
                        help="Path to SmoothQuant stats .pt (新格式, from Stage 0)")
    parser.add_argument("--awq_stats", type=str, default=None,
                        help="Path to AWQ stats .pt (新格式, from Stage 0)")

    # MME data paths (defaults to Bagel project paths)
    parser.add_argument("--mme_data_root", type=str,
                        default=str(BAGEL_DIR / "data" / "mme" / "MME_Benchmark_release_version"))
    parser.add_argument("--mme_questions_root", type=str,
                        default=str(BAGEL_DIR / "eval" / "vlm" / "eval" / "mme" / "Your_Results"))

    args = parser.parse_args()

    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("InternVL-U Evaluation")
    print("=" * 60)
    print(f"  Model: {args.model_path}")
    if args.quant_config:
        print(f"  Quant config: {args.quant_config}")
    print(f"  Output: {output_dir}")
    print(f"  Benchmarks: {args.benchmarks}")
    print(f"  Seed: {args.seed}")
    print("=" * 60)

    pipeline = load_model(args.model_path, args.gpu_ids)

    if args.quant_config:
        apply_quant_config(
            pipeline, args.quant_config,
            activation_data_file=args.activation_data,
            gptq_hessian_index=args.gptq_hessian_index,
            smoothquant_stats=args.smoothquant_stats,
            awq_stats=args.awq_stats,
        )

    results = {}

    if "mme" in args.benchmarks:
        results["mme"] = run_mme(
            pipeline, output_dir,
            mme_data_root=Path(args.mme_data_root),
            mme_questions_root=Path(args.mme_questions_root),
        )

    if "geneval" in args.benchmarks:
        results["geneval"] = run_geneval(
            pipeline, output_dir,
            resolution=args.geneval_resolution,
            cfg_scale=args.geneval_cfg_scale,
            num_inference_steps=args.geneval_steps,
            num_images_per_prompt=args.geneval_num_images,
            seed=args.seed,
        )

    # Save report
    report = {
        "model_path": args.model_path,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "benchmarks": args.benchmarks,
        "results": results,
    }
    report_file = output_dir / "evaluation_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Evaluation complete! Report: {report_file}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
