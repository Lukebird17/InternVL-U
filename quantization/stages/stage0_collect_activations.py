"""
Stage 0 (InternVL-U): 收集完整激活值用于 GPTQ / AWQ

InternVL-U 是统一 VLM，支持理解 (und) 和生成 (gen) 两种任务。
两种任务共享同一个 language_model（Qwen3 backbone），
但激活分布不同：

  理解任务:
    - 输入序列包含大量 vision token（经 vision_model + mlp1 投影而来）
    - 激活分布受图像内容影响
    - vision_model 被激活

  生成任务:
    - 输入序列是纯文本 + 特殊 token（<img>, <IMG_CONTEXT> 等）
    - 激活分布更像纯语言模型
    - 需要 generation_decoder + VAE（但我们只关注 language_model）
    - generate_hidden_states() 做一步前向传播

为了让量化后的模型在两种任务上都保持精度，
校准数据集应该混合两类样本。

默认策略：50% 理解 + 50% 生成，可通过命令行调节。
"""

import os
import sys
import gc
import json
import shutil
import functools
import argparse
import yaml
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Optional, List, Any
from datetime import datetime
from tqdm import tqdm
from PIL import Image

_QUANT_ROOT = str(Path(__file__).resolve().parent.parent)
if _QUANT_ROOT not in sys.path:
    sys.path.insert(0, _QUANT_ROOT)

from utils.model_loader import load_internvlu
from utils.calibration import CalibrationDataLoader, image_for_processor


# ============================================================
# Default prompts
# ============================================================

DEFAULT_UND_PROMPTS = [
    "What is the capital of France? Answer the question using a single word or phrase.",
    "Describe the main objects and their colors in this image.",
    "Is the sky blue? Answer the question using a single word or phrase.",
    "What color is grass? Answer the question using a single word or phrase.",
    "How many legs does a cat have? Answer the question using a single word or phrase.",
    "What is 2 + 2? Answer the question using a single word or phrase.",
    "Is water wet? Answer the question using a single word or phrase.",
    "What is the largest planet in our solar system? Answer the question using a single word or phrase.",
    "Is the Earth flat? Answer the question using a single word or phrase.",
    "What language is spoken in Japan? Answer the question using a single word or phrase.",
    "How many days are in a week? Answer the question using a single word or phrase.",
    "What is the boiling point of water in Celsius? Answer the question using a single word or phrase.",
    "Is the sun a star? Answer the question using a single word or phrase.",
    "What is the chemical symbol for gold? Answer the question using a single word or phrase.",
    "How many continents are there? Answer the question using a single word or phrase.",
    "What is the speed of light approximately? Answer the question using a single word or phrase.",
]

DEFAULT_GEN_PROMPTS = [
    "A beautiful sunset over the ocean with orange and purple clouds.",
    "A futuristic city with flying cars and neon lights at night.",
    "A cozy cabin in the snowy mountains during winter.",
    "A serene Japanese garden with cherry blossoms and a koi pond.",
    "A steampunk robot reading a book in a library.",
    "A underwater coral reef teeming with colorful tropical fish.",
    "A fantasy castle floating among clouds at golden hour.",
    "An astronaut riding a horse on the surface of Mars.",
    "A photorealistic portrait of a cat wearing a top hat and monocle.",
    "A dense enchanted forest with magical glowing mushrooms at twilight.",
    "A bustling medieval marketplace with merchants and colorful banners.",
    "A minimalist modern living room with floor-to-ceiling windows overlooking a lake.",
    "A dragon flying over a volcanic landscape at sunset.",
    "A field of lavender stretching to the horizon under a starry sky.",
    "A cyberpunk alley with holographic advertisements and rain reflections.",
    "A vintage biplane flying above a patchwork of farmland.",
]


# ============================================================
# Helpers
# ============================================================

def _symlink(target: Path, link: Path):
    """创建或更新符号链接。"""
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to(target.name)


# ============================================================
# Activation Collector
# ============================================================

class FullActivationCollector:
    """收集 InternVL-U language_model 中所有 Linear 层的激活统计量。

    核心策略：在线积累，不存原始 activation。
      - Hessian (X^T @ X): GPTQ 的全部需求，(D, D) per layer
      - channel_max, channel_mean: AWQ / SmoothQuant 的全部需求
      - nsamples: token 计数

    所有 1000 条校准样本的所有 token 都完整参与统计，
    内存占用恒定（~48 GB for Hessian matrices on CPU）。
    """

    def __init__(
        self,
        model: nn.Module,
        output_dir: str,
    ):
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 在线积累的 Hessian: H_sum[name] = X^T @ X (float64 避免数值溢出)
        self.hessian_sum: Dict[str, torch.Tensor] = {}
        self.hessian_nsamples: Dict[str, int] = {}

        # channel statistics（在线积累，不占大内存）
        self.act_channel_max: Dict[str, torch.Tensor] = {}
        self.act_channel_mean_sum: Dict[str, torch.Tensor] = {}
        self.act_layer_max: Dict[str, float] = {}

        self.weight_stats: Dict[str, Dict] = {}
        self.hooks: List = []
        self.token_count: Dict[str, int] = {}

    # ---------- hooks ----------

    def _accumulate_hessian(self, name: str, tensor: torch.Tensor):
        """在线积累 Hessian H_sum += X^T @ X (GPU matmul → CPU 积累)。"""
        with torch.no_grad():
            hidden_dim = tensor.shape[-1]
            x = tensor.view(-1, hidden_dim).float()  # (tokens, D) on GPU
            n_tokens = x.shape[0]

            # GPU matmul: (D, tokens) @ (tokens, D) → (D, D)
            xtx = (x.t() @ x).double().cpu()

            if name not in self.hessian_sum:
                self.hessian_sum[name] = xtx
                self.hessian_nsamples[name] = n_tokens
            else:
                self.hessian_sum[name].add_(xtx)
                self.hessian_nsamples[name] += n_tokens
            del xtx

    def _accumulate_channel_stats(self, name: str, tensor: torch.Tensor):
        """在线积累 channel_max 和 channel_mean_sum。"""
        with torch.no_grad():
            hidden_dim = tensor.shape[-1]
            t_flat = tensor.view(-1, hidden_dim).abs()
            n_tokens = t_flat.shape[0]

            layer_max = t_flat.max().float().cpu().item()
            if name in self.act_layer_max:
                self.act_layer_max[name] = max(self.act_layer_max[name], layer_max)
            else:
                self.act_layer_max[name] = layer_max

            ch_max = t_flat.max(dim=0)[0].float().cpu()
            if name in self.act_channel_max:
                self.act_channel_max[name] = torch.maximum(
                    self.act_channel_max[name], ch_max
                )
            else:
                self.act_channel_max[name] = ch_max

            ch_mean_sum = t_flat.sum(dim=0).float().cpu()
            if name in self.act_channel_mean_sum:
                self.act_channel_mean_sum[name].add_(ch_mean_sum)
            else:
                self.act_channel_mean_sum[name] = ch_mean_sum

            self.token_count[name] = self.token_count.get(name, 0) + n_tokens

    def _input_hook(self, m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        x_det = x.detach()
        self._accumulate_hessian(name, x_det)
        self._accumulate_channel_stats(name, x_det)

        if name not in self.weight_stats and hasattr(m, "weight") and m.weight is not None:
            with torch.no_grad():
                if m.weight.device.type != "meta":
                    w = m.weight.detach()
                    self.weight_stats[name] = {
                        "channel_max_input": w.abs().max(dim=0)[0].float().cpu(),
                        "channel_max_output": w.abs().max(dim=1)[0].float().cpu(),
                        "channel_max": w.abs().max(dim=0)[0].float().cpu(),
                        "layer_max": w.abs().max().float().cpu().item(),
                        "layer_mean": w.abs().mean().float().cpu().item(),
                        "num_channels_in": w.shape[1],
                        "num_channels_out": w.shape[0],
                        "num_channels": w.shape[1],
                    }

    def register_hooks(self, module_patterns: Optional[List[str]] = None):
        if module_patterns is None:
            module_patterns = ["language_model"]
        print(f"\nRegistering hooks (patterns={module_patterns}) ...")
        for name, m in self.model.named_modules():
            if not any(p in name for p in module_patterns):
                continue
            if isinstance(m, nn.Linear):
                self.hooks.append(
                    m.register_forward_hook(
                        functools.partial(self._input_hook, name=name)
                    )
                )
        print(f"  Registered {len(self.hooks)} hooks")
        print(f"  Mode: online Hessian accumulation (no raw activation storage)")

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    # ---------- inference ----------

    def collect_und_activations(
        self,
        model: nn.Module,
        processor: Any,
        tokenizer: Any,
        samples: List[Dict],
    ):
        """理解任务前向传播：所有样本都跑，Hessian 和 channel stats 在线积累。"""
        print(f"\n  [UND] Running {len(samples)} understanding samples "
              f"(online Hessian + channel stats) ...")

        for idx, sample in enumerate(tqdm(samples, desc="UND")):
            prompt = sample["prompt"]
            img = sample.get("image", None)
            pil_img = image_for_processor(img)

            with torch.no_grad():
                try:
                    inputs = processor(
                        prompt=prompt,
                        image=pil_img,
                        generation_mode="text",
                        padding=True,
                        return_tensors="pt",
                    )
                    for k, v in inputs.items():
                        if isinstance(v, torch.Tensor):
                            inputs[k] = v.to(model.device)

                    model.generate_hidden_states(
                        pixel_values=inputs.get("pixel_values", None),
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                    )
                except Exception as e:
                    print(f"\n  [UND] Error on sample {idx}: {e}")
                    continue

            if idx % 10 == 0:
                torch.cuda.empty_cache()
                gc.collect()

    def collect_gen_activations(
        self,
        model: nn.Module,
        processor: Any,
        tokenizer: Any,
        prompts: List[str],
    ):
        """生成任务前向传播：所有样本都跑，Hessian 和 channel stats 在线积累。"""
        print(f"\n  [GEN] Running {len(prompts)} generation samples "
              f"(online Hessian + channel stats) ...")

        for idx, prompt in enumerate(tqdm(prompts, desc="GEN")):
            with torch.no_grad():
                try:
                    inputs = processor(
                        prompt=prompt,
                        image=None,
                        generation_mode="image",
                        padding=True,
                        return_tensors="pt",
                    )
                    for k, v in inputs.items():
                        if isinstance(v, torch.Tensor):
                            inputs[k] = v.to(model.device)

                    model.generate_hidden_states(
                        pixel_values=None,
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                    )
                except Exception as e:
                    print(f"\n  [GEN] Error on sample {idx}: {e}")
                    continue

            if idx % 10 == 0:
                torch.cuda.empty_cache()
                gc.collect()

    # ---------- save ----------

    def collect_weight_stats(self, module_patterns: Optional[List[str]] = None):
        if module_patterns is None:
            module_patterns = ["language_model"]
        for name, m in self.model.named_modules():
            if not any(p in name for p in module_patterns):
                continue
            if name in self.weight_stats:
                continue
            if isinstance(m, nn.Linear) and hasattr(m, "weight") and m.weight is not None:
                with torch.no_grad():
                    if m.weight.device.type == "meta":
                        continue
                    w = m.weight.detach()
                    self.weight_stats[name] = {
                        "channel_max_input": w.abs().max(dim=0)[0].float().cpu(),
                        "channel_max_output": w.abs().max(dim=1)[0].float().cpu(),
                        "channel_max": w.abs().max(dim=0)[0].float().cpu(),
                        "layer_max": w.abs().max().float().cpu().item(),
                        "layer_mean": w.abs().mean().float().cpu().item(),
                        "num_channels_in": w.shape[1],
                        "num_channels_out": w.shape[0],
                        "num_channels": w.shape[1],
                    }

    @staticmethod
    def _safe_name(layer_name: str) -> str:
        return layer_name.replace(".", "__")

    def save(self):
        """分三类保存校准数据，各方法各取所需：

        1. gptq_hessian/   — GPTQ 专用
           每层一个 .pt: {"hessian_sum": (D,D) float64, "nsamples": int}
           Stage 20 用 Cholesky 分解重建等价 activation

        2. smoothquant/    — SmoothQuant 专用
           单个 .pt: {layer_name: {"channel_max": (D,), "weight_channel_max": (D,)}}
           直接用于计算迁移因子 s = act_max^α / w_max^(1-α)

        3. awq/            — AWQ 专用
           单个 .pt: {layer_name: {"channel_mean": (D,), "channel_max": (D,)}}
           channel_mean 用于 AWQ scale = act_mean^α

        Returns: (gptq_dir, smoothquant_path, awq_path)
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        layer_names = sorted(self.hessian_sum.keys())

        # ====== 1. GPTQ: per-layer Hessian ======
        gptq_dir = self.output_dir / "gptq_hessian"
        gptq_dir.mkdir(parents=True, exist_ok=True)
        gptq_index = {}
        gptq_disk_mb = 0.0

        print(f"\n[GPTQ] Saving per-layer Hessian ({len(layer_names)} layers) ...")
        for li, name in enumerate(layer_names):
            H = self.hessian_sum[name]
            n = self.hessian_nsamples[name]
            D = H.shape[0]

            safe = self._safe_name(name)
            out_path = gptq_dir / f"{safe}.pt"
            torch.save({"hessian_sum": H, "nsamples": n}, out_path)

            file_mb = out_path.stat().st_size / (1024 * 1024)
            gptq_disk_mb += file_mb
            gptq_index[name] = {
                "path": str(out_path), "hidden_dim": D,
                "nsamples": n, "file_mb": round(file_mb, 1),
            }

            if (li + 1) % 40 == 0 or (li + 1) == len(layer_names):
                print(f"  [{li+1}/{len(layer_names)}] cumulative: {gptq_disk_mb:.0f} MB")

        gptq_index_data = {
            "format": "gptq_hessian_v1",
            "num_layers": len(gptq_index),
            "total_disk_mb": round(gptq_disk_mb, 1),
            "timestamp": timestamp,
            "layers": gptq_index,
        }
        gptq_index_path = self.output_dir / f"gptq_hessian_index_{timestamp}.json"
        with open(gptq_index_path, "w") as f:
            json.dump(gptq_index_data, f, indent=2)
        _symlink(gptq_index_path, self.output_dir / "gptq_hessian_index_latest.json")
        print(f"  GPTQ Hessian: {gptq_dir} ({gptq_disk_mb:.0f} MB)")

        # ====== 2. SmoothQuant: channel_max (activation + weight) ======
        smooth_data = {}
        for name in layer_names:
            n = self.token_count.get(name, 0)
            D = self.hessian_sum[name].shape[0]
            smooth_data[name] = {
                "act_channel_max": self.act_channel_max.get(name, torch.zeros(D)),
                "weight_channel_max": (
                    self.weight_stats[name]["channel_max"]
                    if name in self.weight_stats else torch.zeros(D)
                ),
                "nsamples": n,
            }
        smooth_path = self.output_dir / f"smoothquant_stats_{timestamp}.pt"
        torch.save(smooth_data, smooth_path)
        _symlink(smooth_path, self.output_dir / "smoothquant_stats_latest.pt")
        smooth_mb = smooth_path.stat().st_size / (1024 * 1024)
        print(f"  SmoothQuant: {smooth_path} ({smooth_mb:.1f} MB)")

        # ====== 3. AWQ: channel_mean + channel_max ======
        awq_data = {}
        for name in layer_names:
            n = self.token_count.get(name, 0)
            D = self.hessian_sum[name].shape[0]
            ch_mean = (
                self.act_channel_mean_sum[name] / n
                if name in self.act_channel_mean_sum and n > 0
                else torch.zeros(D)
            )
            awq_data[name] = {
                "channel_mean": ch_mean,
                "channel_max": self.act_channel_max.get(name, torch.zeros(D)),
                "nsamples": n,
            }
        awq_path = self.output_dir / f"awq_stats_{timestamp}.pt"
        torch.save(awq_data, awq_path)
        _symlink(awq_path, self.output_dir / "awq_stats_latest.pt")
        awq_mb = awq_path.stat().st_size / (1024 * 1024)
        print(f"  AWQ: {awq_path} ({awq_mb:.1f} MB)")

        # ====== 总结 ======
        total_mb = gptq_disk_mb + smooth_mb + awq_mb
        print(f"\n  Total disk: {total_mb:.0f} MB")
        print(f"    GPTQ Hessian : {gptq_disk_mb:.0f} MB ({len(gptq_index)} layers)")
        print(f"    SmoothQuant  : {smooth_mb:.1f} MB")
        print(f"    AWQ          : {awq_mb:.1f} MB")

        return gptq_index_path, smooth_path, awq_path


# ============================================================
# Calibration Dataset Builder
# ============================================================

def load_external_calibration_dataset(dataset_path: str) -> List[Dict]:
    """
    从 build_calibration_dataset.py 生成的 JSON 文件加载校准样本。
    所有样本统一作为理解任务 (generation_mode="text") 处理。
    """
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Calibration dataset not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    raw = data.get("samples", data)
    if isinstance(raw, dict):
        raw = list(raw.values())

    samples = []
    for s in raw:
        img_path = s.get("image_path")
        img = None
        if img_path and Path(img_path).exists():
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception:
                pass
        samples.append({
            "prompt": s.get("question", s.get("prompt", "")),
            "image": img,
            "task_type": "und",
        })

    img_count = sum(1 for s in samples if s["image"] is not None)
    print(f"  Loaded external calibration dataset: {len(samples)} samples "
          f"({img_count} with images)")
    return samples


def build_calibration_data(
    num_und: int = 8,
    num_gen: int = 8,
    mme_data_root: Optional[str] = None,
) -> tuple:
    """
    构建混合校准数据集。

    Returns:
        (und_samples, gen_prompts)
        und_samples: List[Dict] with keys {prompt, image(PIL or None), task_type}
        gen_prompts: List[str]
    """
    # ------ UND ------
    und_samples = []

    if mme_data_root and Path(mme_data_root).exists():
        data_root = Path(mme_data_root)
        categories = sorted([d.name for d in data_root.iterdir() if d.is_dir()])
        if categories:
            samples_per_cat = max(1, num_und // len(categories))
            for cat in categories:
                cat_path = data_root / cat
                for txt_file in sorted(cat_path.glob("*.txt"))[:samples_per_cat]:
                    if len(und_samples) >= num_und:
                        break
                    with open(txt_file, "r") as f:
                        lines = f.readlines()
                    img_name = txt_file.stem
                    img_path = None
                    for ext in [".png", ".jpg", ".jpeg"]:
                        p = cat_path / f"{img_name}{ext}"
                        if p.exists():
                            img_path = p
                            break
                    if img_path is None:
                        img_dir = cat_path / "images"
                        if img_dir.exists():
                            for ext in [".png", ".jpg", ".jpeg"]:
                                p = img_dir / f"{img_name}{ext}"
                                if p.exists():
                                    img_path = p
                                    break
                    if img_path is None:
                        continue
                    for line in lines:
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split("\t")
                        if len(parts) >= 2:
                            try:
                                img = Image.open(str(img_path)).convert("RGB")
                                und_samples.append({
                                    "prompt": parts[0] + " Answer the question using a single word or phrase.",
                                    "image": img,
                                    "task_type": "und",
                                })
                            except Exception:
                                pass
                            if len(und_samples) >= num_und:
                                break

    if len(und_samples) < num_und:
        for p in DEFAULT_UND_PROMPTS[: num_und - len(und_samples)]:
            und_samples.append({"prompt": p, "image": None, "task_type": "und"})

    # ------ GEN ------
    gen_prompts = DEFAULT_GEN_PROMPTS[:num_gen]
    if len(gen_prompts) < num_gen:
        reps = (num_gen // len(DEFAULT_GEN_PROMPTS)) + 1
        gen_prompts = (DEFAULT_GEN_PROMPTS * reps)[:num_gen]

    return und_samples, gen_prompts


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Stage 0 (InternVL-U): Collect full activations for GPTQ/AWQ"
    )
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--gpu_ids", type=str, default=None)
    parser.add_argument("--max_mem_per_gpu", type=str, default="40GiB")
    parser.add_argument("--num_und", type=int, default=8,
                        help="Number of understanding calibration samples")
    parser.add_argument("--num_gen", type=int, default=8,
                        help="Number of generation calibration samples")
    parser.add_argument("--und_ratio", type=float, default=None,
                        help="Override: fraction of total samples for UND (e.g. 0.5)")
    parser.add_argument("--total_samples", type=int, default=None,
                        help="Total calibration samples (used with --und_ratio)")
    parser.add_argument("--mme_data_root", type=str, default=None,
                        help="MME Benchmark directory for real VQA images")
    parser.add_argument("--calibration_dataset", type=str, default=None,
                        help="External calibration dataset JSON "
                             "(from build_calibration_dataset.py). "
                             "If provided, --num_und/--num_gen/--mme_data_root are ignored.")
    parser.add_argument("--module_patterns", type=str, default="language_model",
                        help="Comma-separated module name patterns to hook")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.config:
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)
        for k, v in cfg.items():
            if hasattr(args, k) and getattr(args, k) is None:
                setattr(args, k, v)

    if args.output_dir is None:
        args.output_dir = str(
            Path(__file__).resolve().parent.parent
            / "quantization_outputs"
            / "stage0_full_activation"
        )

    # und / gen sample counts
    if args.und_ratio is not None and args.total_samples is not None:
        args.num_und = int(args.total_samples * args.und_ratio)
        args.num_gen = args.total_samples - args.num_und

    torch.manual_seed(args.seed)

    print("\n" + "=" * 70)
    print("Stage 0 (InternVL-U): Online Activation Statistics Collection")
    print("=" * 70)
    print(f"  Model path    : {args.model_path}")
    print(f"  Output dir    : {args.output_dir}")
    print(f"  GPU IDs       : {args.gpu_ids or 'all'}")
    print(f"  Mode          : Online Hessian (GPTQ) + Channel Stats (AWQ/SmoothQuant)")
    print(f"                  No raw activation storage — all tokens contribute")
    if args.calibration_dataset:
        print(f"  Calib source  : EXTERNAL ({args.calibration_dataset})")
    else:
        print(f"  UND samples   : {args.num_und}")
        print(f"  GEN samples   : {args.num_gen}")
    print(f"  Module patterns: {args.module_patterns}")

    # ---- Load model ----
    print("\n[1/4] Loading InternVL-U model ...")
    components = load_internvlu(
        model_path=args.model_path,
        gpu_ids=args.gpu_ids,
        torch_dtype=torch.bfloat16,
    )
    model = components["model"]
    tokenizer = components["tokenizer"]
    processor = components["processor"]
    model.eval()

    print(f"  Model device: {model.device}")
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Total params: {total_params:.1f}M")

    # ---- Calibration data ----
    print("\n[2/4] Building calibration data ...")

    use_external = args.calibration_dataset is not None
    if use_external:
        print(f"  Using external calibration dataset: {args.calibration_dataset}")
        ext_samples = load_external_calibration_dataset(args.calibration_dataset)
        und_samples = ext_samples
        gen_prompts = []
    else:
        und_samples, gen_prompts = build_calibration_data(
            num_und=args.num_und,
            num_gen=args.num_gen,
            mme_data_root=args.mme_data_root,
        )
    print(f"  UND: {len(und_samples)} samples"
          f" ({'with images' if any(s.get('image') is not None for s in und_samples) else 'text-only'})")
    print(f"  GEN: {len(gen_prompts)} prompts")

    # ---- Collect ----
    print("\n[3/4] Collecting activations ...")
    module_patterns = [p.strip() for p in args.module_patterns.split(",")]

    collector = FullActivationCollector(
        model=model,
        output_dir=args.output_dir,
    )
    collector.register_hooks(module_patterns)

    if und_samples:
        collector.collect_und_activations(model, processor, tokenizer, und_samples)
    if gen_prompts:
        collector.collect_gen_activations(model, processor, tokenizer, gen_prompts)

    collector.remove_hooks()
    collector.collect_weight_stats(module_patterns)

    # ---- Save (三类分开存) ----
    print("\n[4/4] Saving results ...")
    gptq_path, smooth_path, awq_path = collector.save()

    total_input = len(und_samples) + len(gen_prompts)
    print("\n" + "=" * 70)
    print("Stage 0 Completed!")
    print(f"  GPTQ Hessian  : {gptq_path}")
    print(f"  SmoothQuant   : {smooth_path}")
    print(f"  AWQ           : {awq_path}")
    print(f"  Layers        : {len(collector.hessian_sum)}")
    print(f"  Input samples : {total_input}")

    token_counts = sorted(collector.token_count.items())
    if token_counts:
        min_n = min(c for _, c in token_counts)
        max_n = max(c for _, c in token_counts)
        print(f"  Tokens/layer  : min={min_n}, max={max_n}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
