"""
Stage 22 (InternVL-U): Modality-Specific Feature Distribution & Outlier Analysis

方向 1 实验 1：模态特征分布的"不对称性"研究

核心目标：
  1. 分离视觉 Token 和文本 Token 的隐状态
  2. 逐层统计激活值分布（均值、方差、最大值-Outliers、峰度）
  3. 在 Int8 / Int4 下分别计算视觉/文本 CKA 波动
  4. 输出统计 JSON + 可视化就绪数据

关键发现预期：
  处理视觉 Token 的子层激活值波动剧烈且 Outliers 极多，
  证明了对其进行专门保护的必要性。
"""

import os
import sys
import json
import gc
import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from tqdm import tqdm
from PIL import Image
import yaml
import argparse

_current_dir = Path(__file__).resolve().parent
_quant_dir = _current_dir.parent

if str(_quant_dir) not in sys.path:
    sys.path.insert(0, str(_quant_dir))

from utils.bagel_imports import HybridQuantLinear, LinearCKA, ALGORITHM_POOL
from utils.model_loader import load_internvlu
from utils.calibration import image_for_processor


# ================================================================
# Quantization Configs for Phase 3
# ================================================================

_W4_BASE = {
    'weight_bit': 4, 'act_bit': 4, 'act_unsigned': True,
    'use_smoothquant': False, 'smoothquant_alpha': 0.5,
    'use_awq': False, 'awq_alpha': 0.5, 'awq_n_grid': 20,
    'use_svd': False, 'svd_rank': 32,
    'use_sparse': False, 'sparse_threshold': 0.0001,
    'use_gptq': False,
    'gptq_group_size': 64, 'gptq_damp_percentage': 0.01, 'gptq_block_size': 128,
    'use_block_quant': False, 'use_block_quant_act': False,
    'block_size_weight': 256, 'block_size_act': 256,
}


def build_stage22_quant_configs():
    """构建多种量化配置，用于 Phase 3 CKA 敏感度测试。"""
    configs = {}
    configs['rtn_w4a4'] = {'weight_bit': 4, 'act_bit': 4, 'act_unsigned': True}

    c = dict(_W4_BASE)
    c.update(use_gptq=True, gptq_group_size=64)
    configs['gptq_w4a4'] = c

    for alpha in [0.5, 0.7]:
        c = dict(_W4_BASE)
        c.update(use_smoothquant=True, smoothquant_alpha=alpha,
                 use_gptq=True, gptq_group_size=64)
        configs[f'smooth_a{int(alpha*100)}_gptq_w4a4'] = c

    for alpha in [0.5, 0.7]:
        c = dict(_W4_BASE)
        c.update(use_smoothquant=True, smoothquant_alpha=alpha,
                 use_svd=True, svd_rank=32,
                 use_gptq=True, gptq_group_size=64)
        configs[f'svdquant_a{int(alpha*100)}_w4a4'] = c

    c = dict(_W4_BASE)
    c.update(use_awq=True, awq_n_grid=20, use_svd=True, svd_rank=32)
    configs['awq_svd_rtn_w4a4'] = c

    return configs


# ================================================================
# LazyActivationProvider (shared)
# ================================================================

class LazyActivationProvider:
    def __init__(self, gptq_hessian_index=None, smoothquant_stats=None,
                 awq_stats=None, legacy_activation_file=None,
                 cache_size=20, damp_ratio=0.01):
        self._cache = OrderedDict()
        self.cache_size = cache_size
        self.damp_ratio = damp_ratio
        self.hessian_index = None
        self.smooth_data = None
        self.awq_data = None
        if gptq_hessian_index and Path(gptq_hessian_index).exists():
            with open(gptq_hessian_index, "r") as f:
                self.hessian_index = json.load(f).get("layers", {})
            print(f"  [GPTQ] Loaded Hessian index: {len(self.hessian_index)} layers")
        if smoothquant_stats and Path(smoothquant_stats).exists():
            self.smooth_data = torch.load(smoothquant_stats, map_location="cpu")
            print(f"  [SmoothQuant] Loaded stats: {len(self.smooth_data)} layers")
        if awq_stats and Path(awq_stats).exists():
            self.awq_data = torch.load(awq_stats, map_location="cpu")
            print(f"  [AWQ] Loaded stats: {len(self.awq_data)} layers")
        self._legacy = None
        if legacy_activation_file and Path(legacy_activation_file).exists():
            self._legacy = torch.load(legacy_activation_file, map_location="cpu")
        self.available = (self.hessian_index is not None or self._legacy is not None)

    def __contains__(self, name):
        return (self._legacy and name in self._legacy) or \
               (self.hessian_index and name in self.hessian_index)

    def get_activation(self, name):
        if name in self._cache:
            self._cache.move_to_end(name)
            return self._cache[name]
        act = self._load(name)
        if act is not None:
            self._cache[name] = act
            if len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return act

    def get_activation_max(self, name):
        if self.smooth_data and name in self.smooth_data:
            return self.smooth_data[name].get("act_channel_max")
        if self.awq_data and name in self.awq_data:
            return self.awq_data[name].get("channel_max")
        return None

    def _load(self, name):
        if self._legacy and name in self._legacy:
            return self._legacy[name]
        if self.hessian_index and name in self.hessian_index:
            info = self.hessian_index[name]
            data = torch.load(info["path"], map_location="cpu", weights_only=True)
            H = data["hessian_sum"].double()
            D = H.shape[0]
            damp = self.damp_ratio * H.diagonal().mean()
            H_reg = H + damp * torch.eye(D, dtype=H.dtype)
            try:
                return torch.linalg.cholesky(H_reg).t().float()
            except torch.linalg.LinAlgError:
                e, v = torch.linalg.eigh(H_reg)
                return (v * e.clamp(min=0).sqrt().unsqueeze(0)).t().float()
        return None

    def clear_cache(self):
        self._cache.clear()
        gc.collect()


def _linear_restore_device(module: nn.Module) -> torch.device:
    """推断 Linear / HybridQuantLinear 所在设备。prepare_weight 后 HybridQuantLinear.weight 可能为 None。"""
    w = getattr(module, "weight", None)
    if w is not None:
        return w.device
    if isinstance(module, HybridQuantLinear):
        if module.quantized_weight is not None:
            return module.quantized_weight.device
        if getattr(module, "weight_scale", None) is not None:
            return module.weight_scale.device
    b = getattr(module, "bias", None)
    if b is not None:
        return b.device
    raise RuntimeError(f"Cannot infer device for {type(module).__name__}")


# ================================================================
# Modality-Aware Hidden State Collector
# ================================================================

class ModalityAwareCollector:
    """收集隐状态时同步记录 input_ids，以便后续按模态分离。"""

    def __init__(self, model, processor, tokenizer):
        self.model = model
        self.processor = processor
        self.tokenizer = tokenizer
        self.img_context_token_id = model.img_context_token_id
        self._last_input_ids = None

    def forward_and_capture(
        self,
        sample: Dict,
        layer_idx: int,
    ) -> Optional[Dict]:
        """前向一条样本，返回 {hidden_states, vision_mask, text_mask}。"""
        prompt = sample.get("prompt", "")
        gen_mode = sample.get("generation_mode", "text")
        pil_image = image_for_processor(sample.get("image"))

        inputs = self.processor(
            prompt=[prompt],
            image=[pil_image] if pil_image is not None else [None],
            generation_mode=gen_mode, padding=True, return_tensors="pt",
        )
        device = next(self.model.parameters()).device
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(device)

        input_ids = inputs["input_ids"]
        self._last_input_ids = input_ids.detach().cpu()

        captured = {}

        def hook_fn(module, inp, output):
            h = output[0] if isinstance(output, tuple) else output
            captured["output"] = h.detach().cpu()

        decoder_layer = self.model.language_model.model.layers[layer_idx]
        handle = decoder_layer.register_forward_hook(hook_fn)
        try:
            with torch.no_grad():
                pv = inputs.get("pixel_values")
                if gen_mode == "image":
                    pv = None
                self.model.generate_hidden_states(
                    pixel_values=pv,
                    input_ids=input_ids,
                    attention_mask=inputs.get("attention_mask"),
                )
        except Exception:
            return None
        finally:
            handle.remove()

        if "output" not in captured:
            return None

        hs = captured["output"]  # [B, S, D]
        ids = self._last_input_ids  # [B, S]

        B, S = ids.shape
        ids_flat = ids.reshape(-1)
        vision_mask = (ids_flat == self.img_context_token_id)
        text_mask = ~vision_mask

        return {
            "hidden_states": hs,
            "vision_mask": vision_mask,
            "text_mask": text_mask,
            "has_vision": vision_mask.any().item(),
        }


# ================================================================
# Activation Statistics Calculator
# ================================================================

def compute_activation_stats(tensor: torch.Tensor) -> Dict:
    """对 (N, D) tensor 计算统计量。"""
    if tensor.numel() == 0:
        return {
            "mean": 0.0, "std": 0.0, "abs_max": 0.0,
            "abs_mean": 0.0, "kurtosis": 0.0,
            "outlier_ratio_3sigma": 0.0, "outlier_ratio_6sigma": 0.0,
            "num_tokens": 0, "percentile_99": 0.0, "percentile_999": 0.0,
        }
    t = tensor.float()
    mean_val = t.mean().item()
    std_val = t.std().item()
    abs_max = t.abs().max().item()
    abs_mean = t.abs().mean().item()

    centered = t - t.mean()
    var = centered.pow(2).mean()
    kurt = (centered.pow(4).mean() / (var.pow(2) + 1e-10) - 3.0).item()

    threshold_3sigma = 3.0 * std_val
    threshold_6sigma = 6.0 * std_val
    outlier_3 = (t.abs() > (abs_mean + threshold_3sigma)).float().mean().item()
    outlier_6 = (t.abs() > (abs_mean + threshold_6sigma)).float().mean().item()

    sorted_abs = t.abs().flatten().sort().values
    n = sorted_abs.numel()
    p99 = sorted_abs[min(int(n * 0.99), n - 1)].item()
    p999 = sorted_abs[min(int(n * 0.999), n - 1)].item()

    return {
        "mean": round(mean_val, 6),
        "std": round(std_val, 6),
        "abs_max": round(abs_max, 6),
        "abs_mean": round(abs_mean, 6),
        "kurtosis": round(kurt, 4),
        "outlier_ratio_3sigma": round(outlier_3, 6),
        "outlier_ratio_6sigma": round(outlier_6, 6),
        "num_tokens": tensor.shape[0],
        "percentile_99": round(p99, 6),
        "percentile_999": round(p999, 6),
    }


def _apply_quant_to_layer(model, layer_idx, algo_config, original_weights,
                          activation_provider=None):
    """对 decoder layer 的所有 Linear 替换为量化层（不恢复），返回量化的 name 列表。"""
    prefix = f"language_model.model.layers.{layer_idx}"
    decoder_layer = model.language_model.model.layers[layer_idx]

    def _get_module(name):
        parts = name.split(".")
        m = model
        for p in parts:
            m = getattr(m, p)
        return m

    def _replace_module(name, new_mod):
        parts = name.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], new_mod)

    linear_names = []
    for sub_name, sub_module in decoder_layer.named_modules():
        if isinstance(sub_module, (nn.Linear, HybridQuantLinear)):
            full_name = f"{prefix}.{sub_name}" if sub_name else prefix
            linear_names.append(full_name)

    for lname in linear_names:
        module = _get_module(lname)
        device = module.weight.device
        dtype = module.weight.dtype
        config = algo_config.copy()

        has_act = activation_provider is not None and activation_provider.available
        if config.get("use_gptq") and not has_act:
            config["use_gptq"] = False
        if config.get("use_awq") and not has_act:
            config["use_awq"] = False

        quant_layer = HybridQuantLinear(
            in_features=module.in_features, out_features=module.out_features,
            bias=module.bias is not None,
            weight_bit=config.get("weight_bit", 4), act_bit=config.get("act_bit", 4),
            quant_percentile=config.get("quant_percentile", 0.999999),
            act_unsigned=config.get("act_unsigned", True),
            use_sparse=config.get("use_sparse", False),
            sparse_ratio=config.get("sparse_ratio", 0.0),
            sparse_threshold=config.get("sparse_threshold", None),
            use_smoothquant=config.get("use_smoothquant", False),
            smoothquant_alpha=config.get("smoothquant_alpha", 0.5),
            use_svd=config.get("use_svd", False), svd_rank=config.get("svd_rank", 0),
            use_block_quant=config.get("use_block_quant", False),
            use_block_quant_act=config.get("use_block_quant_act", False),
            block_size_weight=config.get("block_size_weight", 256),
            block_size_act=config.get("block_size_act", 256),
            use_gptq=config.get("use_gptq", False),
            gptq_group_size=config.get("gptq_group_size", 64),
            gptq_damp_percentage=config.get("gptq_damp_percentage", 0.01),
            gptq_block_size=config.get("gptq_block_size", 128),
            use_awq=config.get("use_awq", False),
            awq_alpha=config.get("awq_alpha", 0.5),
            awq_n_grid=config.get("awq_n_grid", 20),
            device=device, dtype=dtype,
        )
        if lname in original_weights:
            orig_w = original_weights[lname]["weight"].clone()
            orig_b = original_weights[lname]["bias"]
            if orig_b is not None:
                orig_b = orig_b.clone()
        else:
            orig_w = module.weight.data.clone()
            orig_b = module.bias.data.clone() if module.bias is not None else None
        quant_layer.weight.data = orig_w.to(device)
        if orig_b is not None:
            quant_layer.bias.data = orig_b.to(device)
        quant_layer = quant_layer.to(device)

        act_data = None
        act_max = None
        if has_act and lname in activation_provider:
            act_data = activation_provider.get_activation(lname)
        if activation_provider is not None:
            act_max = activation_provider.get_activation_max(lname)

        quant_layer.prepare_weight(
            activation_max=act_max, activation_data=act_data,
            layer_name=lname, verbose=False,
        )
        _replace_module(lname, quant_layer)

    return linear_names


def _restore_layer_weights(model, layer_idx, original_weights):
    """恢复 decoder layer 的所有 HybridQuantLinear 为原始 FP16 Linear。"""
    prefix = f"language_model.model.layers.{layer_idx}"
    decoder_layer = model.language_model.model.layers[layer_idx]

    def _replace_module(name, new_mod):
        parts = name.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], new_mod)

    for sub_name, sub_module in decoder_layer.named_modules():
        if isinstance(sub_module, HybridQuantLinear):
            full_name = f"{prefix}.{sub_name}" if sub_name else prefix
            if full_name in original_weights:
                device = _linear_restore_device(sub_module)
                dtype = original_weights[full_name]["weight"].dtype
                linear = nn.Linear(
                    sub_module.in_features, sub_module.out_features,
                    bias=sub_module.bias is not None, device=device, dtype=dtype,
                )
                linear.weight.data.copy_(original_weights[full_name]["weight"].to(device))
                if linear.bias is not None and original_weights[full_name]["bias"] is not None:
                    linear.bias.data.copy_(original_weights[full_name]["bias"].to(device))
                _replace_module(full_name, linear)


# ================================================================
# Stage 22 Analyzer
# ================================================================

class ModalityOutlierAnalyzer:
    """Stage 22: 模态特征分布 Outlier 分析。"""

    def __init__(
        self,
        model_path: str,
        output_dir: str = "./quantization_outputs/stage22_modality_analysis",
        calibration_dataset: Optional[str] = None,
        gptq_hessian_index: Optional[str] = None,
        smoothquant_stats: Optional[str] = None,
        awq_stats: Optional[str] = None,
        activation_data_file: Optional[str] = None,
        gpu_ids: Optional[str] = None,
        target_decoder_layers: Optional[List[int]] = None,
        num_samples: int = 1000,
        subsample_step: int = 5,
        seed: int = 42,
        run_date: Optional[str] = None,
        quant_methods: Optional[str] = None,
    ):
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.calibration_dataset = calibration_dataset
        self.gpu_ids = gpu_ids
        self.target_decoder_layers = target_decoder_layers
        self.num_samples = num_samples
        self.subsample_step = subsample_step
        self.seed = seed
        self.run_date = run_date or datetime.now().strftime("%Y%m%d")
        self.quant_methods = quant_methods

        if self.gpu_ids:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_ids)

        self._set_seed()
        self.activation_provider = LazyActivationProvider(
            gptq_hessian_index=gptq_hessian_index,
            smoothquant_stats=smoothquant_stats,
            awq_stats=awq_stats,
            legacy_activation_file=activation_data_file,
        )
        self._load_model()
        self.original_weights = {}
        self._save_original_weights()

        self.num_decoder_layers = len(self.model.language_model.model.layers)
        if self.target_decoder_layers is None:
            self.target_decoder_layers = list(range(self.num_decoder_layers))

        self.collector = ModalityAwareCollector(
            self.model, self.processor, self.tokenizer
        )

        self._print_banner()

    def _set_seed(self):
        import random
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    def _load_model(self):
        print(f"\nLoading InternVL-U from {self.model_path} ...")
        components = load_internvlu(
            self.model_path, gpu_ids=self.gpu_ids, torch_dtype=torch.bfloat16,
        )
        self.model = components["model"].eval()
        self.tokenizer = components["tokenizer"]
        self.processor = components["processor"]
        print("  Model loaded successfully")

    def _save_original_weights(self):
        count = 0
        for name, module in self.model.named_modules():
            if not name.startswith("language_model.model.layers."):
                continue
            if isinstance(module, nn.Linear) and not isinstance(module, HybridQuantLinear):
                self.original_weights[name] = {
                    "weight": module.weight.data.clone().cpu(),
                    "bias": module.bias.data.clone().cpu() if module.bias is not None else None,
                }
                count += 1
        print(f"  Saved {count} original weights")

    def _print_banner(self):
        print("\n" + "=" * 80)
        print("Stage 22 (InternVL-U): Modality-Specific Outlier Analysis")
        print("=" * 80)
        print(f"  Model: {self.model_path}")
        print(f"  Run date: {self.run_date}")
        print(f"  Decoder layers: {self.num_decoder_layers}")
        print(f"  Target layers: {self.target_decoder_layers}")
        print(f"  Num samples: {self.num_samples}")
        print(f"  Subsample step: {self.subsample_step}")
        print(f"  Output: {self.output_dir}")
        print("=" * 80 + "\n")

    def _load_calibration_samples(self) -> List[Dict]:
        """加载校准样本，优先使用带图样本。"""
        if self.calibration_dataset:
            from stages.stage20_largecalib_search import LargeCalibrationLoader
            loader = LargeCalibrationLoader(
                dataset_path=self.calibration_dataset,
                max_samples=self.num_samples,
            )
            samples = loader.load()
            with_img = [s for s in samples if s.get("image") is not None]
            without_img = [s for s in samples if s.get("image") is None]
            if len(with_img) < 10:
                print("  [WARN] Too few image samples for vision analysis, "
                      "results may be unreliable")
            return with_img + without_img
        else:
            from utils.calibration import CalibrationDataLoader
            loader = CalibrationDataLoader(
                num_und_samples=self.num_samples,
                num_gen_samples=0,
            )
            return loader.prepare_calibration_samples()

    def analyze(self) -> Dict:
        """执行完整的模态 Outlier 分析。"""
        print("\n" + "=" * 80)
        print("Phase 1: Loading calibration samples")
        print("=" * 80)
        samples = self._load_calibration_samples()
        img_samples = [s for s in samples if s.get("image") is not None]
        txt_samples = [s for s in samples if s.get("image") is None]
        print(f"  With images: {len(img_samples)}, Text-only: {len(txt_samples)}")

        print("\n" + "=" * 80)
        print("Phase 2: Per-layer modality activation analysis")
        print("=" * 80)

        layer_results = {}

        for layer_idx in tqdm(self.target_decoder_layers, desc="Analyzing layers"):
            vision_tokens_all = []
            text_tokens_all = []
            full_tokens_all = []

            for sample in samples:
                result = self.collector.forward_and_capture(sample, layer_idx)
                if result is None:
                    continue

                hs = result["hidden_states"]  # [B, S, D]
                B, S, D = hs.shape
                hs_flat = hs.reshape(B * S, D)

                full_tokens_all.append(hs_flat)

                if result["has_vision"]:
                    v_mask = result["vision_mask"]
                    t_mask = result["text_mask"]
                    if v_mask.any():
                        vision_tokens_all.append(hs_flat[v_mask])
                    if t_mask.any():
                        text_tokens_all.append(hs_flat[t_mask])
                else:
                    text_tokens_all.append(hs_flat)

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            vision_cat = torch.cat(vision_tokens_all, dim=0) if vision_tokens_all else torch.tensor([])
            text_cat = torch.cat(text_tokens_all, dim=0) if text_tokens_all else torch.tensor([])
            full_cat = torch.cat(full_tokens_all, dim=0) if full_tokens_all else torch.tensor([])

            vision_stats = compute_activation_stats(vision_cat)
            text_stats = compute_activation_stats(text_cat)
            full_stats = compute_activation_stats(full_cat)

            # Per-channel outlier 分析
            vision_channel_stats = {}
            text_channel_stats = {}
            if vision_cat.numel() > 0 and vision_cat.dim() == 2:
                v_abs_max_per_channel = vision_cat.abs().max(dim=0).values
                t_abs_max_per_channel = text_cat.abs().max(dim=0).values if text_cat.numel() > 0 and text_cat.dim() == 2 else torch.zeros(vision_cat.shape[1])
                vision_channel_stats = {
                    "channel_abs_max_mean": round(v_abs_max_per_channel.mean().item(), 6),
                    "channel_abs_max_std": round(v_abs_max_per_channel.std().item(), 6),
                    "channel_abs_max_max": round(v_abs_max_per_channel.max().item(), 6),
                    "outlier_channels_count": int((v_abs_max_per_channel > v_abs_max_per_channel.mean() + 3 * v_abs_max_per_channel.std()).sum().item()),
                }
                text_channel_stats = {
                    "channel_abs_max_mean": round(t_abs_max_per_channel.mean().item(), 6),
                    "channel_abs_max_std": round(t_abs_max_per_channel.std().item(), 6),
                    "channel_abs_max_max": round(t_abs_max_per_channel.max().item(), 6),
                    "outlier_channels_count": int((t_abs_max_per_channel > t_abs_max_per_channel.mean() + 3 * t_abs_max_per_channel.std()).sum().item()),
                }

            layer_results[layer_idx] = {
                "vision": {**vision_stats, **{"channel": vision_channel_stats}},
                "text": {**text_stats, **{"channel": text_channel_stats}},
                "full": full_stats,
                "sensitivity_ratio": {
                    "abs_max_ratio": round(
                        vision_stats["abs_max"] / (text_stats["abs_max"] + 1e-10), 4
                    ),
                    "std_ratio": round(
                        vision_stats["std"] / (text_stats["std"] + 1e-10), 4
                    ),
                    "outlier_3sigma_ratio": round(
                        vision_stats["outlier_ratio_3sigma"] / (text_stats["outlier_ratio_3sigma"] + 1e-10), 4
                    ),
                    "kurtosis_diff": round(
                        vision_stats["kurtosis"] - text_stats["kurtosis"], 4
                    ),
                },
            }

            print(f"  Layer {layer_idx}: "
                  f"V_max={vision_stats['abs_max']:.4f} T_max={text_stats['abs_max']:.4f} "
                  f"V_outlier3σ={vision_stats['outlier_ratio_3sigma']:.4f} "
                  f"T_outlier3σ={text_stats['outlier_ratio_3sigma']:.4f}")

            del vision_tokens_all, text_tokens_all, full_tokens_all
            del vision_cat, text_cat, full_cat
            gc.collect()

        # ---- Phase 3: CKA under multiple quantization methods ----
        print("\n" + "=" * 80)
        print("Phase 3: CKA sensitivity under multiple quantization methods (vision vs text)")
        print("=" * 80)

        quant_configs = build_stage22_quant_configs()
        if hasattr(self, 'quant_methods') and self.quant_methods:
            selected = [m.strip() for m in self.quant_methods.split(",")]
            quant_configs = {k: v for k, v in quant_configs.items() if k in selected}
            if not quant_configs:
                print(f"  [WARN] None of {selected} matched pool, using all methods")
                quant_configs = build_stage22_quant_configs()
        print(f"  Quantization methods: {list(quant_configs.keys())}")
        print(f"  Using all {len(samples)} samples on all {len(self.target_decoder_layers)} layers")

        cka_analysis = {}

        for layer_idx in tqdm(self.target_decoder_layers, desc="CKA analysis"):
            fp_vision_tokens = []
            fp_text_tokens = []
            fp_all_tokens = []

            for sample in samples:
                result = self.collector.forward_and_capture(sample, layer_idx)
                if result is None:
                    continue
                hs = result["hidden_states"]
                B, S, D = hs.shape
                hs_flat = hs.reshape(B * S, D)
                fp_all_tokens.append(hs_flat)
                if result["has_vision"]:
                    v_mask = result["vision_mask"]
                    t_mask = result["text_mask"]
                    if v_mask.any():
                        fp_vision_tokens.append(hs_flat[v_mask])
                    if t_mask.any():
                        fp_text_tokens.append(hs_flat[t_mask])
                else:
                    fp_text_tokens.append(hs_flat)

            cka_results = {}
            for method_name, method_config in quant_configs.items():
                quant_vision_tokens = []
                quant_text_tokens = []
                quant_all_tokens = []

                _apply_quant_to_layer(
                    self.model, layer_idx, method_config, self.original_weights,
                    activation_provider=self.activation_provider,
                )

                for sample in samples:
                    result = self.collector.forward_and_capture(sample, layer_idx)
                    if result is None:
                        continue
                    hs = result["hidden_states"]
                    B, S, D = hs.shape
                    hs_flat = hs.reshape(B * S, D)
                    quant_all_tokens.append(hs_flat)
                    if result["has_vision"]:
                        v_mask = result["vision_mask"]
                        t_mask = result["text_mask"]
                        if v_mask.any():
                            quant_vision_tokens.append(hs_flat[v_mask])
                        if t_mask.any():
                            quant_text_tokens.append(hs_flat[t_mask])
                    else:
                        quant_text_tokens.append(hs_flat)

                _restore_layer_weights(
                    self.model, layer_idx, self.original_weights
                )

                cka_all = LinearCKA.compute_batched(
                    fp_all_tokens, quant_all_tokens, subsample_step=self.subsample_step
                ) if quant_all_tokens else 0.0
                cka_vision = LinearCKA.compute_batched(
                    fp_vision_tokens, quant_vision_tokens, subsample_step=self.subsample_step
                ) if quant_vision_tokens and fp_vision_tokens else 0.0
                cka_text = LinearCKA.compute_batched(
                    fp_text_tokens, quant_text_tokens, subsample_step=self.subsample_step
                ) if quant_text_tokens and fp_text_tokens else 0.0

                cka_results[method_name] = {
                    "cka_all": round(cka_all, 6),
                    "cka_vision": round(cka_vision, 6),
                    "cka_text": round(cka_text, 6),
                    "cka_drop_vision": round(1.0 - cka_vision, 6),
                    "cka_drop_text": round(1.0 - cka_text, 6),
                    "vision_text_gap": round(cka_text - cka_vision, 6),
                }

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            cka_analysis[layer_idx] = cka_results
            summary_parts = []
            for mn in list(quant_configs.keys())[:3]:
                if mn in cka_results:
                    summary_parts.append(
                        f"{mn} V={cka_results[mn]['cka_vision']:.4f} T={cka_results[mn]['cka_text']:.4f}")
            print(f"  Layer {layer_idx}: {' | '.join(summary_parts)}")
            self.activation_provider.clear_cache()

        # ---- Export results ----
        print("\n" + "=" * 80)
        print("Phase 4: Exporting analysis results")
        print("=" * 80)

        results = {
            "layer_statistics": {
                str(k): v for k, v in layer_results.items()
            },
            "cka_sensitivity": {
                str(k): v for k, v in cka_analysis.items()
            },
            "metadata": {
                "stage": 22,
                "analysis_type": "modality_outlier_distribution",
                "model_path": self.model_path,
                "run_date": self.run_date,
                "num_decoder_layers": self.num_decoder_layers,
                "target_layers": self.target_decoder_layers,
                "num_samples": self.num_samples,
                "img_context_token_id": self.model.img_context_token_id,
                "seed": self.seed,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            },
            "summary": self._compute_summary(layer_results, cka_analysis),
        }

        results_path = self.output_dir / f"stage22_modality_analysis_{self.run_date}.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Results: {results_path}")

        self._print_summary(results["summary"])
        return results

    def _compute_summary(self, layer_results: Dict, cka_analysis: Dict) -> Dict:
        """生成全局摘要统计。"""
        v_maxes = [v["vision"]["abs_max"] for v in layer_results.values() if v["vision"]["num_tokens"] > 0]
        t_maxes = [v["text"]["abs_max"] for v in layer_results.values() if v["text"]["num_tokens"] > 0]
        v_outliers = [v["vision"]["outlier_ratio_3sigma"] for v in layer_results.values() if v["vision"]["num_tokens"] > 0]
        t_outliers = [v["text"]["outlier_ratio_3sigma"] for v in layer_results.values() if v["text"]["num_tokens"] > 0]

        avg_v_max = np.mean(v_maxes) if v_maxes else 0.0
        avg_t_max = np.mean(t_maxes) if t_maxes else 0.0
        avg_v_outlier = np.mean(v_outliers) if v_outliers else 0.0
        avg_t_outlier = np.mean(t_outliers) if t_outliers else 0.0

        per_method_gaps = {}
        for layer_id, methods_dict in cka_analysis.items():
            for method_name, method_scores in methods_dict.items():
                gap = method_scores.get("vision_text_gap")
                if gap is not None:
                    per_method_gaps.setdefault(method_name, []).append(gap)

        per_method_avg = {m: round(float(np.mean(gaps)), 6) for m, gaps in per_method_gaps.items()}
        all_gaps = [g for gaps in per_method_gaps.values() for g in gaps]

        return {
            "avg_vision_abs_max": round(float(avg_v_max), 4),
            "avg_text_abs_max": round(float(avg_t_max), 4),
            "avg_vision_outlier_3sigma": round(float(avg_v_outlier), 6),
            "avg_text_outlier_3sigma": round(float(avg_t_outlier), 6),
            "vision_to_text_max_ratio": round(float(avg_v_max / (avg_t_max + 1e-10)), 4),
            "vision_to_text_outlier_ratio": round(float(avg_v_outlier / (avg_t_outlier + 1e-10)), 4),
            "avg_cka_gap_vision_vs_text": round(float(np.mean(all_gaps)) if all_gaps else 0.0, 6),
            "per_method_cka_gap": per_method_avg,
            "conclusion": (
                "CONFIRMED" if avg_v_outlier > avg_t_outlier * 1.2
                else "INCONCLUSIVE"
            ),
        }

    def _print_summary(self, summary: Dict):
        print(f"\n{'=' * 80}")
        print("Stage 22 Analysis Summary:")
        print(f"  Vision avg abs_max:        {summary['avg_vision_abs_max']:.4f}")
        print(f"  Text avg abs_max:          {summary['avg_text_abs_max']:.4f}")
        print(f"  Vision/Text max ratio:     {summary['vision_to_text_max_ratio']:.4f}x")
        print(f"  Vision avg outlier(3σ):    {summary['avg_vision_outlier_3sigma']:.6f}")
        print(f"  Text avg outlier(3σ):      {summary['avg_text_outlier_3sigma']:.6f}")
        print(f"  Vision/Text outlier ratio: {summary['vision_to_text_outlier_ratio']:.4f}x")
        print(f"  Avg CKA gap (T-V):         {summary['avg_cka_gap_vision_vs_text']:.6f}")
        per_method = summary.get("per_method_cka_gap", {})
        if per_method:
            print("  Per-method CKA gap (T-V):")
            for method, gap in sorted(per_method.items()):
                print(f"    {method:30s} {gap:.6f}")
        print(f"  Hypothesis:                {summary['conclusion']}")
        print(f"{'=' * 80}\n")


# ================================================================
# CLI
# ================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Stage 22 (InternVL-U): Modality-Specific Outlier Analysis"
    )
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str,
                        default="./quantization_outputs/stage22_modality_analysis")
    parser.add_argument("--calibration_dataset", type=str, default=None,
                        help="Path to calibration dataset JSON (with image samples)")
    parser.add_argument("--gptq_hessian_index", type=str, default=None)
    parser.add_argument("--smoothquant_stats", type=str, default=None)
    parser.add_argument("--awq_stats", type=str, default=None)
    parser.add_argument("--activation_data", type=str, default=None)
    parser.add_argument("--gpu_ids", type=str, default=None)
    parser.add_argument("--target_layers", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--subsample_step", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quant_methods", type=str, default=None,
                        help="Comma-separated quantization methods to test. "
                             "Available: rtn_w4a4,gptq_w4a4,smooth_a50_gptq_w4a4,"
                             "smooth_a70_gptq_w4a4,svdquant_a50_w4a4,svdquant_a70_w4a4,"
                             "awq_svd_rtn_w4a4. Default: all")
    parser.add_argument("--run_date", type=str, default=None)
    parser.add_argument("--config", type=str, default=None, help="YAML config file")

    args = parser.parse_args()

    if args.config:
        with open(args.config, "r") as f:
            yaml_config = yaml.safe_load(f)
        for key, val in yaml_config.items():
            if hasattr(args, key) and getattr(args, key) is None:
                setattr(args, key, val)

    target_layers = None
    if args.target_layers:
        target_layers = [int(x.strip()) for x in args.target_layers.split(",")]

    analyzer = ModalityOutlierAnalyzer(
        model_path=args.model_path,
        output_dir=args.output_dir,
        calibration_dataset=args.calibration_dataset,
        gptq_hessian_index=args.gptq_hessian_index,
        smoothquant_stats=args.smoothquant_stats,
        awq_stats=args.awq_stats,
        activation_data_file=args.activation_data,
        gpu_ids=args.gpu_ids,
        target_decoder_layers=target_layers,
        num_samples=args.num_samples,
        subsample_step=args.subsample_step,
        seed=args.seed,
        run_date=args.run_date,
        quant_methods=args.quant_methods,
    )
    results = analyzer.analyze()
    print("\nStage 22 analysis completed!")


if __name__ == "__main__":
    main()
