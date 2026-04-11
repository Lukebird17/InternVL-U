"""
Stage 24 (InternVL-U): Attention Map Fidelity Analysis & Search

方向 2：空间注意力图的"保真度"保护

三个子阶段：
  Phase A — Attention Divergence 测量
    计算 FP16 与量化后 attention matrix 之间的 KL 散度 / Jensen-Shannon 散度，
    证明即使 hidden state CKA 很高，attention map 本身可能已产生巨大偏差。

  Phase B — Attention-Fidelity 搜索
    在功能组搜索中加入 attention divergence 作为辅助指标，
    联合 CKA + attention fidelity 选择最优量化算法。

  Phase C — Saliency Map 可视化
    生成 FP16 / 量化后的 attention saliency map，
    可视化验证方法如何让注意力焦点保持稳定。

技术要点：
  使用 eager attention 模式替代 flash_attention_2，
  以获取完整的 attention weight matrix。
"""

import os
import sys
import json
import gc
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
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

from utils.bagel_imports import HybridQuantLinear, LinearCKA
from utils.model_loader import load_internvlu
from utils.calibration import image_for_processor


# ================================================================
# Attention Divergence Metrics
# ================================================================

def attention_kl_divergence(
    attn_fp: torch.Tensor,
    attn_quant: torch.Tensor,
    eps: float = 1e-8,
) -> float:
    """逐头 KL(fp || quant) 再平均。

    Args:
        attn_fp: [num_heads, seq_len, seq_len] FP16 attention probs
        attn_quant: [num_heads, seq_len, seq_len] quantized attention probs
    """
    p = attn_fp.float().clamp(min=eps)
    q = attn_quant.float().clamp(min=eps)
    p = p / p.sum(dim=-1, keepdim=True)
    q = q / q.sum(dim=-1, keepdim=True)
    kl = (p * (p.log() - q.log())).sum(dim=-1).mean()
    return kl.item()


def attention_js_divergence(
    attn_fp: torch.Tensor,
    attn_quant: torch.Tensor,
    eps: float = 1e-8,
) -> float:
    """Jensen-Shannon divergence (symmetric, bounded)。"""
    p = attn_fp.float().clamp(min=eps)
    q = attn_quant.float().clamp(min=eps)
    p = p / p.sum(dim=-1, keepdim=True)
    q = q / q.sum(dim=-1, keepdim=True)
    m = 0.5 * (p + q)
    js = 0.5 * (p * (p.log() - m.log())).sum(dim=-1).mean() + \
         0.5 * (q * (q.log() - m.log())).sum(dim=-1).mean()
    return js.item()


def attention_cosine_sim(
    attn_fp: torch.Tensor,
    attn_quant: torch.Tensor,
) -> float:
    """逐头逐行的 cosine similarity 再平均。"""
    fp_flat = attn_fp.float().reshape(attn_fp.shape[0], attn_fp.shape[1], -1)
    qt_flat = attn_quant.float().reshape(attn_quant.shape[0], attn_quant.shape[1], -1)
    sim = F.cosine_similarity(fp_flat, qt_flat, dim=-1)
    return sim.mean().item()


def compute_vision_saliency(
    attentions: Tuple[torch.Tensor],
    vision_mask: torch.Tensor,
    layer_indices: Optional[List[int]] = None,
) -> torch.Tensor:
    """从 attention 矩阵中提取视觉 token 的 saliency 分数。

    对指定层的 attention，计算所有 token 对视觉 token 位置的注意力之和，
    可以 reshape 为 2D 空间 saliency map。

    Returns:
        saliency: [num_vision_tokens] 每个视觉 token 被关注的程度
    """
    if layer_indices is None:
        layer_indices = list(range(len(attentions)))

    saliency = None
    for li in layer_indices:
        attn = attentions[li]  # [B, H, S, S]
        if attn is None:
            continue
        attn_to_vision = attn[:, :, :, vision_mask].mean(dim=1)  # [B, S, V]
        attn_to_vision = attn_to_vision.mean(dim=1)  # [B, V]
        if saliency is None:
            saliency = attn_to_vision
        else:
            saliency = saliency + attn_to_vision

    if saliency is None:
        return torch.zeros(vision_mask.sum().item())
    return saliency.squeeze(0)  # [V]


# ================================================================
# Algorithm Pool & Utilities (same as Stage 21)
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


def _s24_algo(name, desc, **overrides):
    cfg = dict(_W4_BASE)
    cfg.update(overrides)
    return {'name': name, 'description': desc, 'config': cfg}


def build_stage24_pool():
    pool = {}
    pool['gptq_w4a4'] = _s24_algo('GPTQ W4A4', 'GPTQ g=64', use_gptq=True, gptq_group_size=64)
    for alpha in [0.5, 0.7, 0.85]:
        tag = f'a{int(alpha*100)}'
        pool[f'smooth_{tag}_gptq_w4a4'] = _s24_algo(
            f'Smooth(α={alpha})+GPTQ', f'SQ α={alpha}, GPTQ g=64',
            use_smoothquant=True, smoothquant_alpha=alpha, use_gptq=True, gptq_group_size=64)
    for alpha in [0.5, 0.7, 0.85]:
        tag = f'a{int(alpha*100)}'
        pool[f'svdquant_{tag}_w4a4'] = _s24_algo(
            f'SVDQuant α={alpha}', f'SQ+SVD+GPTQ',
            use_smoothquant=True, smoothquant_alpha=alpha,
            use_svd=True, svd_rank=32, use_gptq=True, gptq_group_size=64)
    pool['svd_gptq_w4a4'] = _s24_algo('SVD+GPTQ', 'SVD r=32, GPTQ', use_svd=True, svd_rank=32, use_gptq=True, gptq_group_size=64)
    pool['awq_svd_rtn_w4a4'] = _s24_algo('AWQ+SVD+RTN', 'AWQ+SVD+RTN', use_awq=True, awq_n_grid=20, use_svd=True, svd_rank=32)
    return pool


FUNCTIONAL_GROUPS = [
    {'name': 'attn', 'display': 'Attention (Q/K/V/O)',
     'suffixes': ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj'],
     'hook_target': 'self_attn'},
    {'name': 'mlp', 'display': 'MLP (gate/up/down)',
     'suffixes': ['mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj'],
     'hook_target': 'mlp'},
]


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
        if smoothquant_stats and Path(smoothquant_stats).exists():
            self.smooth_data = torch.load(smoothquant_stats, map_location="cpu")
        if awq_stats and Path(awq_stats).exists():
            self.awq_data = torch.load(awq_stats, map_location="cpu")
        self._legacy = None
        if legacy_activation_file and Path(legacy_activation_file).exists():
            self._legacy = torch.load(legacy_activation_file, map_location="cpu")
        self.available = (self.hessian_index is not None or self._legacy is not None)

    def __contains__(self, name):
        return (self._legacy and name in self._legacy) or (self.hessian_index and name in self.hessian_index)

    def get_activation_max(self, name):
        if self.smooth_data and name in self.smooth_data:
            return self.smooth_data[name].get("act_channel_max")
        if self.awq_data and name in self.awq_data:
            return self.awq_data[name].get("channel_max")
        return None

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


class LargeCalibrationLoader:
    def __init__(self, dataset_path, max_samples=None):
        self.dataset_path = Path(dataset_path)
        self.max_samples = max_samples

    def load(self):
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        raw = data.get("samples", data)
        if isinstance(raw, dict):
            raw = list(raw.values())
        samples = []
        for s in raw:
            img = None
            ip = s.get("image_path")
            if ip and Path(ip).exists():
                try:
                    img = Image.open(ip).convert("RGB")
                except Exception:
                    pass
            samples.append({
                "task_type": "und", "prompt": s.get("question", s.get("prompt", "")),
                "image": img, "generation_mode": "text",
                "question_type": s.get("question_type", "unknown"),
            })
        if self.max_samples and len(samples) > self.max_samples:
            import random
            random.shuffle(samples)
            samples = samples[:self.max_samples]
        return samples


# ================================================================
# Stage 24 Searcher
# ================================================================

class AttentionFidelitySearcher:
    """Stage 24: Attention Fidelity 分析与搜索。

    使用 eager attention 以获取完整 attention weights，
    联合 CKA + attention divergence 作为搜索指标。
    """

    _FULL_CONFIG_TEMPLATE = {
        "weight_bit": 4, "act_bit": 4, "quant_percentile": 0.999999,
        "act_unsigned": True, "use_sparse": False, "sparse_ratio": 0.0,
        "sparse_threshold": None, "use_smoothquant": False,
        "smoothquant_alpha": 0.5, "use_svd": False, "svd_rank": 0,
        "use_gptq": False, "gptq_group_size": 64,
        "gptq_damp_percentage": 0.01, "gptq_block_size": 128,
        "use_block_quant": False, "use_block_quant_act": False,
        "block_size_weight": 256, "block_size_act": 256,
        "use_awq": False, "awq_alpha": 0.5, "awq_n_grid": 20,
    }

    def __init__(
        self,
        model_path: str,
        output_dir: str = "./quantization_outputs/stage24_attention_fidelity",
        calibration_dataset: Optional[str] = None,
        algorithm_pool: Optional[Dict] = None,
        gptq_hessian_index: Optional[str] = None,
        smoothquant_stats: Optional[str] = None,
        awq_stats: Optional[str] = None,
        activation_data_file: Optional[str] = None,
        gpu_ids: Optional[str] = None,
        target_decoder_layers: Optional[List[int]] = None,
        seed: int = 42,
        subsample_step: int = 5,
        max_calib_samples: Optional[int] = None,
        cka_num_samples: int = 50,
        cka_weight: float = 0.6,
        attn_weight: float = 0.4,
        saliency_layers: Optional[List[int]] = None,
        run_date: Optional[str] = None,
    ):
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.calibration_dataset = calibration_dataset
        self.algorithm_pool = algorithm_pool or build_stage24_pool()
        self.functional_groups = FUNCTIONAL_GROUPS
        self.gpu_ids = gpu_ids
        self.target_decoder_layers = target_decoder_layers
        self.seed = seed
        self.subsample_step = subsample_step
        self.max_calib_samples = max_calib_samples
        self.cka_num_samples = cka_num_samples
        self.cka_weight = cka_weight
        self.attn_weight = attn_weight
        self.saliency_layers = saliency_layers
        self.run_date = run_date or datetime.now().strftime("%Y%m%d")

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
        if self.saliency_layers is None:
            self.saliency_layers = self.target_decoder_layers[-4:]

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

        self._switch_to_eager_attention()

    def _switch_to_eager_attention(self):
        """将 LLM 的 attention 从 flash_attention_2 切换到 eager 模式。"""
        llm = self.model.language_model
        if hasattr(llm.config, '_attn_implementation'):
            llm.config._attn_implementation = 'eager'
        if hasattr(llm.config, 'attn_implementation'):
            llm.config.attn_implementation = 'eager'

        for layer in llm.model.layers:
            attn = layer.self_attn
            if hasattr(attn, '_attn_implementation'):
                attn._attn_implementation = 'eager'
            if hasattr(attn, 'config') and hasattr(attn.config, '_attn_implementation'):
                attn.config._attn_implementation = 'eager'

        print("  Switched LLM to eager attention mode for attention weight extraction")

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
        print("Stage 24 (InternVL-U): Attention Fidelity Analysis & Search")
        print("=" * 80)
        print(f"  Model: {self.model_path}")
        print(f"  Attention mode: eager (for attention weight extraction)")
        print(f"  CKA weight: {self.cka_weight}, Attention weight: {self.attn_weight}")
        print(f"  CKA samples: {self.cka_num_samples}")
        print(f"  Saliency layers: {self.saliency_layers}")
        print(f"  Output: {self.output_dir}")
        print("=" * 80 + "\n")

    # ---- Module helpers ----

    def _get_decoder_layer_linear_names(self, layer_idx):
        prefix = f"language_model.model.layers.{layer_idx}"
        decoder_layer = self.model.language_model.model.layers[layer_idx]
        return [f"{prefix}.{n}" for n, m in decoder_layer.named_modules()
                if isinstance(m, (nn.Linear, HybridQuantLinear)) and n]

    def _get_group_sublayer_names(self, layer_idx, group):
        prefix = f"language_model.model.layers.{layer_idx}"
        return [f"{prefix}.{s}" for s in group['suffixes']]

    def _get_module(self, name):
        parts = name.split(".")
        m = self.model
        for p in parts:
            if hasattr(m, p):
                m = getattr(m, p)
            else:
                return None
        return m

    def _replace_module(self, name, new_mod):
        parts = name.split(".")
        parent = self.model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], new_mod)

    # ---- Quantization apply/restore ----

    def _apply_algorithm_to_layer(self, layer_name, algo_config):
        module = self._get_module(layer_name)
        if module is None or not isinstance(module, (nn.Linear, HybridQuantLinear)):
            return
        if isinstance(module, HybridQuantLinear):
            self._restore_layer(layer_name)
            module = self._get_module(layer_name)
        device = module.weight.device
        dtype = module.weight.dtype
        config = algo_config.copy()
        if config.get("use_gptq") and not self.activation_provider.available:
            config["use_gptq"] = False
        if config.get("use_awq") and not self.activation_provider.available:
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
        if layer_name in self.original_weights:
            orig_w = self.original_weights[layer_name]["weight"].clone()
            orig_b = self.original_weights[layer_name]["bias"]
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
        if self.activation_provider.available and layer_name in self.activation_provider:
            act_data = self.activation_provider.get_activation(layer_name)
        act_max = self.activation_provider.get_activation_max(layer_name)
        quant_layer.prepare_weight(
            activation_max=act_max, activation_data=act_data,
            layer_name=layer_name, verbose=False,
        )
        self._replace_module(layer_name, quant_layer)
        del act_data, act_max
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _restore_layer(self, layer_name):
        module = self._get_module(layer_name)
        if module is None:
            return
        if isinstance(module, HybridQuantLinear):
            device = module.weight.device if module.weight is not None else "cuda:0"
            dtype = module.weight.dtype if module.weight is not None else torch.bfloat16
            linear = nn.Linear(module.in_features, module.out_features,
                               bias=module.bias is not None, device=device, dtype=dtype)
            if layer_name in self.original_weights:
                linear.weight.data.copy_(self.original_weights[layer_name]["weight"].to(device))
                if linear.bias is not None and self.original_weights[layer_name]["bias"] is not None:
                    linear.bias.data.copy_(self.original_weights[layer_name]["bias"].to(device))
            self._replace_module(layer_name, linear)

    def _apply_algorithm_to_group(self, layer_idx, group, algo_config):
        for name in self._get_group_sublayer_names(layer_idx, group):
            self._apply_algorithm_to_layer(name, algo_config)

    def _restore_group(self, layer_idx, group):
        for name in self._get_group_sublayer_names(layer_idx, group):
            self._restore_layer(name)

    def _restore_decoder_layer(self, layer_idx):
        for name in self._get_decoder_layer_linear_names(layer_idx):
            self._restore_layer(name)

    def _redispatch_model(self):
        try:
            from accelerate import dispatch_model
            if hasattr(self.model, "hf_device_map"):
                self.model = dispatch_model(self.model, device_map=self.model.hf_device_map)
        except Exception:
            pass

    # ---- Forward with attention output ----

    def _forward_with_attention(self, sample: Dict) -> Optional[Dict]:
        """前向一条样本，返回 hidden_states + attention weights。"""
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
        pv = inputs.get("pixel_values")
        if gen_mode == "image":
            pv = None

        assert self.model.img_context_token_id is not None
        input_embeds = self.model.language_model.get_input_embeddings()(input_ids)
        input_embeds = self.model.replace_img_special_tokens(input_embeds, input_ids)
        B, N, C = input_embeds.shape

        if pv is not None:
            vit_embeds = self.model.extract_feature(pv)
            input_embeds = input_embeds.reshape(B * N, C)
            ids_flat = input_ids.reshape(B * N)
            selected = ids_flat == self.model.img_context_token_id
            if selected.sum() > 0:
                input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)
            input_embeds = input_embeds.reshape(B, N, C)

        with torch.no_grad():
            outputs = self.model.language_model(
                inputs_embeds=input_embeds,
                attention_mask=inputs.get("attention_mask"),
                output_attentions=True,
                output_hidden_states=True,
                return_dict=True,
            )

        vision_mask = (input_ids.reshape(-1) == self.model.img_context_token_id).cpu()

        attentions_cpu = []
        if outputs.attentions:
            for attn in outputs.attentions:
                if attn is not None:
                    attentions_cpu.append(attn.detach().cpu())
                else:
                    attentions_cpu.append(None)

        return {
            "attentions": tuple(attentions_cpu),
            "vision_mask": vision_mask,
            "input_ids": input_ids.detach().cpu(),
            "seq_len": N,
        }

    def _collect_group_hidden_states(self, layer_idx, group, samples):
        hook_target_name = group.get('hook_target')
        decoder_layer = self.model.language_model.model.layers[layer_idx]
        target = getattr(decoder_layer, hook_target_name, decoder_layer) if hook_target_name else decoder_layer
        hs_list = []
        captured = {}

        def hook_fn(module, inp, output):
            h = output[0] if isinstance(output, tuple) else output
            captured["output"] = h.detach().cpu()

        handle = target.register_forward_hook(hook_fn)
        try:
            for sample in samples:
                try:
                    prompt = sample.get("prompt", "")
                    gen_mode = sample.get("generation_mode", "text")
                    pil_image = image_for_processor(sample.get("image"))
                    inputs = self.processor(
                        prompt=[prompt], image=[pil_image] if pil_image else [None],
                        generation_mode=gen_mode, padding=True, return_tensors="pt",
                    )
                    device = next(self.model.parameters()).device
                    for k, v in inputs.items():
                        if isinstance(v, torch.Tensor):
                            inputs[k] = v.to(device)
                    with torch.no_grad():
                        pv = inputs.get("pixel_values")
                        if gen_mode == "image":
                            pv = None
                        self.model.generate_hidden_states(
                            pixel_values=pv, input_ids=inputs["input_ids"],
                            attention_mask=inputs.get("attention_mask"),
                        )
                except Exception:
                    captured.clear()
                    continue
                if "output" in captured:
                    hs_list.append(captured["output"])
                captured.clear()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        finally:
            handle.remove()
        return hs_list

    def _build_full_layer_config(self, algo_key):
        cfg = dict(self._FULL_CONFIG_TEMPLATE)
        cfg.update(self.algorithm_pool[algo_key]["config"])
        if not cfg.get("use_svd"):
            cfg["svd_rank"] = 0
        if not cfg.get("use_sparse"):
            cfg["sparse_threshold"] = None
            cfg["sparse_ratio"] = 0.0
        return cfg

    def _subsample(self, samples):
        """与 Stage 21/23 相同的分层子采样，确保选到的 200 条一致。"""
        import random
        n = self.cka_num_samples
        if len(samples) <= n:
            return samples
        by_type = {}
        for s in samples:
            qt = s.get("question_type", "unknown")
            by_type.setdefault(qt, []).append(s)
        rng = random.Random(self.seed)
        for v in by_type.values():
            rng.shuffle(v)
        selected = []
        types = sorted(by_type.keys())
        per_type = max(1, n // len(types))
        for t in types:
            selected.extend(by_type[t][:per_type])
        remaining = n - len(selected)
        if remaining > 0:
            pool = [s for t in types for s in by_type[t][per_type:]]
            rng.shuffle(pool)
            selected.extend(pool[:remaining])
        selected = selected[:n]
        rng.shuffle(selected)
        return selected

    # ---- Main pipeline ----

    def run(self) -> Dict:
        """执行完整的 Attention Fidelity 流程。"""

        # Phase 1: Load data
        print("\n" + "=" * 80)
        print("Phase 1: Loading calibration data")
        print("=" * 80)
        loader = LargeCalibrationLoader(self.calibration_dataset, self.max_calib_samples)
        all_samples = loader.load()
        cka_samples = self._subsample(all_samples)
        img_samples = [s for s in cka_samples if s.get("image") is not None]
        print(f"  Total: {len(all_samples)}, CKA: {len(cka_samples)}, With images: {len(img_samples)}")

        available_algos = {}
        for key, algo in self.algorithm_pool.items():
            if algo["config"].get("use_gptq") and not self.activation_provider.available:
                continue
            if algo["config"].get("use_awq") and not self.activation_provider.available:
                continue
            available_algos[key] = algo

        # ---- Phase A: Attention Divergence Analysis ----
        print("\n" + "=" * 80)
        print("Phase A: Attention Divergence Analysis (FP16 vs Int4)")
        print("=" * 80)

        analysis_samples = img_samples if img_samples else cka_samples
        if not img_samples:
            print(f"  [WARN] No image samples found! Using {len(analysis_samples)} text samples for attention analysis.")
        attn_analysis = {}

        sample_layers = list(self.target_decoder_layers)

        for sample_idx, sample in enumerate(analysis_samples):
            try:
                print(f"\n  Sample {sample_idx}: collecting FP16 attentions ...")
                fp_result = self._forward_with_attention(sample)
                if fp_result is None:
                    print(f"    [WARN] FP16 forward returned None, skipping")
                    continue

                fp_attn_count = sum(1 for a in fp_result["attentions"] if a is not None)
                print(f"    FP16 attentions collected: {fp_attn_count} layers, "
                      f"seq_len={fp_result['seq_len']}, "
                      f"vision_tokens={fp_result['vision_mask'].sum().item()}")

                default_int4 = {"weight_bit": 4, "act_bit": 4, "act_unsigned": True}
                for layer_idx in sample_layers:
                    for name in self._get_decoder_layer_linear_names(layer_idx):
                        self._apply_algorithm_to_layer(name, default_int4)
                self._redispatch_model()

                print(f"  Sample {sample_idx}: collecting Int4 attentions ...")
                quant_result = self._forward_with_attention(sample)

                for layer_idx in self.target_decoder_layers:
                    self._restore_decoder_layer(layer_idx)

                if quant_result is None:
                    print(f"    [WARN] Quant forward returned None, skipping")
                    continue

                qt_attn_count = sum(1 for a in quant_result["attentions"] if a is not None)
                print(f"    Int4 attentions collected: {qt_attn_count} layers")

                for li in sample_layers:
                    if li < len(fp_result["attentions"]) and li < len(quant_result["attentions"]):
                        fp_attn = fp_result["attentions"][li]
                        qt_attn = quant_result["attentions"][li]
                        if fp_attn is not None and qt_attn is not None:
                            fp_a = fp_attn.squeeze(0)
                            qt_a = qt_attn.squeeze(0)
                            n = min(fp_a.shape[-1], qt_a.shape[-1])
                            fp_a = fp_a[:, :n, :n]
                            qt_a = qt_a[:, :n, :n]
                            kl = attention_kl_divergence(fp_a, qt_a)
                            js = attention_js_divergence(fp_a, qt_a)
                            cos = attention_cosine_sim(fp_a, qt_a)
                            key = f"sample{sample_idx}_layer{li}"
                            attn_analysis[key] = {
                                "kl_divergence": round(kl, 6),
                                "js_divergence": round(js, 6),
                                "cosine_similarity": round(cos, 6),
                            }
                            print(f"    Layer {li}: KL={kl:.4f} JS={js:.4f} CosSim={cos:.4f}")
                        else:
                            which = "FP" if fp_attn is None else "Quant"
                            print(f"    Layer {li}: {which} attention is None, skipping")

                del fp_result, quant_result
            except Exception as e:
                print(f"    [ERROR] Phase A sample {sample_idx} failed: {e}")
                import traceback
                traceback.print_exc()
                for layer_idx in self.target_decoder_layers:
                    self._restore_decoder_layer(layer_idx)

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if not attn_analysis:
            print("  [WARN] Phase A produced no attention analysis data.")
            print("         Possible causes: (1) eager attention not returning weights,")
            print("         (2) OOM during forward, (3) no valid samples.")

        # incremental save: Phase A results
        phaseA_path = self.output_dir / f"stage24_phaseA_{self.run_date}.json"
        with open(phaseA_path, "w") as f:
            json.dump({"attention_analysis": attn_analysis}, f, indent=2)
        print(f"  Phase A saved: {phaseA_path}")

        # ---- Phase B: Attention-Fidelity Guided Search ----
        print("\n" + "=" * 80)
        print("Phase B: Attention-Fidelity Functional-Group Search")
        print("=" * 80)

        fallback = list(available_algos.keys())[0]
        group_assignments = {}
        search_log = []
        config_export_dir = Path(self.output_dir).parent / "configs"
        config_export_dir.mkdir(parents=True, exist_ok=True)
        progress_file = self.output_dir / f"search_progress_{self.run_date}.json"
        partial_config_path = config_export_dir / f"stage24_attn_fidelity_w4a4_{self.run_date}_partial.json"

        attn_analysis_samples = img_samples if img_samples else cka_samples[:min(20, len(cka_samples))]

        for layer_idx in tqdm(self.target_decoder_layers, desc="Attn-fidelity search"):
            for group in self.functional_groups:
                sublayers = self._get_group_sublayer_names(layer_idx, group)
                existing = [n for n in sublayers if self._get_module(n) is not None]
                if not existing:
                    continue

                ref_hs = self._collect_group_hidden_states(layer_idx, group, cka_samples)
                if not ref_hs:
                    for ln in existing:
                        group_assignments[ln] = fallback
                    continue

                fp_attns = []
                if group['name'] == 'attn':
                    for s in attn_analysis_samples:
                        r = self._forward_with_attention(s)
                        if r and layer_idx < len(r["attentions"]) and r["attentions"][layer_idx] is not None:
                            fp_attns.append(r["attentions"][layer_idx].squeeze(0))

                best_algo = None
                best_score = -1.0
                algo_detail = {}

                for algo_key, algo_info in available_algos.items():
                    try:
                        self._apply_algorithm_to_group(layer_idx, group, algo_info["config"])
                        self._redispatch_model()

                        quant_hs = self._collect_group_hidden_states(layer_idx, group, cka_samples)
                        cka = LinearCKA.compute_batched(
                            ref_hs, quant_hs, subsample_step=self.subsample_step
                        ) if quant_hs else 0.0

                        attn_fidelity = 1.0
                        if group['name'] == 'attn' and fp_attns:
                            js_scores = []
                            for si, s in enumerate(attn_analysis_samples):
                                if si >= len(fp_attns):
                                    break
                                r = self._forward_with_attention(s)
                                if r and layer_idx < len(r["attentions"]) and r["attentions"][layer_idx] is not None:
                                    qt_a = r["attentions"][layer_idx].squeeze(0)
                                    fp_a = fp_attns[si]
                                    n = min(fp_a.shape[-1], qt_a.shape[-1])
                                    js = attention_js_divergence(fp_a[:, :n, :n], qt_a[:, :n, :n])
                                    js_scores.append(js)
                            if js_scores:
                                avg_js = np.mean(js_scores)
                                attn_fidelity = 1.0 / (1.0 + avg_js * 10)

                        score = self.cka_weight * cka + self.attn_weight * attn_fidelity
                        algo_detail[algo_key] = {
                            "combined": round(score, 6), "cka": round(cka, 6),
                            "attn_fidelity": round(attn_fidelity, 6),
                        }

                        if score > best_score:
                            best_score = score
                            best_algo = algo_key

                    except Exception as e:
                        algo_detail[algo_key] = {"combined": -1.0, "error": str(e)}
                    finally:
                        self._restore_group(layer_idx, group)
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                if best_algo is None:
                    best_algo = fallback

                for ln in existing:
                    group_assignments[ln] = best_algo
                self._apply_algorithm_to_group(layer_idx, group, available_algos[best_algo]["config"])
                self._redispatch_model()

                search_log.append({
                    'layer_idx': layer_idx, 'group': group['name'],
                    'best_algo': best_algo, 'best_score': round(best_score, 6),
                    'detail': algo_detail,
                })

                del ref_hs, fp_attns
                gc.collect()

            # incremental save: progress + partial config
            partial_cfg = {}
            for sn, ak in group_assignments.items():
                partial_cfg[sn] = self._build_full_layer_config(ak)
            with open(progress_file, 'w') as f:
                json.dump({
                    'search_log': search_log,
                    'group_assignments': group_assignments,
                    'completed_layers': layer_idx + 1,
                    'total_layers': len(self.target_decoder_layers),
                }, f, indent=2)
            with open(partial_config_path, 'w') as f:
                json.dump(dict(sorted(partial_cfg.items())), f, indent=2)
            self.activation_provider.clear_cache()

        # ---- Phase C: Saliency Map Generation ----
        print("\n" + "=" * 80)
        print("Phase C: Saliency Map Generation")
        print("=" * 80)

        saliency_results = {}
        saliency_dir = self.output_dir / "saliency_maps"
        saliency_dir.mkdir(exist_ok=True)

        for si, sample in enumerate(img_samples):
            print(f"  Generating saliency for sample {si} ...")
            result = self._forward_with_attention(sample)
            if result is None or not result["vision_mask"].any():
                continue

            saliency = compute_vision_saliency(
                result["attentions"], result["vision_mask"],
                layer_indices=self.saliency_layers,
            )
            saliency_np = saliency.float().numpy()
            saliency_results[f"sample_{si}"] = {
                "saliency_stats": {
                    "mean": round(float(saliency_np.mean()), 6),
                    "std": round(float(saliency_np.std()), 6),
                    "max": round(float(saliency_np.max()), 6),
                    "min": round(float(saliency_np.min()), 6),
                },
                "num_vision_tokens": int(result["vision_mask"].sum().item()),
                "saliency_layers": self.saliency_layers,
            }

            np.save(saliency_dir / f"saliency_sample{si}.npy", saliency_np)
            print(f"    Saved saliency map: {saliency_np.shape}, "
                  f"range=[{saliency_np.min():.4f}, {saliency_np.max():.4f}]")

        # ---- Export ----
        print("\n" + "=" * 80)
        print("Exporting results")
        print("=" * 80)

        export_cfg = {}
        for sublayer_name, algo_key in group_assignments.items():
            export_cfg[sublayer_name] = self._build_full_layer_config(algo_key)
        config_path = config_export_dir / f"stage24_attn_fidelity_w4a4_{self.run_date}.json"
        with open(config_path, "w") as f:
            json.dump(dict(sorted(export_cfg.items())), f, indent=2)

        results = {
            "attention_analysis": attn_analysis,
            "search_results": {
                "group_assignments": group_assignments,
                "search_log": search_log,
                "config_path": str(config_path),
            },
            "saliency": saliency_results,
            "metadata": {
                "stage": 24,
                "strategy": "attention_fidelity_funcgroup",
                "cka_weight": self.cka_weight,
                "attn_weight": self.attn_weight,
                "attention_mode": "eager",
                "model_path": self.model_path,
                "run_date": self.run_date,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            },
        }
        results_path = self.output_dir / f"stage24_results_{self.run_date}.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"  Config: {config_path}")
        print(f"  Results: {results_path}")
        print(f"  Saliency maps: {saliency_dir}")
        return results

    def run_phase_c_only(self, quant_config_path: str) -> Dict:
        """只运行 Phase C (Saliency Map)，跳过 Phase A/B。

        加载已有量化配置，应用到模型后生成 saliency maps。
        """
        print("\n" + "=" * 80)
        print("Phase C Only: Loading quantization config & generating saliency maps")
        print("=" * 80)

        with open(quant_config_path, "r") as f:
            quant_cfg = json.load(f)
        print(f"  Config: {quant_config_path} ({len(quant_cfg)} sublayers)")

        for layer_name, cfg in quant_cfg.items():
            self._apply_algorithm_to_layer(layer_name, cfg)
        self._redispatch_model()
        print("  Quantization config applied to model")

        loader = LargeCalibrationLoader(self.calibration_dataset, self.max_calib_samples)
        all_samples = loader.load()
        img_samples = [s for s in all_samples if s.get("image") is not None]
        print(f"  Total samples: {len(all_samples)}, with images: {len(img_samples)}")

        if not img_samples:
            print("  [ERROR] No image samples found! Cannot generate saliency maps.")
            return {}

        saliency_results = {}
        saliency_dir = self.output_dir / "saliency_maps"
        saliency_dir.mkdir(exist_ok=True)

        max_saliency = min(len(img_samples), 20)
        for si, sample in enumerate(img_samples[:max_saliency]):
            print(f"  [{si+1}/{max_saliency}] Generating saliency ...")
            try:
                result = self._forward_with_attention(sample)
                if result is None or not result["vision_mask"].any():
                    print(f"    Skipped (no vision tokens)")
                    continue

                saliency = compute_vision_saliency(
                    result["attentions"], result["vision_mask"],
                    layer_indices=self.saliency_layers,
                )
                saliency_np = saliency.float().numpy()
                saliency_results[f"sample_{si}"] = {
                    "saliency_stats": {
                        "mean": round(float(saliency_np.mean()), 6),
                        "std": round(float(saliency_np.std()), 6),
                        "max": round(float(saliency_np.max()), 6),
                        "min": round(float(saliency_np.min()), 6),
                    },
                    "num_vision_tokens": int(result["vision_mask"].sum().item()),
                    "saliency_layers": self.saliency_layers,
                }
                np.save(saliency_dir / f"saliency_sample{si}.npy", saliency_np)
                print(f"    Saved: shape={saliency_np.shape}, "
                      f"range=[{saliency_np.min():.4f}, {saliency_np.max():.4f}]")

                del result
            except Exception as e:
                print(f"    [ERROR] Sample {si} failed: {e}")
                import traceback
                traceback.print_exc()

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        results_path = self.output_dir / f"stage24_results_{self.run_date}.json"
        if results_path.exists():
            with open(results_path) as f:
                existing = json.load(f)
            existing["saliency"] = saliency_results
            with open(results_path, "w") as f:
                json.dump(existing, f, indent=2)
            print(f"\n  Updated saliency in: {results_path}")
        else:
            with open(results_path, "w") as f:
                json.dump({"saliency": saliency_results}, f, indent=2)

        print(f"  Saliency maps: {saliency_dir} ({len(saliency_results)} maps)")
        return saliency_results


# ================================================================
# CLI
# ================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Stage 24 (InternVL-U): Attention Fidelity Analysis & Search"
    )
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str,
                        default="./quantization_outputs/stage24_attention_fidelity")
    parser.add_argument("--calibration_dataset", type=str, required=True)
    parser.add_argument("--gptq_hessian_index", type=str, default=None)
    parser.add_argument("--smoothquant_stats", type=str, default=None)
    parser.add_argument("--awq_stats", type=str, default=None)
    parser.add_argument("--activation_data", type=str, default=None)
    parser.add_argument("--gpu_ids", type=str, default=None)
    parser.add_argument("--target_layers", type=str, default=None)
    parser.add_argument("--max_calib_samples", type=int, default=None)
    parser.add_argument("--cka_num_samples", type=int, default=200,
                        help="Number of samples for CKA computation (default: 200, same as Stage 21/23)")
    parser.add_argument("--cka_weight", type=float, default=0.6)
    parser.add_argument("--attn_weight", type=float, default=0.4)
    parser.add_argument("--subsample_step", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_date", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--phase_c_only", action="store_true",
                        help="Skip Phase A/B, only run Phase C (saliency maps) using existing quant config")
    parser.add_argument("--quant_config", type=str, default=None,
                        help="Path to quantization config JSON (required for --phase_c_only)")

    args = parser.parse_args()
    if args.config:
        with open(args.config, "r") as f:
            for k, v in yaml.safe_load(f).items():
                if hasattr(args, k) and getattr(args, k) is None:
                    setattr(args, k, v)

    target_layers = None
    if args.target_layers:
        target_layers = [int(x.strip()) for x in args.target_layers.split(",")]

    searcher = AttentionFidelitySearcher(
        model_path=args.model_path,
        output_dir=args.output_dir,
        calibration_dataset=args.calibration_dataset,
        gptq_hessian_index=args.gptq_hessian_index,
        smoothquant_stats=args.smoothquant_stats,
        awq_stats=args.awq_stats,
        activation_data_file=args.activation_data,
        gpu_ids=args.gpu_ids,
        target_decoder_layers=target_layers,
        seed=args.seed,
        subsample_step=args.subsample_step,
        max_calib_samples=args.max_calib_samples,
        cka_num_samples=args.cka_num_samples,
        cka_weight=args.cka_weight,
        attn_weight=args.attn_weight,
        run_date=args.run_date,
    )

    if args.phase_c_only:
        qcfg = args.quant_config
        if qcfg is None:
            cfg_dir = Path(args.output_dir).parent / "configs"
            candidates = sorted(cfg_dir.glob("stage24_attn_fidelity_w4a4_*.json"),
                                key=lambda p: p.stat().st_mtime, reverse=True)
            candidates = [c for c in candidates if "partial" not in c.name]
            if candidates:
                qcfg = str(candidates[0])
                print(f"  Auto-detected config: {qcfg}")
            else:
                print("[ERROR] No --quant_config provided and no config found in configs/")
                sys.exit(1)
        searcher.run_phase_c_only(qcfg)
        print("\nPhase C completed!")
    else:
        results = searcher.run()
        print("\nStage 24 completed!")


if __name__ == "__main__":
    main()
