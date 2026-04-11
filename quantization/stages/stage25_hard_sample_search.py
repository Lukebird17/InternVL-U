"""
Stage 25 (InternVL-U): Hard-Sample Aware Calibration Search

方向 3：基于"硬样本"的动态校准

核心假设：
  使用"易混淆、低置信度"的硬样本作为校准集，
  能搜出更鲁棒的混合量化策略。

两个子阶段：
  Phase 0 — 硬样本筛选
    用 FP16 模型对校准集跑推理，收集 token 级困惑度(perplexity)，
    按困惑度排序筛选出"低置信度"硬样本。

  Phase 1 — 对比搜索
    分别用随机校准集和硬样本校准集跑 functional group 搜索，
    对比产出的量化策略差异。

基于 Stage 21 的 functional group 搜索粒度 + Flickr8K 校准集。
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

from utils.bagel_imports import HybridQuantLinear, LinearCKA
from utils.model_loader import load_internvlu
from utils.calibration import image_for_processor


# ================================================================
# Algorithm Pool & Functional Groups (same as Stage 21)
# ================================================================

FUNCTIONAL_GROUPS = [
    {'name': 'attn', 'display': 'Attention (Q/K/V/O)',
     'suffixes': ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj'],
     'hook_target': 'self_attn'},
    {'name': 'mlp', 'display': 'MLP (gate/up/down)',
     'suffixes': ['mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj'],
     'hook_target': 'mlp'},
]

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


def _s25_algo(name, desc, **overrides):
    cfg = dict(_W4_BASE)
    cfg.update(overrides)
    return {'name': name, 'description': desc, 'config': cfg}


def build_stage25_pool():
    pool = {}
    pool['gptq_w4a4'] = _s25_algo('GPTQ W4A4', 'GPTQ g=64', use_gptq=True, gptq_group_size=64)
    for alpha in [0.5, 0.7, 0.85]:
        tag = f'a{int(alpha*100)}'
        pool[f'smooth_{tag}_gptq_w4a4'] = _s25_algo(
            f'Smooth(α={alpha})+GPTQ', f'SQ+GPTQ',
            use_smoothquant=True, smoothquant_alpha=alpha,
            use_gptq=True, gptq_group_size=64)
    for alpha in [0.5, 0.7, 0.85]:
        tag = f'a{int(alpha*100)}'
        pool[f'svdquant_{tag}_w4a4'] = _s25_algo(
            f'SVDQuant α={alpha}', f'SQ+SVD+GPTQ',
            use_smoothquant=True, smoothquant_alpha=alpha,
            use_svd=True, svd_rank=32, use_gptq=True, gptq_group_size=64)
    pool['svd_gptq_w4a4'] = _s25_algo('SVD+GPTQ', 'SVD+GPTQ', use_svd=True, svd_rank=32, use_gptq=True, gptq_group_size=64)
    pool['awq_svd_rtn_w4a4'] = _s25_algo('AWQ+SVD+RTN', 'AWQ+SVD+RTN', use_awq=True, awq_n_grid=20, use_svd=True, svd_rank=32)
    return pool


# ================================================================
# Shared Utilities
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
        img_count = sum(1 for s in samples if s["image"] is not None)
        print(f"    Loaded {len(samples)} samples ({img_count} with images)")
        return samples


# ================================================================
# Hard Sample Selector
# ================================================================

class HardSampleSelector:
    """基于模型困惑度筛选硬样本。

    对每条校准样本计算 token 级 perplexity，
    选出困惑度最高的 top-K 作为"硬样本"。
    """

    def __init__(self, model, processor, tokenizer):
        self.model = model
        self.processor = processor
        self.tokenizer = tokenizer

    @torch.no_grad()
    def compute_sample_difficulty(self, sample: Dict) -> float:
        """计算单条样本的困难度（token 级 perplexity）。

        只在真实文本 token 上计算 CE loss，排除 <IMG_CONTEXT> 和 pad。
        """
        prompt = sample.get("prompt", "")
        gen_mode = sample.get("generation_mode", "text")
        pil_image = image_for_processor(sample.get("image"))

        try:
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

            outputs = self.model.generate_hidden_states(
                pixel_values=pv,
                input_ids=input_ids,
                attention_mask=inputs.get("attention_mask"),
            )

            logits = outputs.logits  # [B, S, V]
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()

            loss_fn = nn.CrossEntropyLoss(reduction='none')
            losses = loss_fn(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
            )

            flat_labels = shift_labels.reshape(-1)
            valid_mask = (flat_labels != self.tokenizer.pad_token_id)
            img_token_id = getattr(self.model, 'img_context_token_id', None)
            if img_token_id is not None:
                valid_mask = valid_mask & (flat_labels != img_token_id)

            if valid_mask.any():
                avg_loss = losses[valid_mask].mean().item()
            else:
                avg_loss = losses.mean().item()

            perplexity = min(float(np.exp(avg_loss)), 1e6)
            return perplexity

        except Exception:
            return 0.0

    def select_hard_samples(
        self,
        samples: List[Dict],
        num_hard: int,
        num_easy: Optional[int] = None,
    ) -> Dict[str, List[Dict]]:
        """筛选硬样本和简单样本。

        Returns:
            {"hard": [...], "easy": [...], "random": [...], "difficulties": [...]}
        """
        print(f"\n  Computing sample difficulty for {len(samples)} samples ...")
        difficulties = []
        for i, sample in enumerate(tqdm(samples, desc="  Scoring")):
            ppl = self.compute_sample_difficulty(sample)
            difficulties.append({"index": i, "perplexity": ppl})
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        sorted_by_ppl = sorted(difficulties, key=lambda x: x["perplexity"], reverse=True)
        hard_indices = [d["index"] for d in sorted_by_ppl[:num_hard]]
        hard_samples = [samples[i] for i in hard_indices]

        if num_easy is None:
            num_easy = num_hard
        easy_indices = [d["index"] for d in sorted_by_ppl[-num_easy:]]
        easy_samples = [samples[i] for i in easy_indices]

        import random
        rng = random.Random(42)
        hard_set = set(hard_indices)
        remaining_indices = [i for i in range(len(samples)) if i not in hard_set]
        rng.shuffle(remaining_indices)
        random_samples = [samples[i] for i in remaining_indices[:num_hard]]

        hard_ppls = [difficulties[i]["perplexity"] for i in hard_indices]
        easy_ppls = [difficulties[i]["perplexity"] for i in easy_indices]
        all_ppls = [d["perplexity"] for d in difficulties]

        print(f"\n  Difficulty distribution:")
        print(f"    All:    mean={np.mean(all_ppls):.2f}, median={np.median(all_ppls):.2f}")
        print(f"    Hard:   mean={np.mean(hard_ppls):.2f} (top-{num_hard})")
        print(f"    Easy:   mean={np.mean(easy_ppls):.2f} (bottom-{num_easy})")

        return {
            "hard": hard_samples,
            "easy": easy_samples,
            "random": random_samples,
            "difficulties": difficulties,
            "stats": {
                "all_mean_ppl": round(float(np.mean(all_ppls)), 4),
                "all_median_ppl": round(float(np.median(all_ppls)), 4),
                "hard_mean_ppl": round(float(np.mean(hard_ppls)), 4),
                "easy_mean_ppl": round(float(np.mean(easy_ppls)), 4),
            },
        }


# ================================================================
# Stage 25 Searcher
# ================================================================

class HardSampleSearcher:
    """Stage 25: 硬样本感知校准 + functional group 搜索。

    运行两次搜索：
    1. 随机校准集 → 基线策略
    2. 硬样本校准集 → 硬样本策略
    对比两种策略差异。
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
        output_dir: str = "./quantization_outputs/stage25_hard_sample",
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
        num_hard_samples: int = 100,
        num_search_samples: int = 100,
        run_date: Optional[str] = None,
    ):
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.calibration_dataset = calibration_dataset
        self.algorithm_pool = algorithm_pool or build_stage25_pool()
        self.functional_groups = FUNCTIONAL_GROUPS
        self.gpu_ids = gpu_ids
        self.target_decoder_layers = target_decoder_layers
        self.seed = seed
        self.subsample_step = subsample_step
        self.max_calib_samples = max_calib_samples
        self.num_hard_samples = num_hard_samples
        self.num_search_samples = num_search_samples
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

        self._print_banner()

    def _set_seed(self):
        import random
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    def _load_model(self):
        components = load_internvlu(
            self.model_path, gpu_ids=self.gpu_ids, torch_dtype=torch.bfloat16,
        )
        self.model = components["model"].eval()
        self.tokenizer = components["tokenizer"]
        self.processor = components["processor"]

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
        print("Stage 25 (InternVL-U): Hard-Sample Aware Calibration Search")
        print("=" * 80)
        print(f"  Model: {self.model_path}")
        print(f"  Num hard samples: {self.num_hard_samples}")
        print(f"  Num search samples: {self.num_search_samples}")
        print(f"  Algorithm pool: {list(self.algorithm_pool.keys())}")
        print(f"  Output: {self.output_dir}")
        print("=" * 80 + "\n")

    # ---- Module helpers ----

    def _get_group_sublayer_names(self, layer_idx, group):
        prefix = f"language_model.model.layers.{layer_idx}"
        return [f"{prefix}.{s}" for s in group['suffixes']]

    def _get_decoder_layer_linear_names(self, layer_idx):
        prefix = f"language_model.model.layers.{layer_idx}"
        decoder_layer = self.model.language_model.model.layers[layer_idx]
        return [f"{prefix}.{n}" for n, m in decoder_layer.named_modules()
                if isinstance(m, (nn.Linear, HybridQuantLinear)) and n]

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

    # ---- Quantization ----

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

    def _restore_all_layers(self):
        for layer_idx in self.target_decoder_layers:
            for name in self._get_decoder_layer_linear_names(layer_idx):
                self._restore_layer(name)

    def _redispatch_model(self):
        try:
            from accelerate import dispatch_model
            if hasattr(self.model, "hf_device_map"):
                self.model = dispatch_model(self.model, device_map=self.model.hf_device_map)
        except Exception:
            pass

    # ---- Hidden state collection ----

    def _forward_sample(self, sample):
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

    def _collect_group_hs(self, layer_idx, group, samples):
        hook_target = group.get('hook_target')
        decoder_layer = self.model.language_model.model.layers[layer_idx]
        target = getattr(decoder_layer, hook_target, decoder_layer) if hook_target else decoder_layer
        hs_list = []
        captured = {}

        def hook_fn(module, inp, output):
            h = output[0] if isinstance(output, tuple) else output
            captured["output"] = h.detach().cpu()

        handle = target.register_forward_hook(hook_fn)
        try:
            for sample in samples:
                try:
                    self._forward_sample(sample)
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

    # ---- Per-subset activation collection ----

    def _collect_subset_activations(self, samples: List[Dict], tag: str) -> LazyActivationProvider:
        """为指定样本子集在线收集 Hessian / channel stats，返回独立的 ActivationProvider。

        复用 Stage 0 的 hook 逻辑：对每个 Linear 层注册输入 hook，
        在线累加 Hessian (X^T @ X) 和 channel stats (max, mean)。
        """
        print(f"\n  Collecting activation statistics for [{tag}] ({len(samples)} samples) ...")

        hessian_sum = {}
        act_channel_max = {}
        act_channel_mean_sum = {}
        token_count = {}

        hooks = []
        layer_names = []
        for name, module in self.model.named_modules():
            if not name.startswith("language_model.model.layers."):
                continue
            if isinstance(module, (nn.Linear, HybridQuantLinear)) and \
               not isinstance(module, HybridQuantLinear):
                layer_names.append(name)

                def make_hook(lname):
                    def hook_fn(m, inp, out):
                        x = inp[0] if isinstance(inp, tuple) else inp
                        if x is None:
                            return
                        x_det = x.detach().float()
                        if x_det.dim() == 3:
                            x_det = x_det.reshape(-1, x_det.shape[-1])
                        D = x_det.shape[-1]
                        x_cpu = x_det.cpu()
                        xtx = (x_cpu.t() @ x_cpu).double()
                        if lname not in hessian_sum:
                            hessian_sum[lname] = xtx
                            act_channel_max[lname] = x_cpu.abs().max(dim=0)[0]
                            act_channel_mean_sum[lname] = x_cpu.abs().sum(dim=0)
                            token_count[lname] = x_cpu.shape[0]
                        else:
                            hessian_sum[lname] += xtx
                            cur_max = x_cpu.abs().max(dim=0)[0]
                            act_channel_max[lname] = torch.max(act_channel_max[lname], cur_max)
                            act_channel_mean_sum[lname] += x_cpu.abs().sum(dim=0)
                            token_count[lname] += x_cpu.shape[0]
                    return hook_fn

                handle = module.register_forward_hook(make_hook(name))
                hooks.append(handle)

        for sample in tqdm(samples, desc=f"    [{tag}] activations"):
            try:
                self._forward_sample(sample)
            except Exception:
                pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        for h in hooks:
            h.remove()

        subset_dir = self.output_dir / f"activation_stats_{tag}_{self.run_date}"
        subset_dir.mkdir(parents=True, exist_ok=True)
        hessian_dir = subset_dir / "gptq_hessian"
        hessian_dir.mkdir(exist_ok=True)

        hessian_index = {}
        smooth_data = {}
        awq_data = {}

        for lname in layer_names:
            if lname not in hessian_sum:
                continue
            n = token_count[lname]
            D = hessian_sum[lname].shape[0]

            safe_name = lname.replace(".", "_")
            h_path = hessian_dir / f"{safe_name}.pt"
            torch.save({"hessian_sum": hessian_sum[lname], "nsamples": n}, h_path)
            hessian_index[lname] = {"path": str(h_path), "hidden_dim": D, "nsamples": n}

            smooth_data[lname] = {
                "act_channel_max": act_channel_max.get(lname, torch.zeros(D)),
                "weight_channel_max": torch.zeros(D),
                "nsamples": n,
            }
            ch_mean = act_channel_mean_sum[lname] / n if n > 0 else torch.zeros(D)
            awq_data[lname] = {
                "channel_mean": ch_mean,
                "channel_max": act_channel_max.get(lname, torch.zeros(D)),
                "nsamples": n,
            }

        idx_path = subset_dir / "gptq_hessian_index.json"
        with open(idx_path, "w") as f:
            json.dump({"format": "gptq_hessian_v1", "layers": hessian_index}, f, indent=2)
        smooth_path = subset_dir / "smoothquant_stats.pt"
        torch.save(smooth_data, smooth_path)
        awq_path = subset_dir / "awq_stats.pt"
        torch.save(awq_data, awq_path)

        print(f"    [{tag}] Saved: {len(hessian_index)} layers, dir={subset_dir}")

        provider = LazyActivationProvider(
            gptq_hessian_index=str(idx_path),
            smoothquant_stats=str(smooth_path),
            awq_stats=str(awq_path),
        )
        del hessian_sum, act_channel_max, act_channel_mean_sum, token_count
        gc.collect()
        return provider

    # ---- Single search run ----

    def _run_funcgroup_search(
        self, cka_samples: List[Dict], tag: str,
        activation_provider: Optional[LazyActivationProvider] = None,
    ) -> Dict:
        """执行一次完整的 functional group 搜索。"""
        ap = activation_provider or self.activation_provider
        old_ap = self.activation_provider
        self.activation_provider = ap

        available = {}
        for key, algo in self.algorithm_pool.items():
            if algo["config"].get("use_gptq") and not ap.available:
                continue
            if algo["config"].get("use_awq") and not ap.available:
                continue
            available[key] = algo

        fallback = list(available.keys())[0]
        assignments = {}
        search_log = []
        progress_file = self.output_dir / f"search_progress_{tag}_{self.run_date}.json"

        for layer_idx in tqdm(self.target_decoder_layers, desc=f"  {tag} search"):
            for group in self.functional_groups:
                sublayers = self._get_group_sublayer_names(layer_idx, group)
                existing = [n for n in sublayers if self._get_module(n) is not None]
                if not existing:
                    continue

                ref_hs = self._collect_group_hs(layer_idx, group, cka_samples)
                if not ref_hs:
                    for ln in existing:
                        assignments[ln] = fallback
                    continue

                best_algo = None
                best_cka = -1.0
                algo_scores = {}

                for algo_key, algo_info in available.items():
                    try:
                        self._apply_algorithm_to_group(layer_idx, group, algo_info["config"])
                        self._redispatch_model()
                        quant_hs = self._collect_group_hs(layer_idx, group, cka_samples)
                        cka = LinearCKA.compute_batched(
                            ref_hs, quant_hs, subsample_step=self.subsample_step
                        ) if quant_hs else 0.0
                        algo_scores[algo_key] = round(cka, 6)
                        if cka > best_cka:
                            best_cka = cka
                            best_algo = algo_key
                    except Exception:
                        algo_scores[algo_key] = -1.0
                    finally:
                        self._restore_group(layer_idx, group)
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                if best_algo is None:
                    best_algo = fallback

                for ln in existing:
                    assignments[ln] = best_algo
                self._apply_algorithm_to_group(layer_idx, group, available[best_algo]["config"])
                self._redispatch_model()

                search_log.append({
                    'layer_idx': layer_idx, 'group': group['name'],
                    'best_algo': best_algo, 'best_cka': round(best_cka, 6),
                    'all_scores': algo_scores,
                })

                del ref_hs
                gc.collect()

            # incremental save per layer
            with open(progress_file, 'w') as f:
                json.dump({
                    'tag': tag, 'search_log': search_log,
                    'assignments': assignments,
                    'completed_layers': layer_idx + 1,
                    'total_layers': len(self.target_decoder_layers),
                }, f, indent=2)
            ap.clear_cache()

        self.activation_provider = old_ap
        return {"assignments": assignments, "search_log": search_log}

    # ---- Main ----

    def search(self) -> Dict:
        # Phase 0: Load and score samples
        print("\n" + "=" * 80)
        print("Phase 0: Loading calibration data and computing difficulty scores")
        print("=" * 80)

        loader = LargeCalibrationLoader(self.calibration_dataset, self.max_calib_samples)
        all_samples = loader.load()

        selector = HardSampleSelector(self.model, self.processor, self.tokenizer)
        selection = selector.select_hard_samples(
            all_samples,
            num_hard=self.num_hard_samples,
        )

        hard_samples = selection["hard"][:self.num_search_samples]
        random_samples = selection["random"][:self.num_search_samples]
        difficulty_stats = selection["stats"]

        diff_path = self.output_dir / f"difficulty_scores_{self.run_date}.json"
        with open(diff_path, "w") as f:
            json.dump({
                "stats": difficulty_stats,
                "difficulties": selection["difficulties"],
            }, f, indent=2)
        print(f"  Difficulty scores saved: {diff_path}")

        # Phase 1a: Collect activation stats for RANDOM samples
        print("\n" + "=" * 80)
        print("Phase 1a: Collecting activation statistics for RANDOM samples")
        print("=" * 80)
        random_ap = self._collect_subset_activations(random_samples, "random")

        # Phase 1b: Search with RANDOM calibration
        print("\n" + "=" * 80)
        print("Phase 1b: Baseline search with RANDOM calibration samples")
        print("=" * 80)
        random_result = self._run_funcgroup_search(
            random_samples, "random", activation_provider=random_ap)

        # incremental save: random search results + config
        config_dir = Path(self.output_dir).parent / "configs"
        config_dir.mkdir(parents=True, exist_ok=True)
        random_cfg = {}
        for name, algo_key in random_result["assignments"].items():
            random_cfg[name] = self._build_full_layer_config(algo_key)
        random_cfg_path = config_dir / f"stage25_random_w4a4_{self.run_date}.json"
        with open(random_cfg_path, "w") as f:
            json.dump(dict(sorted(random_cfg.items())), f, indent=2)
        random_partial = self.output_dir / f"stage25_random_result_{self.run_date}.json"
        with open(random_partial, "w") as f:
            json.dump({"random_search": random_result, "status": "random_done"}, f, indent=2)
        print(f"  Random search saved: {random_cfg_path}")

        self._restore_all_layers()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Phase 2a: Collect activation stats for HARD samples
        print("\n" + "=" * 80)
        print("Phase 2a: Collecting activation statistics for HARD samples")
        print("=" * 80)
        hard_ap = self._collect_subset_activations(hard_samples, "hard")

        # Phase 2b: Search with HARD calibration
        print("\n" + "=" * 80)
        print("Phase 2b: Search with HARD calibration samples")
        print("=" * 80)
        hard_result = self._run_funcgroup_search(
            hard_samples, "hard", activation_provider=hard_ap)

        # incremental save: hard search results + config
        hard_cfg = {}
        for name, algo_key in hard_result["assignments"].items():
            hard_cfg[name] = self._build_full_layer_config(algo_key)
        hard_cfg_path = config_dir / f"stage25_hard_w4a4_{self.run_date}.json"
        with open(hard_cfg_path, "w") as f:
            json.dump(dict(sorted(hard_cfg.items())), f, indent=2)
        hard_partial = self.output_dir / f"stage25_hard_result_{self.run_date}.json"
        with open(hard_partial, "w") as f:
            json.dump({"hard_search": hard_result, "status": "hard_done"}, f, indent=2)
        print(f"  Hard search saved: {hard_cfg_path}")

        # Phase 3: Compare and export
        print("\n" + "=" * 80)
        print("Phase 3: Comparing strategies and exporting")
        print("=" * 80)

        print(f"  RANDOM config: {random_cfg_path}")
        print(f"  HARD config:   {hard_cfg_path}")

        diff_count = 0
        diff_details = []
        for name in random_result["assignments"]:
            r_algo = random_result["assignments"].get(name, "")
            h_algo = hard_result["assignments"].get(name, "")
            if r_algo != h_algo:
                diff_count += 1
                diff_details.append({
                    "sublayer": name,
                    "random_algo": r_algo,
                    "hard_algo": h_algo,
                })

        total = len(random_result["assignments"])
        print(f"\n  Strategy differences: {diff_count}/{total} sublayers "
              f"({100*diff_count/total:.1f}%)")

        results = {
            "random_search": random_result,
            "hard_search": hard_result,
            "comparison": {
                "total_sublayers": total,
                "different_sublayers": diff_count,
                "difference_rate": round(diff_count / total, 4) if total > 0 else 0,
                "differences": diff_details,
            },
            "difficulty_stats": difficulty_stats,
            "metadata": {
                "stage": 25,
                "strategy": "hard_sample_aware_calibration",
                "model_path": self.model_path,
                "run_date": self.run_date,
                "num_hard_samples": len(hard_samples),
                "num_random_samples": len(random_samples),
                "algorithm_pool": list(self.algorithm_pool.keys()),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            },
        }
        results_path = self.output_dir / f"stage25_results_{self.run_date}.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n{'=' * 80}")
        print("Stage 25 Summary:")
        print(f"  Random calibration: {len(random_samples)} samples")
        print(f"  Hard calibration:   {len(hard_samples)} samples")
        print(f"  Strategy diff rate: {100*diff_count/total:.1f}%")
        for d in diff_details[:10]:
            print(f"    {d['sublayer'].split('.')[-2]}.{d['sublayer'].split('.')[-1]}: "
                  f"{d['random_algo']} → {d['hard_algo']}")
        if len(diff_details) > 10:
            print(f"    ... and {len(diff_details) - 10} more")
        print(f"  Results: {results_path}")
        print(f"{'=' * 80}\n")

        return results


# ================================================================
# CLI
# ================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Stage 25 (InternVL-U): Hard-Sample Aware Calibration Search"
    )
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str,
                        default="./quantization_outputs/stage25_hard_sample")
    parser.add_argument("--calibration_dataset", type=str, required=True)
    parser.add_argument("--gptq_hessian_index", type=str, default=None)
    parser.add_argument("--smoothquant_stats", type=str, default=None)
    parser.add_argument("--awq_stats", type=str, default=None)
    parser.add_argument("--activation_data", type=str, default=None)
    parser.add_argument("--gpu_ids", type=str, default=None)
    parser.add_argument("--target_layers", type=str, default=None)
    parser.add_argument("--max_calib_samples", type=int, default=None)
    parser.add_argument("--num_hard_samples", type=int, default=100,
                        help="Number of hard samples to select (default: 100)")
    parser.add_argument("--num_search_samples", type=int, default=100,
                        help="Samples used per search run (default: 100)")
    parser.add_argument("--subsample_step", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_date", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)

    args = parser.parse_args()
    if args.config:
        with open(args.config, "r") as f:
            for k, v in yaml.safe_load(f).items():
                if hasattr(args, k) and getattr(args, k) is None:
                    setattr(args, k, v)

    target_layers = None
    if args.target_layers:
        target_layers = [int(x.strip()) for x in args.target_layers.split(",")]

    searcher = HardSampleSearcher(
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
        num_hard_samples=args.num_hard_samples,
        num_search_samples=args.num_search_samples,
        run_date=args.run_date,
    )
    results = searcher.search()
    print("\nStage 25 completed!")


if __name__ == "__main__":
    main()
