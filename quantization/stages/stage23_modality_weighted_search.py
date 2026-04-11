"""
Stage 23 (InternVL-U): Modality-Weighted CKA Functional-Group Search

方向 1 实验 2：模态优先搜索 (Modality-Priority Search)

基于 Stage 21 的 functional group 搜索，修改搜索目标函数：
  原始 CALM:  score = CKA(fp_hs, quant_hs)
  本方法:     score = w_v * CKA_v + w_l * CKA_l

其中 CKA_v / CKA_l 分别在视觉/文本 Token 子集上计算。
通过调高 w_v 来显式保护视觉 Token 的表征质量。

关键创新点：
  不再使用单一的通用 CKA，而是使用模态加权指标，
  在搜索时显式地调高视觉权重，让量化算法更关注
  对视觉判别任务重要的特征保护。
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
# Modality-Aware CKA
# ================================================================

class ModalityCKA:
    """模态分离 CKA 计算。

    将隐状态按 vision/text token 分离，分别计算 CKA 后加权合并。
    """

    @staticmethod
    def compute_batched(
        fp_batches: List[torch.Tensor],
        quant_batches: List[torch.Tensor],
        vision_masks: List[torch.Tensor],
        subsample_step: int = 5,
        vision_weight: float = 0.7,
        text_weight: float = 0.3,
    ) -> Dict[str, float]:
        """分模态 CKA + 加权合并。

        跨样本聚合 vision/text tokens 后统一计算 CKA，
        避免单样本 vision token 数量过少导致无法计算的问题。

        Args:
            fp_batches: FP16 hidden states per sample, each [B*S, D]
            quant_batches: Quantized hidden states per sample
            vision_masks: Boolean mask per sample, True for vision tokens
            subsample_step: token subsampling stride
            vision_weight: CKA_v 权重
            text_weight: CKA_l 权重

        Returns:
            {"combined", "cka_vision", "cka_text", "cka_all"}
        """
        all_fp_vision, all_qt_vision = [], []
        all_fp_text, all_qt_text = [], []
        all_fp, all_qt = [], []

        for fp, qt, vmask in zip(fp_batches, quant_batches, vision_masks):
            if fp.dim() == 3:
                fp = fp[:, ::subsample_step, :].reshape(-1, fp.shape[-1])
            if qt.dim() == 3:
                qt = qt[:, ::subsample_step, :].reshape(-1, qt.shape[-1])

            if fp.shape[0] != qt.shape[0]:
                n = min(fp.shape[0], qt.shape[0])
                fp, qt = fp[:n], qt[:n]

            if vmask.shape[0] != fp.shape[0]:
                n = min(vmask.shape[0], fp.shape[0])
                vmask = vmask[:n]
                fp, qt = fp[:n], qt[:n]

            if fp.shape[0] < 2:
                continue

            all_fp.append(fp)
            all_qt.append(qt)

            t_mask = ~vmask
            if vmask.any():
                all_fp_vision.append(fp[vmask])
                all_qt_vision.append(qt[vmask])
            if t_mask.any():
                all_fp_text.append(fp[t_mask])
                all_qt_text.append(qt[t_mask])

        cka_a = 0.0
        if all_fp:
            fp_cat = torch.cat(all_fp, dim=0)
            qt_cat = torch.cat(all_qt, dim=0)
            step = max(1, fp_cat.shape[0] // 2000)
            cka_a = LinearCKA.compute(fp_cat[::step], qt_cat[::step])

        cka_v = 0.0
        if all_fp_vision:
            fp_v = torch.cat(all_fp_vision, dim=0)
            qt_v = torch.cat(all_qt_vision, dim=0)
            if fp_v.shape[0] >= 2:
                step = max(1, fp_v.shape[0] // 2000)
                cka_v = LinearCKA.compute(fp_v[::step], qt_v[::step])

        cka_t = 0.0
        if all_fp_text:
            fp_t = torch.cat(all_fp_text, dim=0)
            qt_t = torch.cat(all_qt_text, dim=0)
            if fp_t.shape[0] >= 2:
                step = max(1, fp_t.shape[0] // 2000)
                cka_t = LinearCKA.compute(fp_t[::step], qt_t[::step])

        has_vision = len(all_fp_vision) > 0 and sum(t.shape[0] for t in all_fp_vision) >= 2
        has_text = len(all_fp_text) > 0 and sum(t.shape[0] for t in all_fp_text) >= 2

        if has_vision and has_text:
            combined = vision_weight * cka_v + text_weight * cka_t
        else:
            combined = cka_a

        return {
            "combined": combined,
            "cka_vision": cka_v,
            "cka_text": cka_t,
            "cka_all": cka_a,
        }


# ================================================================
# Functional Group Definitions (same as Stage 21)
# ================================================================

FUNCTIONAL_GROUPS = [
    {
        'name': 'attn',
        'display': 'Attention (Q/K/V/O)',
        'suffixes': ['self_attn.q_proj', 'self_attn.k_proj',
                     'self_attn.v_proj', 'self_attn.o_proj'],
        'hook_target': 'self_attn',
    },
    {
        'name': 'mlp',
        'display': 'MLP (gate/up/down)',
        'suffixes': ['mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj'],
        'hook_target': 'mlp',
    },
]


# ================================================================
# Algorithm Pool (same as Stage 21)
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


def _s23_algo(name, desc, **overrides):
    cfg = dict(_W4_BASE)
    cfg.update(overrides)
    return {'name': name, 'description': desc, 'config': cfg}


def build_stage23_pool() -> Dict:
    pool = {}
    pool['gptq_w4a4'] = _s23_algo(
        'GPTQ W4A4', 'GPTQ group_size=64 W4A4',
        use_gptq=True, gptq_group_size=64,
    )
    for alpha in [0.5, 0.7, 0.85]:
        tag = f'a{int(alpha*100)}'
        pool[f'smooth_{tag}_gptq_w4a4'] = _s23_algo(
            f'Smooth(α={alpha})+GPTQ W4A4',
            f'SmoothQuant α={alpha}, GPTQ group_size=64 W4A4',
            use_smoothquant=True, smoothquant_alpha=alpha,
            use_gptq=True, gptq_group_size=64,
        )
    for alpha in [0.5, 0.7, 0.85]:
        tag = f'a{int(alpha*100)}'
        pool[f'svdquant_{tag}_w4a4'] = _s23_algo(
            f'SVDQuant α={alpha} W4A4',
            f'Smooth(α={alpha})+SVD(rank=32)+GPTQ W4A4',
            use_smoothquant=True, smoothquant_alpha=alpha,
            use_svd=True, svd_rank=32,
            use_gptq=True, gptq_group_size=64,
        )
    pool['svd_gptq_w4a4'] = _s23_algo(
        'SVD+GPTQ W4A4', 'SVD rank=32, GPTQ on residual W4A4',
        use_svd=True, svd_rank=32,
        use_gptq=True, gptq_group_size=64,
    )
    pool['awq_svd_rtn_w4a4'] = _s23_algo(
        'AWQ+SVD+RTN W4A4', 'AWQ n_grid=20, SVD rank=32, RTN W4A4',
        use_awq=True, awq_n_grid=20,
        use_svd=True, svd_rank=32,
    )
    return pool


# ================================================================
# LazyActivationProvider (reuse from Stage 20)
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
                idx = json.load(f)
            self.hessian_index = idx.get("layers", {})
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
            print(f"  [Legacy] Loaded activation data: {len(self._legacy)} layers")

        self.available = (self.hessian_index is not None or self._legacy is not None)

    def __contains__(self, layer_name):
        if self._legacy and layer_name in self._legacy:
            return True
        if self.hessian_index and layer_name in self.hessian_index:
            return True
        return False

    def get_activation(self, layer_name):
        if layer_name in self._cache:
            self._cache.move_to_end(layer_name)
            return self._cache[layer_name]
        act = self._load_layer(layer_name)
        if act is not None:
            self._cache[layer_name] = act
            if len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return act

    def get_activation_max(self, layer_name):
        """返回 Stage 0 收集的 per-channel activation max（用于 SmoothQuant）。"""
        if self.smooth_data and layer_name in self.smooth_data:
            return self.smooth_data[layer_name].get("act_channel_max")
        if self.awq_data and layer_name in self.awq_data:
            return self.awq_data[layer_name].get("channel_max")
        return None

    def _load_layer(self, layer_name):
        if self._legacy and layer_name in self._legacy:
            return self._legacy[layer_name]
        if self.hessian_index and layer_name in self.hessian_index:
            info = self.hessian_index[layer_name]
            data = torch.load(info["path"], map_location="cpu", weights_only=True)
            H_sum = data["hessian_sum"].double()
            return self._reconstruct_from_hessian(H_sum, data["nsamples"])
        return None

    def _reconstruct_from_hessian(self, H_sum, n):
        D = H_sum.shape[0]
        damp = self.damp_ratio * H_sum.diagonal().mean()
        H_reg = H_sum + damp * torch.eye(D, dtype=H_sum.dtype)
        try:
            L = torch.linalg.cholesky(H_reg)
            return L.t().float()
        except torch.linalg.LinAlgError:
            eigvals, eigvecs = torch.linalg.eigh(H_reg)
            eigvals = eigvals.clamp(min=0)
            return (eigvecs * eigvals.sqrt().unsqueeze(0)).t().float()

    def clear_cache(self):
        self._cache.clear()
        gc.collect()


# ================================================================
# Calibration Loader
# ================================================================

class LargeCalibrationLoader:
    def __init__(self, dataset_path, max_samples=None):
        self.dataset_path = Path(dataset_path)
        self.max_samples = max_samples

    def load(self):
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Calibration dataset not found: {self.dataset_path}")
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        raw_samples = data.get("samples", data)
        if isinstance(raw_samples, dict):
            raw_samples = list(raw_samples.values())
        samples = []
        for s in raw_samples:
            img_path = s.get("image_path")
            img = None
            if img_path and Path(img_path).exists():
                try:
                    img = Image.open(img_path).convert("RGB")
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
# Stage 23 Searcher
# ================================================================

class ModalityWeightedSearcher:
    """Stage 23: 模态加权 CKA 功能组搜索。

    核心区别（与 Stage 21）：
    1. 使用 ModalityCKA 代替 LinearCKA
    2. 收集 hidden states 时同步记录 vision/text mask
    3. 搜索目标 = w_v * CKA_v + w_l * CKA_l
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
        output_dir: str = "./quantization_outputs/stage23_modality_weighted",
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
        cka_num_samples: int = 200,
        vision_weight: float = 0.7,
        text_weight: float = 0.3,
        run_date: Optional[str] = None,
    ):
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.calibration_dataset = calibration_dataset
        self.algorithm_pool = algorithm_pool or build_stage23_pool()
        self.functional_groups = FUNCTIONAL_GROUPS
        self.gpu_ids = gpu_ids
        self.target_decoder_layers = target_decoder_layers
        self.seed = seed
        self.subsample_step = subsample_step
        self.max_calib_samples = max_calib_samples
        self.cka_num_samples = cka_num_samples
        self.vision_weight = vision_weight
        self.text_weight = text_weight
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
        print(f"\nLoading InternVL-U from {self.model_path} ...")
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
        print("Stage 23 (InternVL-U): Modality-Weighted CKA Functional-Group Search")
        print("=" * 80)
        print(f"  Model: {self.model_path}")
        print(f"  Vision weight (w_v): {self.vision_weight}")
        print(f"  Text weight (w_l):   {self.text_weight}")
        print(f"  Algorithm pool: {list(self.algorithm_pool.keys())}")
        print(f"  Functional groups: {[g['name'] for g in self.functional_groups]}")
        print(f"  CKA samples: {self.cka_num_samples}")
        print(f"  Output: {self.output_dir}")
        print("=" * 80 + "\n")

    # ---- module helpers (same as Stage 21) ----

    def _get_decoder_layer_linear_names(self, layer_idx):
        prefix = f"language_model.model.layers.{layer_idx}"
        decoder_layer = self.model.language_model.model.layers[layer_idx]
        names = []
        for sub_name, sub_module in decoder_layer.named_modules():
            if isinstance(sub_module, (nn.Linear, HybridQuantLinear)):
                full_name = f"{prefix}.{sub_name}" if sub_name else prefix
                names.append(full_name)
        return names

    def _get_group_sublayer_names(self, layer_idx, group):
        prefix = f"language_model.model.layers.{layer_idx}"
        return [f"{prefix}.{suffix}" for suffix in group['suffixes']]

    def _get_module(self, name):
        parts = name.split(".")
        module = self.model
        for part in parts:
            if hasattr(module, part):
                module = getattr(module, part)
            else:
                return None
        return module

    def _replace_module(self, name, new_module):
        parts = name.split(".")
        parent = self.model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], new_module)

    # ---- quantization apply / restore (same as Stage 21) ----

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
            linear = nn.Linear(
                module.in_features, module.out_features,
                bias=module.bias is not None, device=device, dtype=dtype,
            )
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
                self.model = dispatch_model(
                    self.model, device_map=self.model.hf_device_map, offload_dir=None,
                )
        except Exception as e:
            print(f"  [Warning] Re-dispatch failed: {e}")

    # ---- CORE DIFFERENCE: modality-aware hidden state collection ----

    def _forward_and_get_input_ids(self, sample):
        """前向一条样本并返回 input_ids（用于构建 vision mask）。"""
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
        return inputs

    def _collect_group_hidden_states_with_masks(
        self, layer_idx, group, calibration_samples,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """收集组级 hidden states，同时返回 vision masks。

        Returns:
            (hidden_states_list, vision_masks_list)
            每个元素对应一个校准样本。
        """
        hook_target_name = group.get('hook_target')
        decoder_layer = self.model.language_model.model.layers[layer_idx]
        target_module = getattr(decoder_layer, hook_target_name, decoder_layer) if hook_target_name else decoder_layer

        hs_list = []
        mask_list = []
        captured = {}

        def hook_fn(module, inp, output):
            h = output[0] if isinstance(output, tuple) else output
            captured["output"] = h.detach().cpu()

        handle = target_module.register_forward_hook(hook_fn)
        img_ctx_id = self.model.img_context_token_id

        try:
            with torch.no_grad():
                for sample in calibration_samples:
                    try:
                        inputs = self._forward_and_get_input_ids(sample)
                        input_ids = inputs["input_ids"]

                        pv = inputs.get("pixel_values")
                        if sample.get("generation_mode") == "image":
                            pv = None
                        self.model.generate_hidden_states(
                            pixel_values=pv,
                            input_ids=input_ids,
                            attention_mask=inputs.get("attention_mask"),
                        )
                    except Exception:
                        captured.clear()
                        continue

                    if "output" in captured:
                        hs = captured["output"]  # [B, S, D]
                        B, S, D = hs.shape
                        hs_flat = hs.reshape(B * S, D)
                        hs_list.append(hs_flat)

                        ids_flat = input_ids.detach().cpu().reshape(-1)
                        if ids_flat.shape[0] != hs_flat.shape[0]:
                            n = min(ids_flat.shape[0], hs_flat.shape[0])
                            ids_flat = ids_flat[:n]
                        vision_mask = (ids_flat == img_ctx_id)
                        mask_list.append(vision_mask)

                    captured.clear()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        finally:
            handle.remove()

        return hs_list, mask_list

    # ---- subsample ----

    def _subsample_for_cka(self, full_samples):
        import random
        n = self.cka_num_samples
        if len(full_samples) <= n:
            return full_samples
        by_type = {}
        for s in full_samples:
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

    def _build_full_layer_config(self, algo_key):
        cfg = dict(self._FULL_CONFIG_TEMPLATE)
        cfg.update(self.algorithm_pool[algo_key]["config"])
        if not cfg.get("use_svd", False):
            cfg["svd_rank"] = 0
        if not cfg.get("use_sparse", False):
            cfg["sparse_threshold"] = None
            cfg["sparse_ratio"] = 0.0
        return cfg

    # ---- search ----

    def search(self) -> Dict:
        print("\n" + "=" * 80)
        print("Phase 1: Loading calibration dataset")
        print("=" * 80)

        loader = LargeCalibrationLoader(
            dataset_path=self.calibration_dataset,
            max_samples=self.max_calib_samples,
        )
        all_samples = loader.load()
        cka_samples = self._subsample_for_cka(all_samples)

        available_algos = {}
        for key, algo in self.algorithm_pool.items():
            if algo["config"].get("use_gptq") and not self.activation_provider.available:
                continue
            if algo["config"].get("use_awq") and not self.activation_provider.available:
                continue
            available_algos[key] = algo
        w4_algos = {k: v for k, v in available_algos.items() if v["config"].get("weight_bit", 4) == 4}

        print(f"  Available algorithms: {list(w4_algos.keys())}")
        print(f"  Vision weight: {self.vision_weight}, Text weight: {self.text_weight}")

        if not w4_algos:
            raise RuntimeError("No W4A4 algorithms available.")

        # ---- Phase 2: Modality-weighted functional group search ----
        print("\n" + "=" * 80)
        print("Phase 2: Modality-Weighted Functional-Group Search")
        print("=" * 80)

        fallback = list(w4_algos.keys())[0]
        group_assignments = {}
        group_scores = {}
        search_log = []
        progress_file = self.output_dir / f"search_progress_{self.run_date}.json"
        config_export_dir = Path(self.output_dir).parent / "configs"
        config_export_dir.mkdir(parents=True, exist_ok=True)
        partial_config_path = config_export_dir / f"stage23_modality_weighted_w4a4_{self.run_date}_partial.json"

        for layer_idx in tqdm(self.target_decoder_layers, desc="Modality-weighted search"):
            print(f"\n  {'─' * 60}")
            print(f"  Layer {layer_idx}/{self.num_decoder_layers - 1}")

            for group in self.functional_groups:
                sublayer_names = self._get_group_sublayer_names(layer_idx, group)
                existing = [n for n in sublayer_names if self._get_module(n) is not None]
                if not existing:
                    continue

                print(f"    Group: {group['display']}")

                ref_hs, ref_masks = self._collect_group_hidden_states_with_masks(
                    layer_idx, group, cka_samples,
                )
                if not ref_hs:
                    for ln in existing:
                        group_assignments[ln] = fallback
                    continue

                best_algo = None
                best_score = -1.0
                algo_detail = {}

                for algo_key, algo_info in w4_algos.items():
                    try:
                        self._apply_algorithm_to_group(layer_idx, group, algo_info["config"])
                        self._redispatch_model()

                        quant_hs, quant_masks = self._collect_group_hidden_states_with_masks(
                            layer_idx, group, cka_samples,
                        )

                        scores = ModalityCKA.compute_batched(
                            ref_hs, quant_hs, ref_masks,
                            subsample_step=self.subsample_step,
                            vision_weight=self.vision_weight,
                            text_weight=self.text_weight,
                        ) if quant_hs else {"combined": 0.0, "cka_vision": 0.0, "cka_text": 0.0, "cka_all": 0.0}

                        algo_detail[algo_key] = {k: round(v, 6) for k, v in scores.items()}
                        print(f"      {algo_key:28s}: combined={scores['combined']:.6f} "
                              f"(V={scores['cka_vision']:.4f} T={scores['cka_text']:.4f})")

                        if scores["combined"] > best_score:
                            best_score = scores["combined"]
                            best_algo = algo_key

                    except Exception as e:
                        print(f"      {algo_key:28s}: FAILED ({e})")
                        algo_detail[algo_key] = {"combined": -1.0}
                    finally:
                        self._restore_group(layer_idx, group)
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                if best_algo is None:
                    best_algo = fallback

                for ln in existing:
                    group_assignments[ln] = best_algo
                self._apply_algorithm_to_group(layer_idx, group, w4_algos[best_algo]["config"])
                self._redispatch_model()

                group_scores[(layer_idx, group['name'])] = algo_detail
                print(f"      >>> Best: {best_algo} (score={best_score:.6f})")

                search_log.append({
                    'layer_idx': layer_idx, 'group': group['name'],
                    'best_algo': best_algo, 'best_score': round(best_score, 6),
                    'all_scores': algo_detail,
                })

                del ref_hs, ref_masks
                gc.collect()

            # incremental save: progress + partial config + partial results
            partial_cfg = {}
            for sn, ak in group_assignments.items():
                partial_cfg[sn] = self._build_full_layer_config(ak)
            serializable = {}
            for (li, gn), detail in group_scores.items():
                serializable[f"{li}_{gn}"] = detail
            with open(progress_file, 'w') as f:
                json.dump({
                    'search_log': search_log,
                    'group_assignments': group_assignments,
                    'group_scores': serializable,
                    'completed_layers': layer_idx + 1,
                    'total_layers': len(self.target_decoder_layers),
                }, f, indent=2)
            with open(partial_config_path, 'w') as f:
                json.dump(dict(sorted(partial_cfg.items())), f, indent=2)
            self.activation_provider.clear_cache()

        # ---- Phase 3: Export ----
        print("\n" + "=" * 80)
        print("Phase 3: Exporting config")
        print("=" * 80)

        export_cfg = {}
        for sublayer_name, algo_key in group_assignments.items():
            export_cfg[sublayer_name] = self._build_full_layer_config(algo_key)
        config_path = config_export_dir / f"stage23_modality_weighted_w4a4_{self.run_date}.json"
        with open(config_path, "w") as f:
            json.dump(dict(sorted(export_cfg.items())), f, indent=2)
        print(f"  Config: {config_path}")

        serializable_scores = {}
        for (li, gn), detail in group_scores.items():
            serializable_scores[f"{li}_{gn}"] = detail

        results = {
            "bitwidth_results": {"4": {
                "group_assignments": group_assignments,
                "group_scores": serializable_scores,
                "search_log": search_log,
                "exported_config_path": str(config_path),
            }},
            "metadata": {
                "stage": 23,
                "strategy": "modality_weighted_funcgroup",
                "vision_weight": self.vision_weight,
                "text_weight": self.text_weight,
                "model_path": self.model_path,
                "run_date": self.run_date,
                "num_decoder_layers": self.num_decoder_layers,
                "algorithm_pool": list(self.algorithm_pool.keys()),
                "cka_search_samples": len(cka_samples),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            },
        }
        results_path = self.output_dir / f"stage23_search_results_{self.run_date}.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n{'=' * 80}")
        print(f"Stage 23 Summary (w_v={self.vision_weight}, w_l={self.text_weight}):")
        algo_counts = {}
        for a in group_assignments.values():
            algo_counts[a] = algo_counts.get(a, 0) + 1
        for a, c in sorted(algo_counts.items(), key=lambda x: -x[1]):
            print(f"  {a}: {c} sublayers")
        print(f"{'=' * 80}\n")

        return results


# ================================================================
# CLI
# ================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Stage 23 (InternVL-U): Modality-Weighted CKA Functional-Group Search"
    )
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str,
                        default="./quantization_outputs/stage23_modality_weighted")
    parser.add_argument("--calibration_dataset", type=str, required=True)
    parser.add_argument("--gptq_hessian_index", type=str, default=None)
    parser.add_argument("--smoothquant_stats", type=str, default=None)
    parser.add_argument("--awq_stats", type=str, default=None)
    parser.add_argument("--activation_data", type=str, default=None)
    parser.add_argument("--gpu_ids", type=str, default=None)
    parser.add_argument("--target_layers", type=str, default=None)
    parser.add_argument("--max_calib_samples", type=int, default=None)
    parser.add_argument("--cka_num_samples", type=int, default=200)
    parser.add_argument("--subsample_step", type=int, default=5)
    parser.add_argument("--vision_weight", type=float, default=0.7,
                        help="Weight for vision CKA (default: 0.7)")
    parser.add_argument("--text_weight", type=float, default=0.3,
                        help="Weight for text CKA (default: 0.3)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_date", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)

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

    searcher = ModalityWeightedSearcher(
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
        vision_weight=args.vision_weight,
        text_weight=args.text_weight,
        run_date=args.run_date,
    )
    results = searcher.search()
    print("\nStage 23 search completed!")


if __name__ == "__main__":
    main()
