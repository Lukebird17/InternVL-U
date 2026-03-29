"""
Stage 21 (InternVL-U): Large-Calibration Functional-Group CKA Search

基于 Stage 20 + Stage 18 融合：
  1. Stage 20 的大校准集 (~1000 条 Flickr8K VQA)
  2. Stage 20 的 LazyActivationProvider (Hessian + channel stats)
  3. Stage 20 的精简 W4A4 算法池 (9 种配置, alpha 可调)
  4. Stage 18 的 functional group 搜索粒度 (attn / mlp 分开选算法)

与 Stage 20 的区别：
  Stage 20: 整个 decoder layer 共用一套算法 (CALM 路线)
  Stage 21: attention (Q/K/V/O) 和 MLP (gate/up/down) 各自独立选算法
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
# Functional Group Definitions
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
# Algorithm Pool (same as Stage 20)
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


def _s21_algo(name, desc, **overrides):
    cfg = dict(_W4_BASE)
    cfg.update(overrides)
    return {'name': name, 'description': desc, 'config': cfg}


def build_stage21_pool() -> Dict:
    """Stage 21 算法池 — 与 Stage 20 一致（仅 W4A4, alpha 可调）。"""
    pool = {}
    pool['gptq_w4a4'] = _s21_algo(
        'GPTQ W4A4', 'GPTQ group_size=64 W4A4',
        use_gptq=True, gptq_group_size=64,
    )
    for alpha in [0.5, 0.7, 0.85]:
        tag = f'a{int(alpha*100)}'
        pool[f'smooth_{tag}_gptq_w4a4'] = _s21_algo(
            f'Smooth(α={alpha})+GPTQ W4A4',
            f'SmoothQuant α={alpha}, GPTQ group_size=64 W4A4',
            use_smoothquant=True, smoothquant_alpha=alpha,
            use_gptq=True, gptq_group_size=64,
        )
    for alpha in [0.5, 0.7, 0.85]:
        tag = f'a{int(alpha*100)}'
        pool[f'svdquant_{tag}_w4a4'] = _s21_algo(
            f'SVDQuant α={alpha} W4A4',
            f'Smooth(α={alpha})+SVD(rank=32)+GPTQ W4A4',
            use_smoothquant=True, smoothquant_alpha=alpha,
            use_svd=True, svd_rank=32,
            use_gptq=True, gptq_group_size=64,
        )
    pool['svd_gptq_w4a4'] = _s21_algo(
        'SVD+GPTQ W4A4', 'SVD rank=32, GPTQ on residual W4A4',
        use_svd=True, svd_rank=32,
        use_gptq=True, gptq_group_size=64,
    )
    pool['awq_svd_rtn_w4a4'] = _s21_algo(
        'AWQ+SVD+RTN W4A4', 'AWQ n_grid=20, SVD rank=32, RTN W4A4',
        use_awq=True, awq_n_grid=20,
        use_svd=True, svd_rank=32,
    )
    return pool


# ================================================================
# LazyActivationProvider (reuse from Stage 20)
# ================================================================

class LazyActivationProvider:
    """从 Stage 0 输出懒加载 activation 数据。"""

    def __init__(
        self,
        gptq_hessian_index: Optional[str] = None,
        smoothquant_stats: Optional[str] = None,
        awq_stats: Optional[str] = None,
        legacy_activation_file: Optional[str] = None,
        cache_size: int = 20,
        damp_ratio: float = 0.01,
    ):
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

        self.available = (
            self.hessian_index is not None
            or self._legacy is not None
        )

    def __contains__(self, layer_name: str) -> bool:
        if self._legacy and layer_name in self._legacy:
            return True
        if self.hessian_index and layer_name in self.hessian_index:
            return True
        return False

    def __len__(self) -> int:
        if self._legacy:
            return len(self._legacy)
        if self.hessian_index:
            return len(self.hessian_index)
        return 0

    def get_activation(self, layer_name: str) -> Optional[torch.Tensor]:
        if layer_name in self._cache:
            self._cache.move_to_end(layer_name)
            return self._cache[layer_name]
        act = self._load_layer(layer_name)
        if act is not None:
            self._cache[layer_name] = act
            if len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return act

    def _load_layer(self, layer_name: str) -> Optional[torch.Tensor]:
        if self._legacy and layer_name in self._legacy:
            return self._legacy[layer_name]
        if self.hessian_index and layer_name in self.hessian_index:
            info = self.hessian_index[layer_name]
            pt_path = info["path"]
            data = torch.load(pt_path, map_location="cpu", weights_only=True)
            H_sum = data["hessian_sum"].double()
            n = data["nsamples"]
            return self._reconstruct_from_hessian(H_sum, n)
        return None

    def _reconstruct_from_hessian(self, H_sum: torch.Tensor, n: int) -> torch.Tensor:
        D = H_sum.shape[0]
        damp = self.damp_ratio * H_sum.diagonal().mean()
        H_reg = H_sum + damp * torch.eye(D, dtype=H_sum.dtype)
        try:
            L = torch.linalg.cholesky(H_reg)
            X_equiv = L.t().float()
        except torch.linalg.LinAlgError:
            eigvals, eigvecs = torch.linalg.eigh(H_reg)
            eigvals = eigvals.clamp(min=0)
            X_equiv = (eigvecs * eigvals.sqrt().unsqueeze(0)).t().float()
        return X_equiv

    def get_channel_max(self, layer_name: str) -> Optional[torch.Tensor]:
        if self.smooth_data and layer_name in self.smooth_data:
            return self.smooth_data[layer_name]["act_channel_max"]
        if self.awq_data and layer_name in self.awq_data:
            return self.awq_data[layer_name]["channel_max"]
        return None

    def get_channel_mean(self, layer_name: str) -> Optional[torch.Tensor]:
        if self.awq_data and layer_name in self.awq_data:
            return self.awq_data[layer_name]["channel_mean"]
        return None

    def clear_cache(self):
        self._cache.clear()
        gc.collect()


# ================================================================
# Calibration Loader (same as Stage 20)
# ================================================================

class LargeCalibrationLoader:
    def __init__(self, dataset_path: str, max_samples: Optional[int] = None):
        self.dataset_path = Path(dataset_path)
        self.max_samples = max_samples

    def load(self) -> List[Dict]:
        if not self.dataset_path.exists():
            raise FileNotFoundError(
                f"Calibration dataset not found: {self.dataset_path}\n"
                f"Run: python utils/build_calibration_dataset.py first"
            )
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
                "task_type": "und",
                "prompt": s.get("question", s.get("prompt", "")),
                "image": img,
                "generation_mode": "text",
                "question_type": s.get("question_type", "unknown"),
            })

        if self.max_samples and len(samples) > self.max_samples:
            import random
            random.shuffle(samples)
            samples = samples[:self.max_samples]

        type_counts = {}
        img_count = sum(1 for s in samples if s["image"] is not None)
        for s in samples:
            qt = s["question_type"]
            type_counts[qt] = type_counts.get(qt, 0) + 1

        print(f"    Loaded {len(samples)} calibration samples "
              f"({img_count} with images)")
        print(f"    Question types: {dict(sorted(type_counts.items()))}")
        return samples


# ================================================================
# Stage 21 Searcher
# ================================================================

class Stage21Searcher:
    """Stage 21: 大校准集 + functional group 搜索 (attn/mlp 独立选算法)。

    搜索粒度: functional group (attention 4 层一组, MLP 3 层一组)
    每个 decoder layer 的 attn 和 mlp 可以分配不同的量化算法。
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
        output_dir: str = "./quantization_outputs/stage21_funcgroup",
        calibration_dataset: Optional[str] = None,
        algorithm_pool: Optional[Dict] = None,
        gptq_hessian_index: Optional[str] = None,
        smoothquant_stats: Optional[str] = None,
        awq_stats: Optional[str] = None,
        activation_data_file: Optional[str] = None,
        gpu_ids: Optional[str] = None,
        max_mem_per_gpu: str = "40GiB",
        target_decoder_layers: Optional[List[int]] = None,
        seed: int = 42,
        subsample_step: int = 5,
        max_calib_samples: Optional[int] = None,
        cka_num_samples: int = 200,
        run_date: Optional[str] = None,
    ):
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.calibration_dataset = calibration_dataset
        self.algorithm_pool = algorithm_pool or build_stage21_pool()
        self.functional_groups = FUNCTIONAL_GROUPS
        self.gpu_ids = gpu_ids
        self.max_mem_per_gpu = max_mem_per_gpu
        self.target_decoder_layers = target_decoder_layers
        self.seed = seed
        self.subsample_step = subsample_step
        self.max_calib_samples = max_calib_samples
        self.cka_num_samples = cka_num_samples
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
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

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
        print("\nSaving original weights (language_model.model.layers) ...")
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
        print(f"  Saved {count} layers")

    def _print_banner(self):
        n_groups = len(self.functional_groups)
        print("\n" + "=" * 80)
        print("Stage 21 (InternVL-U): Large-Calibration Functional-Group CKA Search")
        print("=" * 80)
        print(f"  Model: {self.model_path}")
        print(f"  Run date: {self.run_date}")
        print(f"  Decoder layers: {self.num_decoder_layers}")
        print(f"  Target layers: {self.target_decoder_layers}")
        print(f"  Functional groups per layer: {n_groups}")
        for g in self.functional_groups:
            print(f"    - {g['name']}: {g['display']} ({len(g['suffixes'])} sublayers)")
        print(f"  Algorithm pool: {list(self.algorithm_pool.keys())}")
        print(f"  Calibration dataset: {self.calibration_dataset}")
        print(f"  Max calib samples: {self.max_calib_samples or 'all'}")
        print(f"  CKA search samples: {self.cka_num_samples} (subsampled from calib)")
        print(f"  Subsample step: {self.subsample_step}")
        print(f"  Weight bits: [4] (W4A4 only)")
        print(f"  Search granularity: functional group (attn / mlp independent)")
        print(f"  Output: {self.output_dir}")
        print("=" * 80 + "\n")

    # ---- module helpers ----

    def _get_decoder_layer_linear_names(self, layer_idx: int) -> List[str]:
        prefix = f"language_model.model.layers.{layer_idx}"
        decoder_layer = self.model.language_model.model.layers[layer_idx]
        names = []
        for sub_name, sub_module in decoder_layer.named_modules():
            if isinstance(sub_module, (nn.Linear, HybridQuantLinear)):
                full_name = f"{prefix}.{sub_name}" if sub_name else prefix
                names.append(full_name)
        return names

    def _get_group_sublayer_names(self, layer_idx: int, group: Dict) -> List[str]:
        prefix = f"language_model.model.layers.{layer_idx}"
        return [f"{prefix}.{suffix}" for suffix in group['suffixes']]

    def _get_module(self, name: str) -> Optional[nn.Module]:
        parts = name.split(".")
        module = self.model
        for part in parts:
            if hasattr(module, part):
                module = getattr(module, part)
            else:
                return None
        return module

    def _replace_module(self, name: str, new_module: nn.Module):
        parts = name.split(".")
        parent = self.model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], new_module)

    # ---- quantization apply / restore ----

    def _apply_algorithm_to_layer(self, layer_name: str, algo_config: Dict):
        module = self._get_module(layer_name)
        if module is None or not isinstance(module, (nn.Linear, HybridQuantLinear)):
            return
        if isinstance(module, HybridQuantLinear):
            self._restore_layer(layer_name)
            module = self._get_module(layer_name)

        device = module.weight.device
        dtype = module.weight.dtype
        config = algo_config.copy()
        if config.get("use_gptq", False) and not self.activation_provider.available:
            config["use_gptq"] = False
        if config.get("use_awq", False) and not self.activation_provider.available:
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
        if self.activation_provider.available and layer_name in self.activation_provider:
            act_data = self.activation_provider.get_activation(layer_name)
        quant_layer.prepare_weight(
            activation_data=act_data, layer_name=layer_name, verbose=False
        )
        self._replace_module(layer_name, quant_layer)
        del act_data
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _restore_layer(self, layer_name: str):
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
                linear.weight.data.copy_(
                    self.original_weights[layer_name]["weight"].to(device)
                )
                if (linear.bias is not None
                        and self.original_weights[layer_name]["bias"] is not None):
                    linear.bias.data.copy_(
                        self.original_weights[layer_name]["bias"].to(device)
                    )
            self._replace_module(layer_name, linear)
        elif isinstance(module, nn.Linear) and layer_name in self.original_weights:
            module.weight.data.copy_(
                self.original_weights[layer_name]["weight"].to(module.weight.device)
            )
            if (module.bias is not None
                    and self.original_weights[layer_name]["bias"] is not None):
                module.bias.data.copy_(
                    self.original_weights[layer_name]["bias"].to(module.bias.device)
                )

    def _apply_algorithm_to_group(self, layer_idx: int, group: Dict, algo_config: Dict):
        for name in self._get_group_sublayer_names(layer_idx, group):
            self._apply_algorithm_to_layer(name, algo_config)

    def _restore_group(self, layer_idx: int, group: Dict):
        for name in self._get_group_sublayer_names(layer_idx, group):
            self._restore_layer(name)

    def _restore_decoder_layer(self, layer_idx: int):
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

    # ---- forward & CKA ----

    def _forward_calibration_sample(self, sample: Dict):
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
        with torch.no_grad():
            pv = inputs.get("pixel_values")
            if gen_mode == "image":
                pv = None
            self.model.generate_hidden_states(
                pixel_values=pv,
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
            )

    def _collect_group_hidden_states(
        self, layer_idx: int, group: Dict, calibration_samples: List[Dict],
    ) -> List[torch.Tensor]:
        """Collect hidden states at the group's hook_target (self_attn or mlp)."""
        hook_target_name = group.get('hook_target')
        decoder_layer = self.model.language_model.model.layers[layer_idx]

        if hook_target_name:
            target_module = getattr(decoder_layer, hook_target_name, None)
        else:
            target_module = decoder_layer

        if target_module is None:
            target_module = decoder_layer

        captured_list = []
        captured = {}

        def hook_fn(module, input, output):
            h = output[0] if isinstance(output, tuple) else output
            captured["output"] = h.detach().cpu()

        handle = target_module.register_forward_hook(hook_fn)
        try:
            with torch.no_grad():
                for sample in calibration_samples:
                    try:
                        self._forward_calibration_sample(sample)
                    except Exception:
                        captured.clear()
                        continue
                    if "output" in captured:
                        captured_list.append(captured["output"])
                    captured.clear()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        finally:
            handle.remove()
        return captured_list

    # ---- config helpers ----

    def _build_full_layer_config(self, algo_key: str) -> Dict:
        cfg = dict(self._FULL_CONFIG_TEMPLATE)
        cfg.update(self.algorithm_pool[algo_key]["config"])
        if not cfg.get("use_svd", False):
            cfg["svd_rank"] = 0
        if not cfg.get("use_sparse", False):
            cfg["sparse_threshold"] = None
            cfg["sparse_ratio"] = 0.0
        return cfg

    # ---- subsample ----

    def _subsample_for_cka(self, full_samples: List[Dict]) -> List[Dict]:
        """从完整校准集中按 question_type 分层子采样。"""
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

        remaining_budget = n - len(selected)
        if remaining_budget > 0:
            pool = [s for t in types for s in by_type[t][per_type:]]
            rng.shuffle(pool)
            selected.extend(pool[:remaining_budget])

        selected = selected[:n]
        rng.shuffle(selected)

        type_counts = {}
        for s in selected:
            qt = s.get("question_type", "unknown")
            type_counts[qt] = type_counts.get(qt, 0) + 1
        print(f"    CKA subsample: {len(selected)} from {len(full_samples)} "
              f"(types: {dict(sorted(type_counts.items()))})")
        return selected

    # ---- search ----

    def search(self) -> Dict:
        # ---- Phase 1: Load calibration ----
        print("\n" + "=" * 80)
        print("Phase 1: Loading large calibration dataset")
        print("=" * 80)

        loader = LargeCalibrationLoader(
            dataset_path=self.calibration_dataset,
            max_samples=self.max_calib_samples,
        )
        all_calibration_samples = loader.load()
        cka_samples = self._subsample_for_cka(all_calibration_samples)

        available_algos = {}
        for key, algo in self.algorithm_pool.items():
            if algo["config"].get("use_gptq", False) and not self.activation_provider.available:
                print(f"    [SKIP] {key}: requires GPTQ Hessian data")
                continue
            if algo["config"].get("use_awq", False) and not self.activation_provider.available:
                print(f"    [SKIP] {key}: requires AWQ stats data")
                continue
            available_algos[key] = algo

        w4_algos = {
            k: v for k, v in available_algos.items()
            if v["config"].get("weight_bit", 4) == 4
        }

        print(f"\n  Available W4A4 algorithms: {list(w4_algos.keys())}")
        print(f"  Full calibration: {len(all_calibration_samples)} samples")
        print(f"  CKA search uses:  {len(cka_samples)} samples")
        print(f"  Functional groups: {[g['name'] for g in self.functional_groups]}")

        if not w4_algos:
            raise RuntimeError("No W4A4 algorithms available. Check activation data.")

        # ---- Phase 2: Functional group greedy search ----
        print("\n" + "=" * 80)
        print("Phase 2: Functional-Group Greedy CKA Search (W4A4)")
        print("=" * 80)

        fallback = list(w4_algos.keys())[0]
        group_assignments = {}
        group_cka_scores = {}
        search_log = []
        progress_file = self.output_dir / f"search_progress_funcgroup_w4a4_{self.run_date}.json"

        config_export_dir = Path(self.output_dir).parent / "configs"
        config_export_dir.mkdir(parents=True, exist_ok=True)

        n_groups = len(self.functional_groups)
        total_evals = len(self.target_decoder_layers) * n_groups * len(w4_algos)
        global_eval_idx = 0

        for layer_idx in tqdm(self.target_decoder_layers, desc="  W4A4 funcgroup greedy"):
            print(f"\n  {'─' * 60}")
            print(f"  [W4A4] Decoder Layer {layer_idx}/{self.num_decoder_layers - 1} "
                  f"({n_groups} groups)")

            layer_start = datetime.now()

            for group_idx, group in enumerate(self.functional_groups):
                group_sublayers = self._get_group_sublayer_names(layer_idx, group)
                existing = [n for n in group_sublayers if self._get_module(n) is not None]
                if not existing:
                    continue

                print(f"\n    Group {group_idx+1}/{n_groups}: "
                      f"{group['display']} ({len(existing)} sublayers)")

                ref_hs = self._collect_group_hidden_states(
                    layer_idx, group, cka_samples,
                )
                if not ref_hs:
                    for ln in existing:
                        group_assignments[ln] = fallback
                    continue

                best_algo = None
                best_cka = -1.0
                algo_scores = {}

                for algo_key, algo_info in w4_algos.items():
                    global_eval_idx += 1
                    try:
                        self._apply_algorithm_to_group(
                            layer_idx, group, algo_info["config"],
                        )
                        self._redispatch_model()

                        quant_hs = self._collect_group_hidden_states(
                            layer_idx, group, cka_samples,
                        )
                        cka = LinearCKA.compute_batched(
                            ref_hs, quant_hs,
                            subsample_step=self.subsample_step,
                        ) if quant_hs else 0.0

                        algo_scores[algo_key] = round(cka, 6)
                        print(f"      [{global_eval_idx}/{total_evals}] "
                              f"{algo_key:28s}: CKA={cka:.6f}")

                        if cka > best_cka:
                            best_cka = cka
                            best_algo = algo_key
                    except Exception as e:
                        print(f"      [{global_eval_idx}/{total_evals}] "
                              f"{algo_key:28s}: FAILED ({e})")
                        algo_scores[algo_key] = -1.0
                    finally:
                        self._restore_group(layer_idx, group)
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                if best_algo is None:
                    best_algo = fallback
                    best_cka = algo_scores.get(fallback, 0.0)

                for ln in existing:
                    group_assignments[ln] = best_algo
                self._apply_algorithm_to_group(
                    layer_idx, group, w4_algos[best_algo]["config"],
                )
                self._redispatch_model()

                group_cka_scores[(layer_idx, group['name'])] = algo_scores
                print(f"      >>> Best for {group['name']}: "
                      f"{best_algo} (CKA={best_cka:.6f})")

                search_log.append({
                    'decoder_layer_idx': layer_idx,
                    'group_name': group['name'],
                    'group_display': group['display'],
                    'sublayer_names': existing,
                    'best_algo': best_algo,
                    'best_cka': best_cka,
                    'all_scores': algo_scores,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                })

                del ref_hs
                gc.collect()

            elapsed = (datetime.now() - layer_start).total_seconds()
            print(f"\n    Layer {layer_idx} done in {elapsed:.1f}s")
            for g in self.functional_groups:
                gnames = self._get_group_sublayer_names(layer_idx, g)
                algo = group_assignments.get(gnames[0], '?') if gnames else '?'
                scores = group_cka_scores.get((layer_idx, g['name']), {})
                best = scores.get(algo, 0.0)
                print(f"      {g['name']:6s}: {algo:28s} (CKA={best:.6f})")

            with open(progress_file, 'w') as f:
                json.dump({
                    'weight_bit': 4,
                    'total_evals': global_eval_idx,
                    'search_log': search_log,
                }, f, indent=2)

            self.activation_provider.clear_cache()

        # ---- Phase 3: Export config ----
        print("\n" + "=" * 80)
        print("Phase 3: Exporting quantization config")
        print("=" * 80)

        export_cfg = {}
        for sublayer_name, algo_key in group_assignments.items():
            full_cfg = self._build_full_layer_config(algo_key)
            export_cfg[sublayer_name] = dict(full_cfg)

        config_path = config_export_dir / f"stage21_funcgroup_w4a4_{self.run_date}.json"
        with open(config_path, "w") as f:
            json.dump(dict(sorted(export_cfg.items())), f, indent=2)
        print(f"  Exported: {config_path}")

        # ---- Save results ----
        serializable_cka = {}
        for (li, gn), scores in group_cka_scores.items():
            serializable_cka[f"{li}_{gn}"] = {
                ak: round(av, 6) for ak, av in scores.items()
            }

        results = {
            "bitwidth_results": {
                "4": {
                    "weight_bit": 4,
                    "group_assignments": group_assignments,
                    "group_cka_scores": serializable_cka,
                    "search_log": search_log,
                    "exported_config_path": str(config_path),
                },
            },
            "metadata": {
                "stage": 21,
                "strategy": "functional_group_greedy",
                "model_path": self.model_path,
                "model_type": "internvlu",
                "run_date": self.run_date,
                "num_decoder_layers": self.num_decoder_layers,
                "target_layers": self.target_decoder_layers,
                "functional_groups": [g['name'] for g in self.functional_groups],
                "algorithm_pool": list(self.algorithm_pool.keys()),
                "calibration_dataset": str(self.calibration_dataset),
                "total_calibration_samples": len(all_calibration_samples),
                "cka_search_samples": len(cka_samples),
                "subsample_step": self.subsample_step,
                "seed": self.seed,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            },
        }
        results_path = self.output_dir / f"stage21_search_results_{self.run_date}.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        # ---- Summary ----
        print(f"\n{'=' * 80}")
        print(f"Stage 21 Search Summary (run_date={self.run_date}):")
        print(f"  Strategy: functional group (attn / mlp independent)")

        attn_algos = {}
        mlp_algos = {}
        for sublayer_name, algo_key in group_assignments.items():
            if 'self_attn' in sublayer_name:
                attn_algos[algo_key] = attn_algos.get(algo_key, 0) + 1
            elif 'mlp' in sublayer_name:
                mlp_algos[algo_key] = mlp_algos.get(algo_key, 0) + 1

        print(f"\n  Attention sublayers ({sum(attn_algos.values())} total):")
        for a, c in sorted(attn_algos.items(), key=lambda x: -x[1]):
            print(f"    {a}: {c}")
        print(f"\n  MLP sublayers ({sum(mlp_algos.values())} total):")
        for a, c in sorted(mlp_algos.items(), key=lambda x: -x[1]):
            print(f"    {a}: {c}")

        print(f"\n  Config: {config_path}")
        print(f"  Results: {results_path}")
        print(f"{'=' * 80}\n")

        return results


# ================================================================
# CLI
# ================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Stage 21 (InternVL-U): Large-Calibration Functional-Group CKA Search (W4A4)"
    )
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str,
                        default="./quantization_outputs/stage21_funcgroup")
    parser.add_argument("--calibration_dataset", type=str, required=True,
                        help="Path to calibration dataset JSON")
    parser.add_argument("--gptq_hessian_index", type=str, default=None)
    parser.add_argument("--smoothquant_stats", type=str, default=None)
    parser.add_argument("--awq_stats", type=str, default=None)
    parser.add_argument("--activation_data", type=str, default=None,
                        help="Legacy: Stage 0 activation data (.pt)")
    parser.add_argument("--gpu_ids", type=str, default=None)
    parser.add_argument("--max_mem_per_gpu", type=str, default="40GiB")
    parser.add_argument("--target_layers", type=str, default=None,
                        help="Comma-separated layer indices (default: all)")
    parser.add_argument("--max_calib_samples", type=int, default=None)
    parser.add_argument("--cka_num_samples", type=int, default=200,
                        help="Samples used for CKA search per group (default: 200)")
    parser.add_argument("--subsample_step", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
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

    searcher = Stage21Searcher(
        model_path=args.model_path,
        output_dir=args.output_dir,
        calibration_dataset=args.calibration_dataset,
        gptq_hessian_index=args.gptq_hessian_index,
        smoothquant_stats=args.smoothquant_stats,
        awq_stats=args.awq_stats,
        activation_data_file=args.activation_data,
        gpu_ids=args.gpu_ids,
        max_mem_per_gpu=args.max_mem_per_gpu,
        target_decoder_layers=target_layers,
        seed=args.seed,
        subsample_step=args.subsample_step,
        max_calib_samples=args.max_calib_samples,
        cka_num_samples=args.cka_num_samples,
        run_date=args.run_date,
    )
    results = searcher.search()
    print("\nStage 21 search completed!")
    print(f"Results: {searcher.output_dir / f'stage21_search_results_{searcher.run_date}.json'}")


if __name__ == "__main__":
    main()
