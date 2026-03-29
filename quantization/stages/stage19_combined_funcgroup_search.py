"""
Stage 19 (InternVL-U): Combined-Metric Functional-Group-Level Greedy Search

基于 Stage 18 的功能组贪心搜索框架，引入以下改进：

  1. 组合相似度指标 (CombinedSimilarity)
     = 0.4 * CKA + 0.3 * CosineSim + 0.3 * MSE_score
     CKA 衡量全局结构，Cosine 衡量方向对齐，MSE 衡量幅度保真。

  2. UND + GEN 混合校准
     同时覆盖理解与生成两条推理路径的激活分布，
     解决此前搜索目标仅对齐理解路径的问题。

  3. 仅搜索 W4A4 / W3A4（跳过 W2A4）

  4. 全程日期命名
     所有输出（搜索进度、结果、配置）共享同一个 run_date，
     评测结果目录也沿用该日期。
"""

import os
import sys
import json
import gc
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from tqdm import tqdm
from collections import defaultdict
import yaml
import argparse

_current_dir = Path(__file__).resolve().parent
_quant_dir = _current_dir.parent

if str(_quant_dir) not in sys.path:
    sys.path.insert(0, str(_quant_dir))

from utils.bagel_imports import (
    HybridQuantLinear, CombinedSimilarity, LinearCKA, ALGORITHM_POOL,
)
from utils.model_loader import load_internvlu
from utils.calibration import CalibrationDataLoader, image_for_processor


# ============================================================
# Functional Group Definitions (standard Qwen/Llama structure)
# ============================================================

FUNCTIONAL_GROUPS_INTERNVLU = [
    {
        'name': 'attn_shared',
        'display': 'Attention (Q/K/V/O projections)',
        'suffixes': ['self_attn.q_proj', 'self_attn.k_proj',
                     'self_attn.v_proj', 'self_attn.o_proj'],
        'hook_target': 'self_attn',
    },
    {
        'name': 'mlp_shared',
        'display': 'MLP (gate/up/down projections)',
        'suffixes': ['mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj'],
        'hook_target': 'mlp',
    },
]

SKIP_WEIGHT_BITS = {2}


# ============================================================
# Stage 19 Searcher
# ============================================================

class CombinedFuncGroupSearcher:
    """
    Stage 19: Combined-Metric Functional-Group Greedy Search.

    与 Stage 18 的区别：
      - 使用 CombinedSimilarity (CKA + Cosine + MSE) 替代纯 CKA
      - 校准数据包含 UND + GEN 两种路径
      - 跳过 W2A4
      - 所有输出附带 run_date 后缀
    """

    _FULL_CONFIG_TEMPLATE = {
        "weight_bit": 4, "act_bit": 4,
        "quant_percentile": 0.999999, "act_unsigned": True,
        "use_sparse": False, "sparse_ratio": 0.0, "sparse_threshold": None,
        "use_smoothquant": False, "smoothquant_alpha": 0.5,
        "use_svd": False, "svd_rank": 0,
        "use_gptq": False,
        "gptq_group_size": 64, "gptq_damp_percentage": 0.01,
        "gptq_block_size": 128, "gptq_num_inv_tries": 250,
        "gptq_hessian_block_size": 512,
        "use_block_quant": False, "use_block_quant_act": False,
        "block_size_weight": 256, "block_size_act": 256,
        "use_awq": False, "awq_alpha": 0.5, "awq_n_grid": 20,
    }

    def __init__(
        self,
        model_path: str,
        output_dir: str = "./quantization_outputs/stage19_combined",
        algorithm_pool: Optional[Dict] = None,
        activation_data_file: Optional[str] = None,
        mme_data_root: Optional[str] = None,
        gpu_ids: Optional[str] = None,
        max_mem_per_gpu: str = "40GiB",
        target_decoder_layers: Optional[List[int]] = None,
        seed: int = 42,
        subsample_step: int = 5,
        num_und_samples: int = 16,
        num_gen_samples: int = 16,
        group_search_order: Optional[List[str]] = None,
        cka_weight: float = 0.4,
        cosine_weight: float = 0.3,
        mse_weight: float = 0.3,
        run_date: Optional[str] = None,
    ):
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.algorithm_pool = algorithm_pool or ALGORITHM_POOL
        self.activation_data_file = activation_data_file
        self.mme_data_root = mme_data_root
        self.gpu_ids = gpu_ids
        self.max_mem_per_gpu = max_mem_per_gpu
        self.target_decoder_layers = target_decoder_layers
        self.seed = seed
        self.subsample_step = subsample_step
        self.num_und_samples = num_und_samples
        self.num_gen_samples = num_gen_samples

        self.cka_weight = cka_weight
        self.cosine_weight = cosine_weight
        self.mse_weight = mse_weight

        self.run_date = run_date or datetime.now().strftime("%Y%m%d")

        self.functional_groups = FUNCTIONAL_GROUPS_INTERNVLU
        if group_search_order:
            name_to_group = {g['name']: g for g in FUNCTIONAL_GROUPS_INTERNVLU}
            self.functional_groups = [name_to_group[n] for n in group_search_order
                                      if n in name_to_group]

        if self.gpu_ids:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu_ids)

        self._set_seed()
        self.activation_data = self._load_activation_data()
        self._load_model()

        self.original_weights = {}
        self._save_original_weights()

        self.num_decoder_layers = len(self.model.language_model.model.layers)
        if self.target_decoder_layers is None:
            self.target_decoder_layers = list(range(self.num_decoder_layers))

        self._print_banner()

    # --------------------------------------------------------
    # Initialization helpers
    # --------------------------------------------------------

    def _set_seed(self):
        import random
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _load_activation_data(self) -> Optional[Dict]:
        if not self.activation_data_file:
            print("\n  No activation data file provided, GPTQ/AWQ will be excluded")
            return None
        path = Path(self.activation_data_file)
        if not path.exists():
            print(f"\n  Activation file not found: {path}, GPTQ/AWQ excluded")
            return None
        try:
            data = torch.load(path, map_location='cpu')
            print(f"  Loaded activation data: {len(data)} layers")
            return data
        except Exception as e:
            print(f"  Failed to load activation data: {e}")
            return None

    def _load_model(self):
        print(f"\nLoading InternVL-U from {self.model_path} ...")
        components = load_internvlu(
            self.model_path,
            gpu_ids=self.gpu_ids,
            torch_dtype=torch.bfloat16,
        )
        self.model = components["model"].eval()
        self.tokenizer = components["tokenizer"]
        self.processor = components["processor"]
        print("  Model loaded successfully")

    def _save_original_weights(self):
        print("\nSaving original weights (language_model.model.layers only) ...")
        count = 0
        for name, module in self.model.named_modules():
            if not name.startswith("language_model.model.layers."):
                continue
            if isinstance(module, nn.Linear) and not isinstance(module, HybridQuantLinear):
                self.original_weights[name] = {
                    'weight': module.weight.data.clone().cpu(),
                    'bias': module.bias.data.clone().cpu() if module.bias is not None else None,
                }
                count += 1
        print(f"  Saved {count} layers")

    def _print_banner(self):
        n_groups = len(self.functional_groups)
        avail_algos = sum(
            1 for a in self.algorithm_pool.values()
            if not (a['config'].get('use_gptq', False) and self.activation_data is None)
            and not (a['config'].get('use_awq', False) and self.activation_data is None)
        )
        total_evals = len(self.target_decoder_layers) * n_groups * avail_algos

        print("\n" + "=" * 80)
        print("Stage 19 (InternVL-U): Combined-Metric Functional-Group Greedy Search")
        print("=" * 80)
        print(f"  Model: {self.model_path}")
        print(f"  Run date: {self.run_date}")
        print(f"  Decoder layers: {self.num_decoder_layers}")
        print(f"  Target layers: {len(self.target_decoder_layers)}")
        print(f"  Functional groups per layer: {n_groups}")
        for g in self.functional_groups:
            print(f"    - {g['name']:12s}: {g['display']} ({len(g['suffixes'])} sublayers)")
        print(f"  Algorithm pool: {avail_algos} available")
        print(f"  Metric weights: CKA={self.cka_weight}, "
              f"Cosine={self.cosine_weight}, MSE={self.mse_weight}")
        print(f"  Calibration: {self.num_und_samples} und + {self.num_gen_samples} gen")
        print(f"  Skip weight bits: {SKIP_WEIGHT_BITS}")
        print(f"  Total CKA evaluations (est): {total_evals}")
        print(f"  Output: {self.output_dir}")
        print("=" * 80 + "\n")

    # --------------------------------------------------------
    # Module helpers
    # --------------------------------------------------------

    def _get_module(self, name: str) -> Optional[nn.Module]:
        parts = name.split('.')
        module = self.model
        for part in parts:
            if hasattr(module, part):
                module = getattr(module, part)
            else:
                return None
        return module

    def _replace_module(self, name: str, new_module: nn.Module):
        parts = name.split('.')
        parent = self.model
        for part in parts[:-1]:
            if hasattr(parent, part):
                parent = getattr(parent, part)
            else:
                return
        setattr(parent, parts[-1], new_module)

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

    # --------------------------------------------------------
    # Quantization apply / restore
    # --------------------------------------------------------

    def _apply_algorithm_to_group(self, layer_idx: int, group: Dict, algo_config: Dict):
        for name in self._get_group_sublayer_names(layer_idx, group):
            self._apply_algorithm_to_sublayer(name, algo_config)

    def _restore_group(self, layer_idx: int, group: Dict):
        for name in self._get_group_sublayer_names(layer_idx, group):
            self._restore_sublayer(name)

    def _apply_algorithm_to_sublayer(self, layer_name: str, algo_config: Dict):
        module = self._get_module(layer_name)
        if module is None or not isinstance(module, (nn.Linear, HybridQuantLinear)):
            return

        if isinstance(module, HybridQuantLinear):
            self._restore_sublayer(layer_name)
            module = self._get_module(layer_name)

        device = module.weight.device
        dtype = module.weight.dtype
        config = algo_config.copy()

        if config.get('use_gptq', False) and self.activation_data is None:
            config['use_gptq'] = False
        if config.get('use_awq', False) and self.activation_data is None:
            config['use_awq'] = False

        quant_layer = HybridQuantLinear(
            in_features=module.in_features,
            out_features=module.out_features,
            bias=module.bias is not None,
            weight_bit=config.get('weight_bit', 4),
            act_bit=config.get('act_bit', 4),
            quant_percentile=config.get('quant_percentile', 0.999999),
            act_unsigned=config.get('act_unsigned', True),
            use_sparse=config.get('use_sparse', False),
            sparse_ratio=config.get('sparse_ratio', 0.0),
            sparse_threshold=config.get('sparse_threshold', None),
            use_smoothquant=config.get('use_smoothquant', False),
            smoothquant_alpha=config.get('smoothquant_alpha', 0.5),
            use_svd=config.get('use_svd', False),
            svd_rank=config.get('svd_rank', 0),
            use_block_quant=config.get('use_block_quant', False),
            use_block_quant_act=config.get('use_block_quant_act', False),
            block_size_weight=config.get('block_size_weight', 256),
            block_size_act=config.get('block_size_act', 256),
            use_gptq=config.get('use_gptq', False),
            gptq_group_size=config.get('gptq_group_size', 64),
            gptq_damp_percentage=config.get('gptq_damp_percentage', 0.01),
            gptq_block_size=config.get('gptq_block_size', 128),
            use_awq=config.get('use_awq', False),
            awq_alpha=config.get('awq_alpha', 0.5),
            awq_n_grid=config.get('awq_n_grid', 20),
            device=device,
            dtype=dtype,
        )

        if layer_name in self.original_weights:
            orig_w = self.original_weights[layer_name]['weight'].clone()
            orig_b = self.original_weights[layer_name]['bias']
            if orig_b is not None:
                orig_b = orig_b.clone()
        else:
            orig_w = module.weight.data.clone()
            orig_b = module.bias.data.clone() if module.bias is not None else None

        quant_layer.weight.data = orig_w.to(device)
        if module.bias is not None and orig_b is not None:
            quant_layer.bias.data = orig_b.to(device)

        quant_layer = quant_layer.to(device)

        act_data = None
        if self.activation_data is not None and layer_name in self.activation_data:
            act_data = self.activation_data[layer_name]

        quant_layer.prepare_weight(
            activation_data=act_data,
            layer_name=layer_name,
            verbose=False,
        )

        self._replace_module(layer_name, quant_layer)

        del act_data
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _restore_sublayer(self, layer_name: str):
        module = self._get_module(layer_name)
        if module is None:
            return
        if isinstance(module, HybridQuantLinear):
            device = module.weight.device if module.weight is not None else 'cuda:0'
            dtype = module.weight.dtype if module.weight is not None else torch.bfloat16
            linear = nn.Linear(
                module.in_features, module.out_features,
                bias=module.bias is not None,
                device=device, dtype=dtype,
            )
            if layer_name in self.original_weights:
                linear.weight.data.copy_(self.original_weights[layer_name]['weight'].to(device))
                if linear.bias is not None and self.original_weights[layer_name]['bias'] is not None:
                    linear.bias.data.copy_(self.original_weights[layer_name]['bias'].to(device))
            self._replace_module(layer_name, linear)
        elif isinstance(module, nn.Linear):
            if layer_name in self.original_weights:
                module.weight.data.copy_(
                    self.original_weights[layer_name]['weight'].to(module.weight.device)
                )
                if module.bias is not None and self.original_weights[layer_name]['bias'] is not None:
                    module.bias.data.copy_(
                        self.original_weights[layer_name]['bias'].to(module.bias.device)
                    )

    def _restore_decoder_layer(self, layer_idx: int):
        for name in self._get_decoder_layer_linear_names(layer_idx):
            self._restore_sublayer(name)

    def _redispatch_model(self):
        try:
            from accelerate import dispatch_model
            if hasattr(self.model, 'hf_device_map'):
                self.model = dispatch_model(
                    self.model, device_map=self.model.hf_device_map, offload_dir=None,
                )
        except Exception as e:
            print(f"  [Warning] Re-dispatch failed: {e}")

    # --------------------------------------------------------
    # Forward pass & hidden state collection
    # --------------------------------------------------------

    def _forward_calibration_sample(self, sample: Dict):
        """UND / GEN 通用前向：根据 sample['generation_mode'] 自动切换。"""
        prompt = sample.get("prompt", "")
        image = sample.get("image")
        gen_mode = sample.get("generation_mode", "text")
        pil_image = image_for_processor(image)

        inputs = self.processor(
            prompt=[prompt],
            image=[pil_image] if pil_image is not None else [None],
            generation_mode=gen_mode,
            padding=True,
            return_tensors="pt",
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
        hook_target_name = group.get('hook_target')
        decoder_layer = self.model.language_model.model.layers[layer_idx]

        if hook_target_name:
            target_module = getattr(decoder_layer, hook_target_name, None)
        else:
            target_module = None

        if target_module is None:
            target_module = decoder_layer

        captured_list = []
        captured = {}

        def hook_fn(module, input, output):
            h = output[0] if isinstance(output, tuple) else output
            captured['output'] = h.detach().cpu()

        handle = target_module.register_forward_hook(hook_fn)

        try:
            with torch.no_grad():
                for sample in calibration_samples:
                    try:
                        self._forward_calibration_sample(sample)
                    except Exception:
                        captured.clear()
                        continue
                    if 'output' in captured:
                        captured_list.append(captured['output'])
                    captured.clear()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        finally:
            handle.remove()

        return captured_list

    # --------------------------------------------------------
    # Core: Functional Group Greedy Search with Combined Metric
    # --------------------------------------------------------

    @staticmethod
    def _group_algos_by_wbit(pool: Dict) -> Dict[int, Dict]:
        groups: Dict[int, Dict] = {}
        for key, info in pool.items():
            wb = info["config"].get("weight_bit", 4)
            if wb in SKIP_WEIGHT_BITS:
                continue
            groups.setdefault(wb, {})[key] = info
        return dict(sorted(groups.items(), reverse=True))

    def _compute_similarity(
        self,
        ref_hs: List[torch.Tensor],
        quant_hs: List[torch.Tensor],
    ) -> Dict[str, float]:
        if not quant_hs:
            return {'combined': 0.0, 'cka': 0.0, 'cosine': 0.0, 'mse_score': 0.0}
        return CombinedSimilarity.compute_batched_detailed(
            ref_hs, quant_hs,
            subsample_step=self.subsample_step,
            cka_weight=self.cka_weight,
            cosine_weight=self.cosine_weight,
            mse_weight=self.mse_weight,
        )

    def _search_single_bitwidth(
        self,
        wbit: int,
        algos: Dict,
        calibration_samples: List[Dict],
    ) -> Dict:
        tag = f"W{wbit}A4"
        fallback = f"rtn_w{wbit}a4"

        print(f"\n{'=' * 70}")
        print(f"  [{tag}] Combined-Metric Functional Group Search ({len(algos)} algorithms)")
        print(f"{'=' * 70}")

        group_assignments = {}
        group_scores = {}
        search_log = []
        progress_file = self.output_dir / f"search_progress_w{wbit}a4_{self.run_date}.json"

        global_eval_idx = 0
        total_evals = len(self.target_decoder_layers) * len(self.functional_groups) * len(algos)

        for layer_idx in self.target_decoder_layers:
            all_linear_names = self._get_decoder_layer_linear_names(layer_idx)

            print(f"\n  {'─' * 55}")
            print(f"  [{tag}] Decoder Layer {layer_idx}/{self.num_decoder_layers - 1} "
                  f"({len(all_linear_names)} sublayers, {len(self.functional_groups)} groups)")

            layer_start = datetime.now()

            for group_idx, group in enumerate(self.functional_groups):
                group_names = self._get_group_sublayer_names(layer_idx, group)
                existing = [n for n in group_names if self._get_module(n) is not None]
                if not existing:
                    continue

                print(f"\n    [{tag}] Group {group_idx+1}/{len(self.functional_groups)}: "
                      f"{group['display']} ({len(existing)} sublayers)")

                ref_hs = self._collect_group_hidden_states(
                    layer_idx, group, calibration_samples
                )
                if not ref_hs:
                    for ln in existing:
                        group_assignments[ln] = fallback
                    continue

                best_algo = None
                best_combined = -float('inf')
                best_detail = {}
                algo_results = {}

                for algo_key, algo_info in algos.items():
                    global_eval_idx += 1
                    try:
                        for ln in existing:
                            self._apply_algorithm_to_sublayer(ln, algo_info['config'])
                        self._redispatch_model()

                        quant_hs = self._collect_group_hidden_states(
                            layer_idx, group, calibration_samples
                        )
                        detail = self._compute_similarity(ref_hs, quant_hs)
                        score = detail['combined']

                        algo_results[algo_key] = {
                            k: round(v, 6) for k, v in detail.items()
                        }
                        print(f"      [{global_eval_idx}/{total_evals}] "
                              f"{algo_key:25s}: combined={score:.6f} "
                              f"(CKA={detail['cka']:.4f} cos={detail['cosine']:.4f} "
                              f"mse_s={detail['mse_score']:.4f})")

                        if score > best_combined:
                            best_combined = score
                            best_algo = algo_key
                            best_detail = detail
                    except Exception as e:
                        print(f"      [{global_eval_idx}/{total_evals}] "
                              f"{algo_key:25s}: FAILED ({e})")
                        algo_results[algo_key] = {
                            'combined': -1.0, 'cka': -1.0,
                            'cosine': -1.0, 'mse_score': -1.0,
                        }
                    finally:
                        self._restore_group(layer_idx, group)
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                if best_algo is None:
                    best_algo = fallback
                    best_combined = algo_results.get(fallback, {}).get('combined', 0.0)

                for ln in existing:
                    group_assignments[ln] = best_algo
                for ln in existing:
                    self._apply_algorithm_to_sublayer(ln, algos[best_algo]['config'])
                self._redispatch_model()

                group_scores[(layer_idx, group['name'])] = algo_results
                print(f"      >>> [{tag}] Best: {best_algo} (combined={best_combined:.6f})")

                search_log.append({
                    'decoder_layer_idx': layer_idx,
                    'group_name': group['name'],
                    'group_display': group['display'],
                    'sublayer_names': existing,
                    'best_algo': best_algo,
                    'best_scores': {k: round(v, 6) for k, v in best_detail.items()},
                    'all_algo_results': algo_results,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                })

                del ref_hs
                gc.collect()

            elapsed = (datetime.now() - layer_start).total_seconds()
            print(f"\n    [{tag}] Layer {layer_idx} done in {elapsed:.1f}s")
            for g in self.functional_groups:
                gnames = self._get_group_sublayer_names(layer_idx, g)
                algo = group_assignments.get(gnames[0], '?') if gnames else '?'
                results = group_scores.get((layer_idx, g['name']), {})
                best_r = results.get(algo, {})
                score_str = f"combined={best_r.get('combined', 0.0):.6f}"
                print(f"      {g['name']:12s}: {algo:25s} ({score_str})")

            with open(progress_file, 'w') as f:
                json.dump({
                    'weight_bit': wbit,
                    'run_date': self.run_date,
                    'total_evals': global_eval_idx,
                    'search_log': search_log,
                }, f, indent=2)

        # Restore all layers to FP
        for layer_idx in self.target_decoder_layers:
            for ln in self._get_decoder_layer_linear_names(layer_idx):
                self._restore_sublayer(ln)
        self._redispatch_model()

        for layer_idx in self.target_decoder_layers:
            for ln in self._get_decoder_layer_linear_names(layer_idx):
                if ln not in group_assignments:
                    group_assignments[ln] = fallback

        serializable_scores = {
            f"{k[0]}_{k[1]}": v for k, v in group_scores.items()
        }

        return {
            'weight_bit': wbit,
            'sublayer_assignments': group_assignments,
            'group_scores': serializable_scores,
            'search_log': search_log,
        }

    def search(self) -> Dict:
        # ---- Phase 1: Calibration (UND + GEN) ----
        print("\n" + "=" * 80)
        print("Phase 1: Preparing calibration data (und + gen)")
        print("=" * 80)

        calib_loader = CalibrationDataLoader(
            num_und_samples=self.num_und_samples,
            num_gen_samples=self.num_gen_samples,
            mme_data_root=self.mme_data_root,
        )
        calibration_samples = calib_loader.prepare_calibration_samples()

        # ---- Phase 2: Algorithm setup ----
        available_algos = {}
        for key, algo in self.algorithm_pool.items():
            if algo['config'].get('use_gptq', False) and self.activation_data is None:
                continue
            if algo['config'].get('use_awq', False) and self.activation_data is None:
                continue
            available_algos[key] = algo

        wbit_groups = self._group_algos_by_wbit(available_algos)
        print(f"\n  Algorithm groups by weight-bit (W2 skipped):")
        for wb, grp in wbit_groups.items():
            print(f"    W{wb}A4: {len(grp)} algos — {list(grp.keys())}")

        # ---- Phase 3: Per-bitwidth functional group search ----
        print("\n" + "=" * 80)
        print("Phase 3: Per-bitwidth Combined-Metric Functional Group Greedy Search")
        print("=" * 80)

        all_bitwidth_results = {}
        config_export_dir = Path(self.output_dir).parent / "configs"
        config_export_dir.mkdir(parents=True, exist_ok=True)

        for wbit, algos in wbit_groups.items():
            tag = f"w{wbit}a4"
            bw_result = self._search_single_bitwidth(wbit, algos, calibration_samples)
            all_bitwidth_results[wbit] = bw_result

            exported_path = self.export_quantization_config(
                bw_result['sublayer_assignments'],
                export_dir=str(config_export_dir),
                tag=tag,
            )
            bw_result['exported_config_path'] = exported_path

        # ---- Phase 4: Save combined results ----
        results = {
            'bitwidth_results': {str(wb): res for wb, res in all_bitwidth_results.items()},
            'metadata': {
                'stage': 'stage19',
                'strategy': 'combined_metric_funcgroup_greedy',
                'run_date': self.run_date,
                'model_path': self.model_path,
                'model_type': 'internvlu',
                'num_decoder_layers': self.num_decoder_layers,
                'target_layers': self.target_decoder_layers,
                'functional_groups': [g['name'] for g in self.functional_groups],
                'algorithm_pool': list(self.algorithm_pool.keys()),
                'bitwidths_searched': sorted(all_bitwidth_results.keys(), reverse=True),
                'skipped_bitwidths': sorted(SKIP_WEIGHT_BITS),
                'metric_weights': {
                    'cka': self.cka_weight,
                    'cosine': self.cosine_weight,
                    'mse': self.mse_weight,
                },
                'calibration': {
                    'num_und_samples': self.num_und_samples,
                    'num_gen_samples': self.num_gen_samples,
                },
                'subsample_step': self.subsample_step,
                'seed': self.seed,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            },
        }
        output_file = self.output_dir / f"combined_search_results_{self.run_date}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n{'=' * 70}")
        print(f"Search Summary (run_date={self.run_date}):")
        for wb, res in sorted(all_bitwidth_results.items(), reverse=True):
            assigns = res['sublayer_assignments']
            algo_counts = defaultdict(int)
            for a in assigns.values():
                algo_counts[a] += 1
            top3 = sorted(algo_counts.items(), key=lambda x: -x[1])[:3]
            print(f"  W{wb}A4: {', '.join(f'{a}({c})' for a, c in top3)}")
            print(f"         config → {res.get('exported_config_path', 'N/A')}")
        print(f"{'=' * 70}\n")

        return results

    # --------------------------------------------------------
    # Config building / export
    # --------------------------------------------------------

    def _build_full_sublayer_config(self, algo_key: str) -> Dict:
        cfg = dict(self._FULL_CONFIG_TEMPLATE)
        if algo_key in self.algorithm_pool:
            cfg.update(self.algorithm_pool[algo_key]['config'])
        if not cfg.get('use_svd', False):
            cfg['svd_rank'] = 0
        if not cfg.get('use_sparse', False):
            cfg['sparse_threshold'] = None
            cfg['sparse_ratio'] = 0.0
        return cfg

    def export_quantization_config(
        self,
        sublayer_assignments: Dict[str, str],
        export_dir: Optional[str] = None,
        tag: Optional[str] = None,
    ) -> str:
        if export_dir is None:
            export_dir = str(self.output_dir)
        export_path = Path(export_dir)
        export_path.mkdir(parents=True, exist_ok=True)

        config = {}
        for layer_name, algo_key in sorted(sublayer_assignments.items()):
            config[layer_name] = self._build_full_sublayer_config(algo_key)

        sorted_config = dict(sorted(config.items()))

        if tag:
            filename = f"combined_funcgroup_{tag}_{self.run_date}.json"
        else:
            filename = f"combined_funcgroup_{self.run_date}.json"
        filepath = export_path / filename

        with open(filepath, 'w') as f:
            json.dump(sorted_config, f, indent=2)

        print(f"\n  Exported quantization config: {filepath}")
        print(f"    Total sublayers: {len(sorted_config)}")

        algo_counts = defaultdict(int)
        for algo in sublayer_assignments.values():
            algo_counts[algo] += 1
        print(f"    Algorithm distribution:")
        for algo, cnt in sorted(algo_counts.items(), key=lambda x: -x[1]):
            pct = 100.0 * cnt / len(sublayer_assignments)
            print(f"      {algo:25s}: {cnt:4d} sublayers ({pct:.1f}%)")

        return str(filepath)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Stage 19 (InternVL-U): Combined-Metric Functional-Group Greedy Search"
    )

    subparsers = parser.add_subparsers(dest='command', help='Sub-commands')

    sp = subparsers.add_parser('search', help='Run combined-metric functional group search')
    sp.add_argument('--model_path', type=str, required=True)
    sp.add_argument('--output_dir', type=str,
                    default='./quantization_outputs/stage19_combined')
    sp.add_argument('--activation_data', type=str, default=None)
    sp.add_argument('--mme_data_root', type=str, default=None)
    sp.add_argument('--num_und_samples', type=int, default=16)
    sp.add_argument('--num_gen_samples', type=int, default=16)
    sp.add_argument('--gpu_ids', type=str, default=None)
    sp.add_argument('--max_mem_per_gpu', type=str, default='40GiB')
    sp.add_argument('--target_layers', type=str, default=None)
    sp.add_argument('--algorithms', type=str, default=None)
    sp.add_argument('--subsample_step', type=int, default=5)
    sp.add_argument('--seed', type=int, default=42)
    sp.add_argument('--config', type=str, default=None)
    sp.add_argument('--group_order', type=str, default=None,
                    help='Comma-separated group search order, e.g. "attn_shared,mlp_shared"')
    sp.add_argument('--cka_weight', type=float, default=0.4)
    sp.add_argument('--cosine_weight', type=float, default=0.3)
    sp.add_argument('--mse_weight', type=float, default=0.3)
    sp.add_argument('--run_date', type=str, default=None,
                    help='Override run date (YYYYMMDD). Default: today.')

    ep = subparsers.add_parser('export', help='Export config from existing results')
    ep.add_argument('--results_file', type=str, required=True)
    ep.add_argument('--export_dir', type=str, required=True)
    ep.add_argument('--run_date', type=str, default=None)

    args = parser.parse_args()

    if args.command is None:
        sp_compat = argparse.ArgumentParser()
        sp_compat.add_argument('--model_path', type=str, required=True)
        sp_compat.add_argument('--output_dir', type=str,
                               default='./quantization_outputs/stage19_combined')
        sp_compat.add_argument('--activation_data', type=str, default=None)
        sp_compat.add_argument('--mme_data_root', type=str, default=None)
        sp_compat.add_argument('--num_und_samples', type=int, default=16)
        sp_compat.add_argument('--num_gen_samples', type=int, default=16)
        sp_compat.add_argument('--gpu_ids', type=str, default=None)
        sp_compat.add_argument('--max_mem_per_gpu', type=str, default='40GiB')
        sp_compat.add_argument('--target_layers', type=str, default=None)
        sp_compat.add_argument('--algorithms', type=str, default=None)
        sp_compat.add_argument('--subsample_step', type=int, default=5)
        sp_compat.add_argument('--seed', type=int, default=42)
        sp_compat.add_argument('--config', type=str, default=None)
        sp_compat.add_argument('--group_order', type=str, default=None)
        sp_compat.add_argument('--cka_weight', type=float, default=0.4)
        sp_compat.add_argument('--cosine_weight', type=float, default=0.3)
        sp_compat.add_argument('--mse_weight', type=float, default=0.3)
        sp_compat.add_argument('--run_date', type=str, default=None)
        args = sp_compat.parse_args()
        args.command = 'search'

    if args.command == 'export':
        run_date = args.run_date or datetime.now().strftime("%Y%m%d")
        print(f"Loading results from {args.results_file} ...")
        with open(args.results_file, 'r') as f:
            results = json.load(f)
        assignments = results.get('sublayer_assignments', {})
        template = CombinedFuncGroupSearcher._FULL_CONFIG_TEMPLATE
        config = {}
        for ln, ak in sorted(assignments.items()):
            cfg = dict(template)
            if ak in ALGORITHM_POOL:
                cfg.update(ALGORITHM_POOL[ak]['config'])
            if not cfg.get('use_svd', False):
                cfg['svd_rank'] = 0
            if not cfg.get('use_sparse', False):
                cfg['sparse_threshold'] = None
                cfg['sparse_ratio'] = 0.0
            config[ln] = cfg
        export_path = Path(args.export_dir)
        export_path.mkdir(parents=True, exist_ok=True)
        filepath = export_path / f"exported_config_{run_date}.json"
        with open(filepath, 'w') as f:
            json.dump(dict(sorted(config.items())), f, indent=2)
        print(f"  Exported: {filepath} ({len(config)} sublayers)")
        return

    if hasattr(args, 'config') and args.config:
        with open(args.config, 'r') as f:
            yaml_config = yaml.safe_load(f)
        for key, val in yaml_config.items():
            if hasattr(args, key) and getattr(args, key) is None:
                setattr(args, key, val)

    target_layers = None
    if args.target_layers:
        target_layers = [int(x.strip()) for x in args.target_layers.split(',')]

    algo_pool = ALGORITHM_POOL
    if args.algorithms:
        selected = [x.strip() for x in args.algorithms.split(',')]
        algo_pool = {k: v for k, v in ALGORITHM_POOL.items() if k in selected}
        if not algo_pool:
            print(f"Warning: no valid algorithms in '{args.algorithms}', using full pool")
            algo_pool = ALGORITHM_POOL

    group_order = None
    if hasattr(args, 'group_order') and args.group_order:
        group_order = [g.strip() for g in args.group_order.split(',')]

    searcher = CombinedFuncGroupSearcher(
        model_path=args.model_path,
        output_dir=args.output_dir,
        algorithm_pool=algo_pool,
        activation_data_file=args.activation_data,
        mme_data_root=args.mme_data_root,
        gpu_ids=args.gpu_ids,
        max_mem_per_gpu=args.max_mem_per_gpu,
        target_decoder_layers=target_layers,
        seed=args.seed,
        subsample_step=args.subsample_step,
        num_und_samples=getattr(args, 'num_und_samples', 16),
        num_gen_samples=getattr(args, 'num_gen_samples', 16),
        group_search_order=group_order,
        cka_weight=args.cka_weight,
        cosine_weight=args.cosine_weight,
        mse_weight=args.mse_weight,
        run_date=args.run_date,
    )

    results = searcher.search()

    print("\nCombined-metric functional group search (InternVL-U) completed!")
    print(f"Results: {searcher.output_dir / f'combined_search_results_{searcher.run_date}.json'}")


if __name__ == "__main__":
    main()
