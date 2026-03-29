"""
Stage 17 (InternVL-U): Sublayer-Level Exhaustive CKA Search

适配 InternVL-U：
- 仅量化 language_model.model.layers（Qwen3/Qwen2 decoder）
- 支持 UND + GEN 混合校准
- 无 MoE 分支，标准 Qwen/Llama decoder 结构

搜索粒度: sublayer (单个 Linear 层)
搜索策略: 暴力遍历 — 每个 sublayer × 每种算法
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

from utils.bagel_imports import HybridQuantLinear, LinearCKA, ALGORITHM_POOL
from utils.model_loader import load_internvlu
from utils.calibration import CalibrationDataLoader, image_for_processor

from PIL import Image


# ============================================================
# Exhaustive Sublayer Searcher (InternVL-U)
# ============================================================

class ExhaustiveSublayerSearcher:
    """
    Sublayer-Level Exhaustive CKA Search, adapted for InternVL-U.
    Only understanding pathway calibration (no T2I gen).
    Standard Qwen/Llama decoder — no MoE branches.
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
        output_dir: str = "./quantization_outputs/stage17_exhaustive_internvlu",
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
        self.run_date = run_date or datetime.now().strftime("%Y%m%d")

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
            print("\n  No activation data file provided, GPTQ will be excluded")
            return None
        path = Path(self.activation_data_file)
        if not path.exists():
            print(f"\n  Activation file not found: {path}, GPTQ excluded")
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
        total_sublayers = sum(
            len(self._get_decoder_layer_linear_names(i))
            for i in self.target_decoder_layers
        )
        avail_algos = sum(
            1 for a in self.algorithm_pool.values()
            if not (a['config'].get('use_gptq', False) and self.activation_data is None)
        )
        total_evals = total_sublayers * avail_algos

        print("\n" + "=" * 80)
        print("Stage 17 (InternVL-U): Sublayer-Level Exhaustive CKA Search")
        print("=" * 80)
        print(f"  Model: {self.model_path}")
        print(f"  Decoder layers: {self.num_decoder_layers}")
        print(f"  Target layers: {len(self.target_decoder_layers)}")
        print(f"  Total sublayers: {total_sublayers}")
        print(f"  Algorithm pool: {list(self.algorithm_pool.keys())} "
              f"({avail_algos} available)")
        print(f"  Total CKA evaluations: {total_evals}")
        print(f"  Run date: {self.run_date}")
        print(f"  Calibration: {self.num_und_samples} und + {self.num_gen_samples} gen")
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

    # --------------------------------------------------------
    # Quantization apply / restore
    # --------------------------------------------------------

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
        gen_mode = sample.get("generation_mode", "text")
        image = sample.get("image")
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

    def _collect_layer_hidden_states(
        self, layer_idx: int, calibration_samples: List[Dict],
    ) -> List[torch.Tensor]:
        captured_list = []
        captured = {}

        def hook_fn(module, input, output):
            h = output[0] if isinstance(output, tuple) else output
            captured['output'] = h.detach().cpu()

        decoder_layer = self.model.language_model.model.layers[layer_idx]
        handle = decoder_layer.register_forward_hook(hook_fn)

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
    # Core: Exhaustive Search
    # --------------------------------------------------------

    @staticmethod
    def _group_algos_by_wbit(pool: Dict) -> Dict[int, Dict]:
        groups: Dict[int, Dict] = {}
        for key, info in pool.items():
            wb = info["config"].get("weight_bit", 4)
            groups.setdefault(wb, {})[key] = info
        return dict(sorted(groups.items(), reverse=True))

    def _search_single_bitwidth(
        self,
        wbit: int,
        algos: Dict,
        calibration_samples: List[Dict],
    ) -> Dict:
        """在单一位宽下执行子层穷举搜索。搜索完后恢复所有层到 FP。"""
        tag = f"W{wbit}A4"
        fallback = f"rtn_w{wbit}a4"

        print(f"\n{'=' * 70}")
        print(f"  [{tag}] Exhaustive Sublayer Search ({len(algos)} algorithms)")
        print(f"{'=' * 70}")

        sublayer_assignments = {}
        sublayer_cka_scores = {}
        search_log = []
        progress_file = self.output_dir / f"search_progress_w{wbit}a4_{self.run_date}.json"

        total_sublayers = sum(
            len(self._get_decoder_layer_linear_names(i))
            for i in self.target_decoder_layers
        )
        global_sublayer_idx = 0

        for layer_idx in self.target_decoder_layers:
            linear_names = self._get_decoder_layer_linear_names(layer_idx)
            n_sublayers = len(linear_names)

            print(f"\n  {'─' * 55}")
            print(f"  [{tag}] Decoder Layer {layer_idx}/{self.num_decoder_layers - 1} "
                  f"({n_sublayers} sublayers)")

            fp_hs = self._collect_layer_hidden_states(layer_idx, calibration_samples)
            if not fp_hs:
                for ln in linear_names:
                    sublayer_assignments[ln] = fallback
                global_sublayer_idx += n_sublayers
                continue

            layer_start = datetime.now()

            for sub_idx, layer_name in enumerate(linear_names):
                global_sublayer_idx += 1
                short_name = layer_name.split(f"layers.{layer_idx}.")[-1]
                print(f"\n    [{global_sublayer_idx}/{total_sublayers}] {short_name}")

                best_algo = None
                best_cka = -1.0
                algo_scores = {}

                for algo_key, algo_info in algos.items():
                    try:
                        self._apply_algorithm_to_sublayer(layer_name, algo_info['config'])
                        self._redispatch_model()
                        quant_hs = self._collect_layer_hidden_states(layer_idx, calibration_samples)
                        cka = LinearCKA.compute_batched(
                            fp_hs, quant_hs, subsample_step=self.subsample_step
                        ) if quant_hs else 0.0
                        algo_scores[algo_key] = round(cka, 6)
                        print(f"      {algo_key:25s}: CKA={cka:.6f}")
                        if cka > best_cka:
                            best_cka = cka
                            best_algo = algo_key
                    except Exception as e:
                        print(f"      {algo_key:25s}: FAILED ({e})")
                        algo_scores[algo_key] = -1.0
                    finally:
                        self._restore_sublayer(layer_name)
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                if best_algo is None:
                    best_algo = fallback
                    best_cka = algo_scores.get(fallback, 0.0)

                sublayer_assignments[layer_name] = best_algo
                sublayer_cka_scores[layer_name] = algo_scores
                print(f"      >>> [{tag}] Best: {best_algo} (CKA={best_cka:.6f})")

                search_log.append({
                    'layer_name': layer_name, 'decoder_layer_idx': layer_idx,
                    'sublayer_type': short_name, 'best_algo': best_algo,
                    'best_cka': best_cka, 'all_scores': algo_scores,
                    'global_idx': global_sublayer_idx,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                })

            # Apply best for this layer (for progressive reference)
            for ln in linear_names:
                ak = sublayer_assignments[ln]
                if ak in algos:
                    self._apply_algorithm_to_sublayer(ln, algos[ak]['config'])
            self._redispatch_model()

            elapsed = (datetime.now() - layer_start).total_seconds()
            print(f"\n    [{tag}] Layer {layer_idx} done in {elapsed:.1f}s")
            self._print_layer_summary(layer_idx, linear_names, sublayer_assignments, sublayer_cka_scores)

            with open(progress_file, 'w') as f:
                json.dump({
                    'weight_bit': wbit,
                    'total_sublayers_searched': global_sublayer_idx,
                    'total_sublayers': total_sublayers,
                    'search_log': search_log,
                }, f, indent=2)

            del fp_hs
            gc.collect()

        # Restore all layers to FP before next bitwidth
        for layer_idx in self.target_decoder_layers:
            for ln in self._get_decoder_layer_linear_names(layer_idx):
                self._restore_sublayer(ln)
        self._redispatch_model()

        return {
            'weight_bit': wbit,
            'sublayer_assignments': sublayer_assignments,
            'sublayer_cka_scores': sublayer_cka_scores,
            'search_log': search_log,
        }

    def search(self) -> Dict:
        # ---- Phase 1: Calibration ----
        print("\n" + "=" * 80)
        print("Phase 1: Preparing calibration data (und only)")
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
        print(f"\n  Algorithm groups by weight-bit:")
        for wb, grp in wbit_groups.items():
            print(f"    W{wb}A4: {len(grp)} algos — {list(grp.keys())}")

        # ---- Phase 3: Per-bitwidth exhaustive search ----
        print("\n" + "=" * 80)
        print("Phase 3: Per-bitwidth Exhaustive Sublayer Search")
        print("=" * 80)

        all_bitwidth_results = {}
        config_export_dir = Path(self.output_dir).parent / "configs"
        config_export_dir.mkdir(parents=True, exist_ok=True)

        for wbit, algos in wbit_groups.items():
            tag = f"w{wbit}a4"
            bw_result = self._search_single_bitwidth(wbit, algos, calibration_samples)
            all_bitwidth_results[wbit] = bw_result

            # Export this bitwidth's config
            exported_path = self.export_quantization_config(
                bw_result['sublayer_assignments'],
                export_dir=str(config_export_dir),
                tag=tag,
            )
            bw_result['exported_config_path'] = exported_path

        # ---- Phase 4: Save combined results ----
        total_sublayers = sum(
            len(self._get_decoder_layer_linear_names(i))
            for i in self.target_decoder_layers
        )
        results = {
            'bitwidth_results': {str(wb): res for wb, res in all_bitwidth_results.items()},
            'metadata': {
                'model_path': self.model_path, 'model_type': 'internvlu',
                'run_date': self.run_date,
                'num_decoder_layers': self.num_decoder_layers,
                'target_layers': self.target_decoder_layers,
                'algorithm_pool': list(self.algorithm_pool.keys()),
                'bitwidths_searched': sorted(all_bitwidth_results.keys(), reverse=True),
                'num_und_samples': self.num_und_samples,
                'num_gen_samples': self.num_gen_samples,
                'subsample_step': self.subsample_step,
                'seed': self.seed, 'total_sublayers': total_sublayers,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            },
        }
        output_file = self.output_dir / f"exhaustive_search_results_{self.run_date}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n{'=' * 70}")
        print("Search Summary:")
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

    def _build_final_config(self, sublayer_assignments: Dict[str, str]) -> Dict[str, Dict]:
        final_config = {}
        for layer_name, algo_key in sublayer_assignments.items():
            full_cfg = self._build_full_sublayer_config(algo_key)
            full_cfg['_algorithm'] = algo_key
            final_config[layer_name] = full_cfg
        return final_config

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
            filename = f"exhaustive_sublayer_{tag}_{self.run_date}.json"
        else:
            filename = f"exhaustive_sublayer_{self.run_date}.json"
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

    # --------------------------------------------------------
    # Summary printing
    # --------------------------------------------------------

    def _print_layer_summary(self, layer_idx, linear_names, assignments, cka_scores):
        algo_counts = defaultdict(int)
        for ln in linear_names:
            algo_counts[assignments.get(ln, 'rtn_w4a4')] += 1
        if len(algo_counts) == 1:
            algo, cnt = list(algo_counts.items())[0]
            print(f"    Summary: all {cnt} sublayers -> {algo}")
        else:
            parts = ", ".join(f"{a}:{c}" for a, c in sorted(algo_counts.items(), key=lambda x: -x[1]))
            print(f"    Summary: [{parts}]")

    def _print_summary(self, sublayer_assignments, sublayer_cka_scores):
        print("\n" + "=" * 80)
        print("Exhaustive Sublayer Search Results Summary (InternVL-U)")
        print("=" * 80)

        algo_counts = defaultdict(int)
        for algo in sublayer_assignments.values():
            algo_counts[algo] += 1

        print(f"\n  Algorithm distribution (sublayer-level):")
        for algo, count in sorted(algo_counts.items(), key=lambda x: -x[1]):
            pct = 100.0 * count / len(sublayer_assignments)
            print(f"    {algo:25s}: {count:4d} sublayers ({pct:.1f}%)")

        per_layer_algos = defaultdict(lambda: defaultdict(int))
        for ln, algo in sublayer_assignments.items():
            parts = ln.split('.')
            for i, p in enumerate(parts):
                if p == 'layers' and i + 1 < len(parts):
                    layer_idx = int(parts[i + 1])
                    per_layer_algos[layer_idx][algo] += 1
                    break

        print(f"\n  Per decoder-layer heterogeneity:")
        for li in sorted(per_layer_algos.keys()):
            algos = per_layer_algos[li]
            if len(algos) == 1:
                algo, cnt = list(algos.items())[0]
                print(f"    Layer {li:3d}: {algo} (all {cnt} sublayers)")
            else:
                parts = ", ".join(
                    f"{a}:{c}" for a, c in sorted(algos.items(), key=lambda x: -x[1])
                )
                print(f"    Layer {li:3d}: [{parts}]")

        all_best_ckas = []
        for ln, scores in sublayer_cka_scores.items():
            best = max(scores.values()) if scores else 0.0
            all_best_ckas.append(best)

        if all_best_ckas:
            print(f"\n  CKA statistics (best per sublayer):")
            print(f"    Mean:   {np.mean(all_best_ckas):.6f}")
            print(f"    Median: {np.median(all_best_ckas):.6f}")
            print(f"    Min:    {np.min(all_best_ckas):.6f}")
            print(f"    Max:    {np.max(all_best_ckas):.6f}")

        worst = sorted(
            sublayer_cka_scores.items(),
            key=lambda x: max(x[1].values()) if x[1] else 0.0,
        )[:10]
        if worst:
            print(f"\n  Top-10 hardest sublayers (lowest best CKA):")
            for ln, scores in worst:
                best = max(scores.values()) if scores else 0.0
                algo = sublayer_assignments.get(ln, '?')
                short = ln.split('layers.')[-1] if 'layers.' in ln else ln
                print(f"    {short:50s}: CKA={best:.6f} ({algo})")

        print(f"\n  Results saved to: {self.output_dir / 'exhaustive_search_results.json'}")
        print("=" * 80)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Stage 17 (InternVL-U): Sublayer-Level Exhaustive CKA Search"
    )

    subparsers = parser.add_subparsers(dest='command', help='Sub-commands')

    sp = subparsers.add_parser('search', help='Run exhaustive sublayer search')
    sp.add_argument('--model_path', type=str, required=True)
    sp.add_argument('--output_dir', type=str,
                    default='./quantization_outputs/stage17_exhaustive_internvlu')
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
    sp.add_argument('--run_date', type=str, default=None,
                    help='Override run date (YYYYMMDD). Default: today.')

    ep = subparsers.add_parser('export', help='Export config from existing results')
    ep.add_argument('--results_file', type=str, required=True)
    ep.add_argument('--export_dir', type=str, required=True)

    args = parser.parse_args()

    if args.command is None:
        sp_compat = argparse.ArgumentParser()
        sp_compat.add_argument('--model_path', type=str, required=True)
        sp_compat.add_argument('--output_dir', type=str,
                               default='./quantization_outputs/stage17_exhaustive_internvlu')
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
        sp_compat.add_argument('--run_date', type=str, default=None)
        args = sp_compat.parse_args()
        args.command = 'search'

    if args.command == 'export':
        print(f"Loading results from {args.results_file} ...")
        with open(args.results_file, 'r') as f:
            results = json.load(f)
        assignments = results.get('sublayer_assignments', {})
        template = ExhaustiveSublayerSearcher._FULL_CONFIG_TEMPLATE
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
        filepath = export_path / "exported_config.json"
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

    searcher = ExhaustiveSublayerSearcher(
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
        run_date=getattr(args, 'run_date', None),
    )

    results = searcher.search()

    print("\nExhaustive sublayer search (InternVL-U) completed!")
    print(f"Results: {searcher.output_dir / f'exhaustive_search_results_{searcher.run_date}.json'}")


if __name__ == "__main__":
    main()
