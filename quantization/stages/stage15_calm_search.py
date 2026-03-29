"""
Stage 15 (InternVL-U): CALM — CKA-Guided Adaptive Layer-Wise Modularization

仅量化 language_model.model.layers（Qwen3/Qwen2 decoder），
支持 UND + GEN 混合校准。
"""

import os
import sys
import json
import gc
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from tqdm import tqdm
import yaml
import argparse

_current_dir = Path(__file__).resolve().parent
_quant_dir = _current_dir.parent

if str(_quant_dir) not in sys.path:
    sys.path.insert(0, str(_quant_dir))

from utils.bagel_imports import HybridQuantLinear, LinearCKA, ALGORITHM_POOL
from utils.model_loader import load_internvlu
from utils.calibration import CalibrationDataLoader, image_for_processor


class CALMSearcher:
    """CALM 逐层贪心搜索，适配 InternVL-U。支持 UND + GEN 混合校准。"""

    def __init__(
        self,
        model_path: str,
        output_dir: str = "./quantization_outputs/stage15_calm",
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
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_ids)

        self._set_seed()
        self.activation_data = self._load_activation_data()
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

    def _load_activation_data(self) -> Optional[Dict]:
        if not self.activation_data_file:
            print("\n  No activation data file provided, GPTQ will be excluded from pool")
            return None
        path = Path(self.activation_data_file)
        if not path.exists():
            print(f"\n  Activation file not found: {path}, GPTQ excluded")
            return None
        try:
            data = torch.load(path, map_location="cpu")
            print(f"  Loaded activation data: {len(data)} layers")
            return data
        except Exception as e:
            print(f"  Failed to load activation data: {e}")
            return None

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
        print("\nSaving original weights (language_model.model.layers only) ...")
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
        print("\n" + "=" * 80)
        print("Stage 15 (InternVL-U): CALM — CKA-Guided Adaptive Layer-Wise Modularization")
        print("=" * 80)
        print(f"  Model: {self.model_path}")
        print(f"  Run date: {self.run_date}")
        print(f"  Decoder layers: {self.num_decoder_layers}")
        print(f"  Target layers: {self.target_decoder_layers}")
        print(f"  Algorithm pool: {list(self.algorithm_pool.keys())}")
        print(f"  Calibration: {self.num_und_samples} und + {self.num_gen_samples} gen")
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
        if config.get("use_gptq", False) and self.activation_data is None:
            config["use_gptq"] = False
        if config.get("use_awq", False) and self.activation_data is None:
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
        if self.activation_data and layer_name in self.activation_data:
            act_data = self.activation_data[layer_name]
        quant_layer.prepare_weight(activation_data=act_data, layer_name=layer_name, verbose=False)
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
            linear = nn.Linear(module.in_features, module.out_features,
                               bias=module.bias is not None, device=device, dtype=dtype)
            if layer_name in self.original_weights:
                linear.weight.data.copy_(self.original_weights[layer_name]["weight"].to(device))
                if linear.bias is not None and self.original_weights[layer_name]["bias"] is not None:
                    linear.bias.data.copy_(self.original_weights[layer_name]["bias"].to(device))
            self._replace_module(layer_name, linear)
        elif isinstance(module, nn.Linear) and layer_name in self.original_weights:
            module.weight.data.copy_(self.original_weights[layer_name]["weight"].to(module.weight.device))
            if module.bias is not None and self.original_weights[layer_name]["bias"] is not None:
                module.bias.data.copy_(self.original_weights[layer_name]["bias"].to(module.bias.device))

    def _restore_decoder_layer(self, layer_idx: int):
        for name in self._get_decoder_layer_linear_names(layer_idx):
            self._restore_layer(name)

    def _apply_algorithm_to_decoder_layer(self, layer_idx: int, algo_config: Dict):
        for name in self._get_decoder_layer_linear_names(layer_idx):
            self._apply_algorithm_to_layer(name, algo_config)

    def _redispatch_model(self):
        try:
            from accelerate import dispatch_model
            if hasattr(self.model, "hf_device_map"):
                self.model = dispatch_model(self.model, device_map=self.model.hf_device_map, offload_dir=None)
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

    def _collect_layer_hidden_states(self, layer_idx: int, calibration_samples: List[Dict]) -> List[torch.Tensor]:
        captured_list = []
        captured = {}

        def hook_fn(module, input, output):
            h = output[0] if isinstance(output, tuple) else output
            captured["output"] = h.detach().cpu()

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
                    if "output" in captured:
                        captured_list.append(captured["output"])
                    captured.clear()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        finally:
            handle.remove()
        return captured_list

    # ---- config helpers ----

    _FULL_CONFIG_TEMPLATE = {
        "weight_bit": 4, "act_bit": 4, "quant_percentile": 0.999999, "act_unsigned": True,
        "use_sparse": False, "sparse_ratio": 0.0, "sparse_threshold": None,
        "use_smoothquant": False, "smoothquant_alpha": 0.5,
        "use_svd": False, "svd_rank": 0,
        "use_gptq": False, "gptq_group_size": 64, "gptq_damp_percentage": 0.01,
        "gptq_block_size": 128, "use_block_quant": False, "use_block_quant_act": False,
        "block_size_weight": 256, "block_size_act": 256,
        "use_awq": False, "awq_alpha": 0.5, "awq_n_grid": 20,
    }

    def _build_full_layer_config(self, algo_key: str) -> Dict:
        cfg = dict(self._FULL_CONFIG_TEMPLATE)
        cfg.update(self.algorithm_pool[algo_key]["config"])
        if not cfg.get("use_svd", False):
            cfg["svd_rank"] = 0
        if not cfg.get("use_sparse", False):
            cfg["sparse_threshold"] = None
            cfg["sparse_ratio"] = 0.0
        return cfg

    # ---- search ----

    @staticmethod
    def _group_algos_by_wbit(pool: Dict) -> Dict[int, Dict]:
        """按 weight_bit 分组：{4: {algo_key: algo_info}, 3: {...}, 2: {...}}"""
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
        progress_file: Path,
    ) -> Dict:
        """在单一位宽下执行逐层贪心搜索，返回该位宽的搜索结果。"""
        tag = f"W{wbit}A4"
        print(f"\n{'=' * 70}")
        print(f"  Searching {tag}  ({len(algos)} algorithms)")
        print(f"{'=' * 70}")

        layer_assignments = {}
        layer_cka_scores = {}
        search_log = []
        fallback = f"rtn_w{wbit}a4"

        for layer_idx in tqdm(self.target_decoder_layers, desc=f"  {tag} greedy"):
            print(f"\n  {'─' * 55}")
            print(f"  [{tag}] Decoder Layer {layer_idx}/{self.num_decoder_layers - 1}")

            fp_hs = self._collect_layer_hidden_states(layer_idx, calibration_samples)
            if not fp_hs:
                layer_assignments[layer_idx] = fallback
                continue

            best_algo_key = None
            best_cka = -1.0
            algo_scores = {}

            for algo_key, algo_info in algos.items():
                print(f"    Trying: {algo_key} ...", end=" ")
                try:
                    self._apply_algorithm_to_decoder_layer(layer_idx, algo_info["config"])
                    self._redispatch_model()
                    quant_hs = self._collect_layer_hidden_states(layer_idx, calibration_samples)
                    cka = LinearCKA.compute_batched(
                        fp_hs, quant_hs, subsample_step=self.subsample_step
                    ) if quant_hs else 0.0
                    algo_scores[algo_key] = cka
                    print(f"CKA = {cka:.6f}")
                    if cka > best_cka:
                        best_cka = cka
                        best_algo_key = algo_key
                except Exception as e:
                    print(f"FAILED: {e}")
                    algo_scores[algo_key] = -1.0
                finally:
                    self._restore_decoder_layer(layer_idx)
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            if best_algo_key is None:
                best_algo_key = fallback
                best_cka = algo_scores.get(fallback, 0.0)

            layer_assignments[layer_idx] = best_algo_key
            layer_cka_scores[layer_idx] = algo_scores
            print(f"\n    >>> [{tag}] Best for layer {layer_idx}: "
                  f"{best_algo_key} (CKA={best_cka:.6f})")

            self._apply_algorithm_to_decoder_layer(
                layer_idx, algos[best_algo_key]["config"]
            )
            self._redispatch_model()

            search_log.append({
                "layer_idx": layer_idx,
                "best_algo": best_algo_key,
                "best_cka": best_cka,
                "all_scores": {k: round(v, 6) for k, v in algo_scores.items()},
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            })
            with open(progress_file, "w") as f:
                json.dump({
                    "weight_bit": wbit,
                    "completed_layers": len(search_log),
                    "total_layers": len(self.target_decoder_layers),
                    "search_log": search_log,
                }, f, indent=2)
            del fp_hs
            gc.collect()

        # Restore all layers to FP before next bitwidth search
        for layer_idx in self.target_decoder_layers:
            self._restore_decoder_layer(layer_idx)
        self._redispatch_model()

        return {
            "weight_bit": wbit,
            "layer_assignments": {str(k): v for k, v in layer_assignments.items()},
            "layer_cka_scores": {
                str(k): {ak: round(av, 6) for ak, av in v.items()}
                for k, v in layer_cka_scores.items()
            },
            "search_log": search_log,
        }

    def search(self) -> Dict:
        print("\n" + "=" * 80)
        print("Phase 1: Preparing calibration data (und + gen)")
        print("=" * 80)

        calib_loader = CalibrationDataLoader(
            num_und_samples=self.num_und_samples,
            num_gen_samples=self.num_gen_samples,
            mme_data_root=self.mme_data_root,
        )
        calibration_samples = calib_loader.prepare_calibration_samples()

        available_algos = {}
        for key, algo in self.algorithm_pool.items():
            if algo["config"].get("use_gptq", False) and self.activation_data is None:
                continue
            if algo["config"].get("use_awq", False) and self.activation_data is None:
                continue
            available_algos[key] = algo

        wbit_groups = self._group_algos_by_wbit(available_algos)
        print(f"\n  Algorithm groups by weight-bit:")
        for wb, grp in wbit_groups.items():
            print(f"    W{wb}A4: {len(grp)} algos — {list(grp.keys())}")

        print("\n" + "=" * 80)
        print("Phase 2: Per-bitwidth Layer-wise Greedy Search")
        print("=" * 80)

        all_bitwidth_results = {}
        config_export_dir = Path(self.output_dir).parent / "configs"
        config_export_dir.mkdir(parents=True, exist_ok=True)

        for wbit, algos in wbit_groups.items():
            tag = f"w{wbit}a4"
            progress_file = self.output_dir / f"calm_search_progress_{tag}_{self.run_date}.json"

            bw_result = self._search_single_bitwidth(
                wbit, algos, calibration_samples, progress_file,
            )
            all_bitwidth_results[wbit] = bw_result

            # Export this bitwidth's config
            export_cfg = {}
            for layer_idx_str, algo_key in bw_result["layer_assignments"].items():
                layer_idx = int(layer_idx_str)
                full_cfg = self._build_full_layer_config(algo_key)
                for name in self._get_decoder_layer_linear_names(layer_idx):
                    export_cfg[name] = dict(full_cfg)

            filepath = config_export_dir / f"calm_layerwise_{tag}_{self.run_date}.json"
            with open(filepath, "w") as f:
                json.dump(dict(sorted(export_cfg.items())), f, indent=2)
            bw_result["exported_config_path"] = str(filepath)
            print(f"\n  Exported {tag.upper()} config: {filepath}")

        # Save combined results
        results = {
            "bitwidth_results": {
                str(wb): res for wb, res in all_bitwidth_results.items()
            },
            "metadata": {
                "model_path": self.model_path, "model_type": "internvlu",
                "run_date": self.run_date,
                "num_decoder_layers": self.num_decoder_layers,
                "target_layers": self.target_decoder_layers,
                "algorithm_pool": list(self.algorithm_pool.keys()),
                "bitwidths_searched": sorted(all_bitwidth_results.keys(), reverse=True),
                "num_und_samples": self.num_und_samples,
                "num_gen_samples": self.num_gen_samples,
                "seed": self.seed,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            },
        }
        output_file = self.output_dir / f"calm_search_results_{self.run_date}.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n{'=' * 80}")
        print("Search Summary:")
        for wb, res in sorted(all_bitwidth_results.items(), reverse=True):
            assigns = res["layer_assignments"]
            algo_counts = {}
            for a in assigns.values():
                algo_counts[a] = algo_counts.get(a, 0) + 1
            top3 = sorted(algo_counts.items(), key=lambda x: -x[1])[:3]
            print(f"  W{wb}A4: {', '.join(f'{a}({c})' for a, c in top3)}")
            print(f"         config → {res.get('exported_config_path', 'N/A')}")
        print(f"{'=' * 80}\n")

        return results


def main():
    parser = argparse.ArgumentParser(description="Stage 15 (InternVL-U): CALM Search")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./quantization_outputs/stage15_calm")
    parser.add_argument("--activation_data", type=str, default=None)
    parser.add_argument("--mme_data_root", type=str, default=None)
    parser.add_argument("--num_und_samples", type=int, default=16)
    parser.add_argument("--num_gen_samples", type=int, default=16)
    parser.add_argument("--gpu_ids", type=str, default=None)
    parser.add_argument("--max_mem_per_gpu", type=str, default="40GiB")
    parser.add_argument("--target_layers", type=str, default=None)
    parser.add_argument("--algorithms", type=str, default=None)
    parser.add_argument("--subsample_step", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--run_date", type=str, default=None,
                        help="Override run date (YYYYMMDD). Default: today.")

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

    algo_pool = ALGORITHM_POOL
    if args.algorithms:
        selected = [x.strip() for x in args.algorithms.split(",")]
        algo_pool = {k: v for k, v in ALGORITHM_POOL.items() if k in selected}
        if not algo_pool:
            algo_pool = ALGORITHM_POOL

    searcher = CALMSearcher(
        model_path=args.model_path, output_dir=args.output_dir,
        algorithm_pool=algo_pool, activation_data_file=args.activation_data,
        mme_data_root=args.mme_data_root, gpu_ids=args.gpu_ids,
        max_mem_per_gpu=args.max_mem_per_gpu, target_decoder_layers=target_layers,
        seed=args.seed, subsample_step=args.subsample_step,
        num_und_samples=args.num_und_samples,
        num_gen_samples=args.num_gen_samples,
        run_date=args.run_date,
    )
    results = searcher.search()
    print("\nCALM search completed!")
    print(f"Results: {searcher.output_dir / f'calm_search_results_{searcher.run_date}.json'}")


if __name__ == "__main__":
    main()
