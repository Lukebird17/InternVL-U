#!/usr/bin/env python3
"""
通用量化配置可视化工具。

三种模式 (--mode):
  per_wbit   — 按 wbit 分图，每图内比较不同 stage（默认）
  per_stage  — 按 stage 分图，每图内比较不同 wbit（展示 bit 变化）
  proportion — 算法比例堆叠柱状图，总览所有 (stage, wbit) 组合

用法:
    python visualize_quant_configs.py \
        --config_dir ./quantization_outputs/configs \
        --output_dir ./quantization_outputs/figures \
        --wbits 4 3 2 --mode per_stage
"""

import json
import re
import argparse
from pathlib import Path
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm

# ============================================================
# Sublayer ordering (standard Qwen/Llama decoder)
# ============================================================

SUBLAYER_ORDER = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
]

SUBLAYER_SHORT = ["Q", "K", "V", "O", "Gate", "Up", "Down"]

# ============================================================
# Algorithm inference from config flags
# ============================================================

def infer_algorithm(cfg: dict) -> str:
    """Infer short algorithm name (without wXaY suffix) from config flags."""
    smooth = cfg.get("use_smoothquant", False)
    awq = cfg.get("use_awq", False)
    svd = cfg.get("use_svd", False)
    gptq = cfg.get("use_gptq", False)
    sparse = cfg.get("use_sparse", False)

    parts = []
    if smooth:
        parts.append("smooth")
    if awq:
        parts.append("awq")
    if sparse:
        parts.append("sparse")
    if svd:
        parts.append("svd")
    if gptq:
        parts.append("gptq")
    else:
        parts.append("rtn")

    return "_".join(parts) if parts else "rtn"


def extract_param_label(cfg: dict) -> str:
    """Extract key tunable parameters as a short annotation string.

    Only shows parameters that are active and non-default:
      - SmoothQuant α (when use_smoothquant=True)
      - SVD rank (when use_svd=True)
      - GPTQ group_size (when use_gptq=True)
      - AWQ n_grid (when use_awq=True)
    """
    parts = []
    if cfg.get("use_smoothquant", False):
        alpha = cfg.get("smoothquant_alpha", 0.5)
        parts.append(f"α{alpha:.2g}")
    if cfg.get("use_svd", False):
        rank = cfg.get("svd_rank", 0)
        if rank > 0:
            parts.append(f"r{rank}")
    if cfg.get("use_gptq", False):
        gs = cfg.get("gptq_group_size", 64)
        parts.append(f"g{gs}")
    if cfg.get("use_awq", False):
        ng = cfg.get("awq_n_grid", 20)
        parts.append(f"n{ng}")
    return "\n".join(parts) if parts else ""


# ============================================================
# Color palette — 10 distinct algorithms + unknown
# ============================================================

ALGO_COLORS = OrderedDict([
    # --- RTN (基础，中性灰) ---
    ("rtn",              "#b8c4cc"),
    # --- GPTQ (基础，暖麦色) ---
    ("gptq",             "#e6c36a"),
    # --- AWQ 系 (陶土/暖橙) ---
    ("awq_rtn",          "#e89a6c"),
    ("awq_svd_rtn",      "#d07a50"),
    # --- SVD/Smooth 系 + RTN (绿色谱) ---
    ("smooth_rtn",       "#8ecda5"),
    ("svd_rtn",          "#62b68a"),
    ("smooth_svd_rtn",   "#3a9970"),
    # --- SVD/Smooth 系 + GPTQ (天蓝色谱) ---
    ("smooth_gptq",      "#82b8d9"),
    ("svd_gptq",         "#5a9ec4"),
    ("smooth_svd_gptq",  "#3878a0"),
    ("_unknown",         "#e8e8e8"),
])

ALGO_LIST = list(ALGO_COLORS.keys())
ALGO_TO_IDX = {a: i for i, a in enumerate(ALGO_LIST)}

ALGO_DISPLAY = OrderedDict([
    ("rtn",              "RTN"),
    ("gptq",             "GPTQ"),
    ("awq_rtn",          "AWQ+RTN"),
    ("awq_svd_rtn",      "AWQ+SVD+RTN"),
    ("smooth_rtn",       "Smooth+RTN"),
    ("svd_rtn",          "SVD+RTN"),
    ("smooth_svd_rtn",   "Smooth+SVD+RTN"),
    ("smooth_gptq",      "Smooth+GPTQ"),
    ("svd_gptq",         "SVD+GPTQ"),
    ("smooth_svd_gptq",  "SVDQuant (Smooth+SVD+GPTQ)"),
    ("_unknown",         "Unknown"),
])


# ============================================================
# Config loading
# ============================================================

def load_config(filepath: Path) -> Dict[str, dict]:
    with open(filepath) as f:
        return json.load(f)


def parse_layer_key(key: str) -> Optional[Tuple[int, str]]:
    """Extract (layer_idx, sublayer_suffix) from full module name."""
    m = re.match(
        r"language_model\.model\.layers\.(\d+)\.(.*)", key
    )
    if not m:
        return None
    return int(m.group(1)), m.group(2)


def config_to_matrix(
    config: Dict[str, dict],
    num_layers: int,
) -> np.ndarray:
    """Convert config dict → (num_layers, 7) int matrix of algo indices."""
    mat = np.full((num_layers, len(SUBLAYER_ORDER)), ALGO_TO_IDX["_unknown"], dtype=int)
    for key, cfg in config.items():
        parsed = parse_layer_key(key)
        if parsed is None:
            continue
        layer_idx, suffix = parsed
        if suffix not in SUBLAYER_ORDER:
            continue
        col = SUBLAYER_ORDER.index(suffix)
        algo = infer_algorithm(cfg)
        mat[layer_idx, col] = ALGO_TO_IDX.get(algo, ALGO_TO_IDX["_unknown"])
    return mat


def config_to_param_matrix(
    config: Dict[str, dict],
    num_layers: int,
) -> List[List[str]]:
    """Convert config dict → (num_layers, 7) string matrix of parameter labels."""
    mat = [[""] * len(SUBLAYER_ORDER) for _ in range(num_layers)]
    for key, cfg in config.items():
        parsed = parse_layer_key(key)
        if parsed is None:
            continue
        layer_idx, suffix = parsed
        if suffix not in SUBLAYER_ORDER:
            continue
        col = SUBLAYER_ORDER.index(suffix)
        mat[layer_idx][col] = extract_param_label(cfg)
    return mat


# ============================================================
# Plotting
# ============================================================

def plot_single_wbit(
    stage_matrices: List[Tuple[str, np.ndarray]],
    wbit: int,
    output_path: Path,
    dpi: int = 150,
    param_matrices: Optional[List[Tuple[str, List[List[str]]]]] = None,
):
    """
    Plot one figure for a given wbit (horizontal layout).
    stage_matrices: list of (stage_label, matrix[num_layers, 7])
    param_matrices: optional list of (stage_label, param_str_matrix) for annotations.

    Layout: stages stacked vertically, X = decoder layer, Y = sublayer.
    """
    n_stages = len(stage_matrices)
    num_layers = stage_matrices[0][1].shape[0]
    n_sublayers = len(SUBLAYER_SHORT)
    show_params = param_matrices is not None

    cmap = ListedColormap([ALGO_COLORS[a] for a in ALGO_LIST])
    bounds = np.arange(len(ALGO_LIST) + 1) - 0.5
    norm = BoundaryNorm(bounds, cmap.N)

    cell_w = 0.55 if show_params else 0.42
    cell_h = 0.50 if show_params else 0.38
    fig_width = max(num_layers * cell_w + 2.5, 12)
    fig_height = max(n_sublayers * cell_h * n_stages + 2.0, 4)
    fig, axes = plt.subplots(
        n_stages, 1, figsize=(fig_width, fig_height),
        sharex=True, constrained_layout=True,
    )
    if n_stages == 1:
        axes = [axes]

    param_lookup = {}
    if show_params:
        for label, pmat in param_matrices:
            param_lookup[label] = pmat

    for ax, (label, mat) in zip(axes, stage_matrices):
        mat_t = mat.T
        im = ax.imshow(
            mat_t, aspect="auto", cmap=cmap, norm=norm,
            interpolation="nearest",
        )
        ax.set_ylabel(label, fontsize=10, fontweight="bold")
        ax.set_yticks(range(n_sublayers))
        ax.set_yticklabels(SUBLAYER_SHORT, fontsize=8)

        if show_params and label in param_lookup:
            pmat = param_lookup[label]
            for li in range(num_layers):
                for si in range(n_sublayers):
                    txt = pmat[li][si] if li < len(pmat) and si < len(pmat[li]) else ""
                    if txt:
                        ax.text(
                            li, si, txt, ha="center", va="center",
                            fontsize=4.5, color="#222222",
                            fontfamily="monospace", linespacing=0.9,
                        )

        for layer_idx in range(num_layers):
            if layer_idx % 4 == 0:
                ax.axvline(x=layer_idx - 0.5, color="white", linewidth=0.3)

    axes[-1].set_xticks(range(num_layers))
    axes[-1].set_xticklabels([str(i) for i in range(num_layers)], fontsize=7)
    axes[-1].set_xlabel("Decoder Layer", fontsize=10)

    used_algos = set()
    for _, mat in stage_matrices:
        used_algos.update(mat.flatten().tolist())

    fig.legend(
        handles=_build_legend_handles(used_algos),
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        fontsize=8,
        frameon=True,
        title="Algorithm",
        title_fontsize=9,
    )

    title_suffix = " (with params)" if show_params else ""
    fig.suptitle(
        f"Quantization Configuration — W{wbit}A4{title_suffix}",
        fontsize=14, fontweight="bold",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def _build_legend_handles(used_algos):
    """Build grouped legend handles for the given set of used algo indices."""
    ALGO_GROUPS = [
        ("RTN",          ["rtn"]),
        ("GPTQ",         ["gptq"]),
        ("AWQ",          ["awq_rtn", "awq_svd_rtn"]),
        ("SVD/Smooth",   ["smooth_rtn", "svd_rtn", "smooth_svd_rtn",
                          "smooth_gptq", "svd_gptq", "smooth_svd_gptq"]),
    ]
    legend_handles = []
    for group_label, members in ALGO_GROUPS:
        group_present = [a for a in members if ALGO_TO_IDX.get(a, -1) in used_algos]
        if not group_present:
            continue
        legend_handles.append(
            mpatches.Patch(color="none", label=f"── {group_label} ──")
        )
        for algo in group_present:
            legend_handles.append(
                mpatches.Patch(
                    color=ALGO_COLORS[algo],
                    label=f"  {ALGO_DISPLAY.get(algo, algo)}",
                )
            )
    return legend_handles


def plot_per_stage(
    all_data: Dict[int, List[Tuple[str, np.ndarray]]],
    stage_labels: List[str],
    wbits: List[int],
    output_dir: Path,
    dpi: int = 150,
    all_param_data: Optional[Dict[int, List[Tuple[str, List[List[str]]]]]] = None,
):
    """
    Per-stage figures: each stage gets one figure with wbits as rows.
    Directly shows how the same layer's algorithm changes with bit-width.
    """
    cmap = ListedColormap([ALGO_COLORS[a] for a in ALGO_LIST])
    bounds = np.arange(len(ALGO_LIST) + 1) - 0.5
    norm = BoundaryNorm(bounds, cmap.N)
    n_sublayers = len(SUBLAYER_SHORT)
    show_params = all_param_data is not None

    for stage_label in stage_labels:
        rows = []
        row_params = []
        for wbit in wbits:
            for label, mat in all_data.get(wbit, []):
                if label == stage_label:
                    rows.append((f"W{wbit}A4", mat))
                    if show_params and all_param_data.get(wbit):
                        for plabel, pmat in all_param_data[wbit]:
                            if plabel == stage_label:
                                row_params.append(pmat)
                                break
                        else:
                            row_params.append(None)
                    break

        if not rows:
            continue

        n_rows = len(rows)
        num_layers = rows[0][1].shape[0]

        cell_w = 0.55 if show_params else 0.42
        cell_h = 0.50 if show_params else 0.38
        fig_width = max(num_layers * cell_w + 2.5, 12)
        fig_height = max(n_sublayers * cell_h * n_rows + 2.0, 4)
        fig, axes = plt.subplots(
            n_rows, 1, figsize=(fig_width, fig_height),
            sharex=True, constrained_layout=True,
        )
        if n_rows == 1:
            axes = [axes]

        for ri, (ax, (row_label, mat)) in enumerate(zip(axes, rows)):
            im = ax.imshow(
                mat.T, aspect="auto", cmap=cmap, norm=norm,
                interpolation="nearest",
            )
            ax.set_ylabel(row_label, fontsize=10, fontweight="bold")
            ax.set_yticks(range(n_sublayers))
            ax.set_yticklabels(SUBLAYER_SHORT, fontsize=8)

            if show_params and ri < len(row_params) and row_params[ri] is not None:
                pmat = row_params[ri]
                for li in range(num_layers):
                    for si in range(n_sublayers):
                        txt = pmat[li][si] if li < len(pmat) and si < len(pmat[li]) else ""
                        if txt:
                            ax.text(
                                li, si, txt, ha="center", va="center",
                                fontsize=4.5, color="#222222",
                                fontfamily="monospace", linespacing=0.9,
                            )

            for li in range(num_layers):
                if li % 4 == 0:
                    ax.axvline(x=li - 0.5, color="white", linewidth=0.3)

        axes[-1].set_xticks(range(num_layers))
        axes[-1].set_xticklabels([str(i) for i in range(num_layers)], fontsize=7)
        axes[-1].set_xlabel("Decoder Layer", fontsize=10)

        used = set()
        for _, mat in rows:
            used.update(mat.flatten().tolist())

        fig.legend(
            handles=_build_legend_handles(used),
            loc="center left", bbox_to_anchor=(1.01, 0.5),
            fontsize=8, frameon=True, title="Algorithm", title_fontsize=9,
        )
        title_suffix = " (with params)" if show_params else ""
        fig.suptitle(
            f"{stage_label} — Config across bit-widths{title_suffix}",
            fontsize=14, fontweight="bold",
        )

        safe_name = stage_label.lower().replace(" ", "_")
        suffix = "_params" if show_params else ""
        out_path = output_dir / f"cross_wbit_{safe_name}{suffix}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out_path), dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_path}")


def plot_proportion(
    all_data: Dict[int, List[Tuple[str, np.ndarray]]],
    stage_labels: List[str],
    wbits: List[int],
    output_dir: Path,
    dpi: int = 150,
):
    """
    Stacked bar chart: algorithm proportion for each (stage, wbit).
    """
    display_algos = [a for a in ALGO_LIST if a != "_unknown"]
    n_algos = len(display_algos)

    groups = []
    proportions = {a: [] for a in display_algos}

    for stage_label in stage_labels:
        for wbit in wbits:
            mat = None
            for label, m in all_data.get(wbit, []):
                if label == stage_label:
                    mat = m
                    break
            if mat is None:
                continue
            total = mat.size
            short_label = stage_label.split()[-1] if " " in stage_label else stage_label
            groups.append(f"{short_label}\nW{wbit}A4")
            for algo in display_algos:
                idx = ALGO_TO_IDX[algo]
                proportions[algo].append(np.sum(mat == idx) / total * 100)

    if not groups:
        return

    n_groups = len(groups)
    x = np.arange(n_groups)
    bar_width = 0.65

    fig, ax = plt.subplots(figsize=(max(n_groups * 1.5, 8), 5.5))

    bottom = np.zeros(n_groups)
    for algo in display_algos:
        vals = np.array(proportions[algo])
        if vals.sum() == 0:
            continue
        ax.bar(
            x, vals, bar_width, bottom=bottom,
            color=ALGO_COLORS[algo],
            label=ALGO_DISPLAY.get(algo, algo),
            edgecolor="white", linewidth=0.3,
        )
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=9, ha="center")
    ax.set_ylabel("Proportion (%)", fontsize=10)
    ax.set_ylim(0, 105)
    ax.set_xlim(-0.5, n_groups - 0.5)
    ax.set_title("Algorithm Distribution across Stages & Bit-widths",
                 fontsize=13, fontweight="bold")

    for i in range(1, len(stage_labels)):
        boundary = i * len(wbits) - 0.5
        ax.axvline(x=boundary, color="#999999", linewidth=1, linestyle="--")

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles[::-1], labels[::-1],
        loc="center left", bbox_to_anchor=(1.01, 0.5),
        fontsize=8, frameon=True, title="Algorithm", title_fontsize=9,
    )

    fig.tight_layout()
    out_path = output_dir / "algo_proportion.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ============================================================
# Main
# ============================================================

DEFAULT_STAGE_SPECS = [
    ("Stage15 CALM",       "calm_layerwise_w{wbit}a4.json"),
    ("Stage17 Exhaustive", "exhaustive_sublayer_w{wbit}a4.json"),
    ("Stage18 FuncGroup",  "funcgroup_w{wbit}a4.json"),
]


def parse_stage_spec(spec: str) -> Tuple[str, str]:
    """Parse 'Label:filename_pattern' string."""
    if ":" in spec:
        label, pattern = spec.split(":", 1)
        return label.strip(), pattern.strip()
    return Path(spec).stem, spec


def _load_all_data(config_dir, stage_specs, wbits, with_params=False):
    """Load all configs and return {wbit: [(label, matrix), ...]} and global num_layers.

    If with_params=True, also returns {wbit: [(label, param_matrix), ...]}.
    """
    all_data = {}
    all_param_data = {} if with_params else None
    global_num_layers = 0

    for wbit in wbits:
        print(f"\n{'='*60}")
        print(f"  W{wbit}A4")
        print(f"{'='*60}")

        stage_matrices = []
        param_matrices = [] if with_params else None
        for label, pattern in stage_specs:
            filename = pattern.format(wbit=wbit)
            filepath = config_dir / filename
            if not filepath.exists():
                print(f"  [SKIP] {label}: {filepath} not found")
                continue
            cfg = load_config(filepath)
            layer_indices = set()
            for key in cfg:
                parsed = parse_layer_key(key)
                if parsed:
                    layer_indices.add(parsed[0])
            n = max(layer_indices) + 1 if layer_indices else 0
            global_num_layers = max(global_num_layers, n)
            mat = config_to_matrix(cfg, n)
            stage_matrices.append((label, mat))

            if with_params:
                pmat = config_to_param_matrix(cfg, n)
                param_matrices.append((label, pmat))

            algos_used = set()
            for key, c in cfg.items():
                algos_used.add(infer_algorithm(c))
            print(f"  {label}: {len(cfg)} sublayers, algos = {sorted(algos_used)}")

        all_data[wbit] = stage_matrices
        if with_params:
            all_param_data[wbit] = param_matrices

    return all_data, all_param_data


def main():
    parser = argparse.ArgumentParser(
        description="Visualize quantization configs across stages (sublayer granularity)"
    )
    parser.add_argument(
        "--config_dir", type=str, required=True,
        help="Directory containing exported config JSONs",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory for figures (default: config_dir/../figures)",
    )
    parser.add_argument(
        "--wbits", type=int, nargs="+", default=[4, 3, 2],
        help="Weight bit-widths to visualize",
    )
    parser.add_argument(
        "--configs", type=str, nargs="+", default=None,
        help='Stage specs as "Label:pattern_with_{wbit}" (default: 3 built-in stages)',
    )
    parser.add_argument(
        "--mode", type=str, default="all",
        choices=["per_wbit", "per_stage", "proportion", "all"],
        help="per_wbit: group by wbit; per_stage: group by stage (cross-wbit); "
             "proportion: stacked bar chart; all: generate everything",
    )
    parser.add_argument(
        "--show_params", action="store_true",
        help="Annotate cells with key parameters (α, rank, group_size, n_grid)",
    )
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    config_dir = Path(args.config_dir)
    output_dir = Path(args.output_dir) if args.output_dir else config_dir.parent / "figures"

    if args.configs:
        stage_specs = [parse_stage_spec(s) for s in args.configs]
    else:
        stage_specs = DEFAULT_STAGE_SPECS

    all_data, all_param_data = _load_all_data(
        config_dir, stage_specs, args.wbits, with_params=args.show_params,
    )
    stage_labels = [label for label, _ in stage_specs]
    mode = args.mode

    if mode in ("per_wbit", "all"):
        print("\n>>> per_wbit figures")
        for wbit in args.wbits:
            if all_data.get(wbit):
                suffix = "_params" if args.show_params else ""
                out_path = output_dir / f"quant_config_w{wbit}a4{suffix}.png"
                pmat_list = all_param_data.get(wbit) if all_param_data else None
                plot_single_wbit(
                    all_data[wbit], wbit, out_path, dpi=args.dpi,
                    param_matrices=pmat_list,
                )

    if mode in ("per_stage", "all"):
        print("\n>>> per_stage figures (cross bit-width)")
        plot_per_stage(
            all_data, stage_labels, args.wbits, output_dir, dpi=args.dpi,
            all_param_data=all_param_data,
        )

    if mode in ("proportion", "all"):
        print("\n>>> proportion figure")
        plot_proportion(all_data, stage_labels, args.wbits, output_dir, dpi=args.dpi)

    print(f"\nAll figures saved to: {output_dir}")


if __name__ == "__main__":
    main()
