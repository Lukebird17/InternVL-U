"""
InternVL-U 量化实验可视化工具

为 Stage 22/23/24/25 的 JSON 结果生成论文级别的图表。
支持自动 Y 轴缩放、Stage 21 benchmark 对比。

用法:
  python utils/visualize_results.py --stage 22 --results_dir quantization_outputs/stage22_modality_analysis
  python utils/visualize_results.py --stage 23 --results_dir quantization_outputs/stage23_modality_weighted
  python utils/visualize_results.py --stage 24 --results_dir quantization_outputs/stage24_attention_fidelity
  python utils/visualize_results.py --stage 25 --results_dir quantization_outputs/stage25_hard_sample
  python utils/visualize_results.py --stage all --root quantization_outputs
"""

import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
})

COLORS = {
    "vision": "#E74C3C",
    "text": "#3498DB",
    "full": "#2ECC71",
    "int8": "#F39C12",
    "int4": "#9B59B6",
    "attn": "#E67E22",
    "mlp": "#1ABC9C",
    "hard": "#C0392B",
    "random": "#7F8C8D",
    "easy": "#27AE60",
    "s21": "#95A5A6",
}


def _save(fig, out_dir: Path, name: str):
    path = out_dir / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def _find_json(directory: Path, prefix: str) -> Optional[Path]:
    candidates = sorted(directory.glob(f"{prefix}*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _auto_ylim(values: List[float], margin_ratio: float = 0.15,
               min_range: float = 0.005) -> Tuple[float, float]:
    """根据数据自动计算 Y 轴范围，聚焦有变化的区间。"""
    vals = [v for v in values if v is not None and np.isfinite(v) and v != 0.0]
    if not vals:
        return (0.0, 1.05)
    vmin, vmax = min(vals), max(vals)
    data_range = max(vmax - vmin, min_range)
    margin = data_range * margin_ratio
    lo = max(0.0, vmin - margin)
    hi = min(1.5, vmax + margin)
    if hi - lo < min_range * 2:
        mid = (vmin + vmax) / 2
        lo = max(0.0, mid - min_range)
        hi = min(1.5, mid + min_range)
    return (lo, hi)


def _algo_signature(cfg: Dict) -> str:
    """从 config dict 提取算法摘要标签。"""
    parts = []
    if cfg.get("use_gptq"):
        parts.append("GPTQ")
    if cfg.get("use_smoothquant"):
        parts.append(f"SQ{cfg.get('smoothquant_alpha', 0.5)}")
    if cfg.get("use_svd"):
        parts.append(f"SVD{cfg.get('svd_rank', 0)}")
    if cfg.get("use_awq"):
        parts.append("AWQ")
    if not parts:
        parts.append("RTN")
    return "+".join(parts)


def _load_s21_config(root: Path) -> Optional[Dict]:
    """尝试加载 Stage 21 的 config 作为 benchmark。"""
    config_dir = root / "configs"
    if not config_dir.exists():
        return None
    candidates = sorted(config_dir.glob("stage21_funcgroup_w4a4_*.json"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        return None
    try:
        with open(candidates[0]) as f:
            return json.load(f)
    except Exception:
        return None


def _normalize_funcgroup_search_log(log: Optional[List]) -> List[Dict]:
    """统一 functional-group search_log 字段名。

    Stage 21 使用 decoder_layer_idx / group_name；
    Stage 23/24/25 使用 layer_idx / group。
    """
    if not log:
        return []
    out: List[Dict] = []
    for e in log:
        if not isinstance(e, dict) or "best_algo" not in e:
            continue
        li = e.get("layer_idx")
        if li is None:
            li = e.get("decoder_layer_idx")
        g = e.get("group")
        if g is None:
            g = e.get("group_name")
        if li is None or g is None:
            continue
        ne = dict(e)
        ne["layer_idx"] = int(li)
        ne["group"] = str(g)
        out.append(ne)
    return out


def _load_s21_search_log(root: Path) -> Optional[List]:
    """尝试加载 Stage 21 的 search_log，并规范为 layer_idx / group。"""
    s21_dir = root / "stage21_funcgroup"
    if not s21_dir.exists():
        s21_dir = root / "stage21"
    if not s21_dir.exists():
        return None
    jf = _find_json(s21_dir, "stage21_search_results")
    if jf is None:
        return None
    try:
        with open(jf) as f:
            data = json.load(f)
        bw = data.get("bitwidth_results", {}).get("4", {})
        raw = bw.get("search_log")
        if not raw:
            return None
        norm = _normalize_funcgroup_search_log(raw)
        return norm if norm else None
    except Exception:
        return None


def _draw_algo_heatmap_with_benchmark(
    fig, axes, search_log, title, s21_log=None, out_dir=None, filename="algo_selection"
):
    """绘制算法选择热力图，可选 S21 benchmark 行。"""
    search_log = _normalize_funcgroup_search_log(search_log or [])
    if s21_log is not None:
        s21_log = _normalize_funcgroup_search_log(s21_log)
        if not s21_log:
            s21_log = None
    all_entries_algos = [e["best_algo"] for e in search_log]
    if s21_log:
        all_entries_algos += [e["best_algo"] for e in s21_log]

    all_algos = sorted(set(all_entries_algos))
    algo_to_idx = {a: i for i, a in enumerate(all_algos)}
    cmap = matplotlib.colormaps.get_cmap("Set3").resampled(max(len(all_algos), 3))

    attn_entries = [e for e in search_log if e["group"] == "attn"]
    mlp_entries = [e for e in search_log if e["group"] == "mlp"]

    for ax, entries, gname in [(axes[0], attn_entries, "Attn"), (axes[1], mlp_entries, "MLP")]:
        if not entries:
            continue
        layer_ids = [e["layer_idx"] for e in entries]
        colors = [cmap(algo_to_idx[e["best_algo"]]) for e in entries]
        ax.bar(range(len(entries)), [1]*len(entries), color=colors, edgecolor="white", linewidth=0.5)
        ax.set_ylabel(gname)
        ax.set_yticks([])
        step = max(1, len(entries) // 20)
        ax.set_xticks(range(0, len(entries), step))
        ax.set_xticklabels([str(layer_ids[i]) for i in range(0, len(entries), step)])

    if s21_log and len(axes) > 2:
        s21_attn = [e for e in s21_log if e["group"] == "attn"]
        s21_mlp = [e for e in s21_log if e["group"] == "mlp"]
        for ax, s21_entries, gname in [(axes[2], s21_attn, "S21 Attn"), (axes[3] if len(axes) > 3 else None, s21_mlp, "S21 MLP")]:
            if ax is None or not s21_entries:
                continue
            colors = [cmap(algo_to_idx.get(e["best_algo"], 0)) for e in s21_entries]
            ax.bar(range(len(s21_entries)), [1]*len(s21_entries), color=colors,
                   edgecolor="white", linewidth=0.5, alpha=0.6)
            ax.set_ylabel(gname, fontsize=9)
            ax.set_yticks([])

    legend_handles = [Patch(facecolor=cmap(algo_to_idx[a]), label=a) for a in all_algos]
    fig.legend(handles=legend_handles, loc="upper right", ncol=min(3, len(all_algos)),
               fontsize=8, title="Algorithm")
    axes[-1].set_xlabel("Decoder Layer")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    if out_dir:
        _save(fig, out_dir, filename)


# ═══════════════════════════════════════════════════════════════════
#  Stage 22: Modality Outlier Analysis
# ═══════════════════════════════════════════════════════════════════

def visualize_stage22(results_dir: Path):
    out_dir = results_dir / "figures"
    out_dir.mkdir(exist_ok=True)
    jf = _find_json(results_dir, "stage22_modality_analysis")
    if jf is None:
        print("  [Stage 22] No results JSON found, skipping.")
        return
    with open(jf) as f:
        data = json.load(f)

    layer_stats = data.get("layer_statistics", {})
    cka_sens = data.get("cka_sensitivity", {})
    summary = data.get("summary", {})

    layers = sorted(layer_stats.keys(), key=int)
    layer_ints = [int(l) for l in layers]

    # ── Fig 1: Vision vs Text abs_max & std per layer ──
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    v_max = [layer_stats[l]["vision"]["abs_max"] for l in layers]
    t_max = [layer_stats[l]["text"]["abs_max"] for l in layers]
    v_std = [layer_stats[l]["vision"]["std"] for l in layers]
    t_std = [layer_stats[l]["text"]["std"] for l in layers]

    w = 0.35
    x = np.arange(len(layers))
    axes[0].bar(x - w/2, v_max, w, color=COLORS["vision"], label="Vision", alpha=0.85)
    axes[0].bar(x + w/2, t_max, w, color=COLORS["text"], label="Text", alpha=0.85)
    axes[0].set_ylabel("Absolute Maximum")
    axes[0].set_title("Per-Layer Activation Absolute Maximum: Vision vs Text")
    axes[0].legend()

    axes[1].bar(x - w/2, v_std, w, color=COLORS["vision"], label="Vision", alpha=0.85)
    axes[1].bar(x + w/2, t_std, w, color=COLORS["text"], label="Text", alpha=0.85)
    axes[1].set_ylabel("Standard Deviation")
    axes[1].set_xlabel("Decoder Layer")
    axes[1].set_title("Per-Layer Activation Std: Vision vs Text")
    axes[1].set_xticks(x[::max(1, len(x)//20)])
    axes[1].set_xticklabels([layers[i] for i in range(0, len(layers), max(1, len(layers)//20))])
    axes[1].legend()

    fig.tight_layout()
    _save(fig, out_dir, "s22_abs_max_std")

    # ── Fig 2: Outlier ratio comparison ──
    fig, ax = plt.subplots(figsize=(14, 5))

    v_outlier = [layer_stats[l]["vision"]["outlier_ratio_3sigma"] for l in layers]
    t_outlier = [layer_stats[l]["text"]["outlier_ratio_3sigma"] for l in layers]

    ax.plot(layer_ints, v_outlier, "o-", color=COLORS["vision"], label="Vision 3σ outlier", markersize=4, linewidth=1.5)
    ax.plot(layer_ints, t_outlier, "s-", color=COLORS["text"], label="Text 3σ outlier", markersize=4, linewidth=1.5)
    ax.fill_between(layer_ints, v_outlier, t_outlier, alpha=0.15, color=COLORS["vision"],
                     where=[v > t for v, t in zip(v_outlier, t_outlier)])
    ax.set_xlabel("Decoder Layer")
    ax.set_ylabel("3σ Outlier Ratio")
    ax.set_title("Vision vs Text Outlier Ratio Across Layers (shaded = vision > text)")
    ax.legend()
    fig.tight_layout()
    _save(fig, out_dir, "s22_outlier_ratio")

    # ── Fig 3: Kurtosis comparison ──
    fig, ax = plt.subplots(figsize=(14, 5))
    v_kurt = [layer_stats[l]["vision"]["kurtosis"] for l in layers]
    t_kurt = [layer_stats[l]["text"]["kurtosis"] for l in layers]

    ax.plot(layer_ints, v_kurt, "o-", color=COLORS["vision"], label="Vision", markersize=4, linewidth=1.5)
    ax.plot(layer_ints, t_kurt, "s-", color=COLORS["text"], label="Text", markersize=4, linewidth=1.5)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Decoder Layer")
    ax.set_ylabel("Excess Kurtosis")
    ax.set_title("Distribution Kurtosis: Vision vs Text (higher = heavier tails)")
    ax.legend()
    fig.tight_layout()
    _save(fig, out_dir, "s22_kurtosis")

    # ── Fig 4: Sensitivity ratio heatmap ──
    if layers:
        metrics = ["abs_max_ratio", "std_ratio", "outlier_3sigma_ratio"]
        metric_labels = ["abs_max V/T", "std V/T", "outlier_3σ V/T"]
        matrix = np.zeros((len(metrics), len(layers)))
        for j, l in enumerate(layers):
            sr = layer_stats[l].get("sensitivity_ratio", {})
            for i, m in enumerate(metrics):
                matrix[i, j] = sr.get(m, 1.0)

        fig, ax = plt.subplots(figsize=(max(14, len(layers)*0.4), 3))
        cmap = LinearSegmentedColormap.from_list("vt", ["#3498DB", "#FFFFFF", "#E74C3C"])
        im = ax.imshow(matrix, aspect="auto", cmap=cmap,
                       vmin=max(0, matrix.min()), vmax=min(matrix.max(), 5))
        ax.set_yticks(range(len(metrics)))
        ax.set_yticklabels(metric_labels)
        step = max(1, len(layers) // 20)
        ax.set_xticks(range(0, len(layers), step))
        ax.set_xticklabels([layers[i] for i in range(0, len(layers), step)])
        ax.set_xlabel("Decoder Layer")
        ax.set_title("Vision / Text Ratio Heatmap (red = vision more extreme)")
        plt.colorbar(im, ax=ax, shrink=0.8, label="Ratio")
        fig.tight_layout()
        _save(fig, out_dir, "s22_sensitivity_heatmap")

    # ── Fig 5: CKA sensitivity per quantization method (AUTO-SCALED) ──
    if cka_sens:
        cka_layers = sorted(cka_sens.keys(), key=int)
        cka_li = [int(l) for l in cka_layers]

        first_layer_methods = cka_sens[cka_layers[0]]
        methods = sorted(first_layer_methods.keys())
        if not methods:
            methods = ["int4"]

        n_methods = len(methods)
        n_cols = min(3, n_methods)
        n_rows = (n_methods + n_cols - 1) // n_cols
        fig, axes_arr = plt.subplots(n_rows, n_cols,
                                      figsize=(6 * n_cols, 4.5 * n_rows),
                                      squeeze=False)
        axes_flat = axes_arr.flatten()

        method_colors = plt.cm.tab10(np.linspace(0, 1, 10))

        for idx, method in enumerate(methods):
            ax = axes_flat[idx]
            cv, ct, ca = [], [], []
            for l in cka_layers:
                m_data = cka_sens[l].get(method, {})
                cv.append(m_data.get("cka_vision", 0))
                ct.append(m_data.get("cka_text", 0))
                ca.append(m_data.get("cka_all", 0))

            ax.plot(cka_li, cv, "o-", color=COLORS["vision"], label="CKA_vision", markersize=4, linewidth=1.5)
            ax.plot(cka_li, ct, "s-", color=COLORS["text"], label="CKA_text", markersize=4, linewidth=1.5)
            ax.plot(cka_li, ca, "^--", color=COLORS["full"], label="CKA_all", markersize=4, alpha=0.7, linewidth=1.2)
            ax.set_xlabel("Decoder Layer")
            ax.set_ylabel("CKA Score")
            ax.set_title(method, fontsize=11)

            all_vals = cv + ct + ca
            ylim = _auto_ylim(all_vals)
            ax.set_ylim(*ylim)
            ax.legend(fontsize=7)

        for idx in range(n_methods, len(axes_flat)):
            axes_flat[idx].axis("off")

        fig.suptitle("CKA Sensitivity: Vision vs Text per Quantization Method", fontsize=14, y=1.02)
        fig.tight_layout()
        _save(fig, out_dir, "s22_cka_sensitivity")

        # ── Fig 5b: Cross-method CKA comparison (layer-averaged) ──
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for ax, metric, label in [
            (axes[0], "cka_vision", "CKA Vision"),
            (axes[1], "cka_text", "CKA Text"),
            (axes[2], "cka_all", "CKA All"),
        ]:
            for m_idx, method in enumerate(methods):
                vals = []
                for l in cka_layers:
                    vals.append(cka_sens[l].get(method, {}).get(metric, 0))
                ax.plot(cka_li, vals, "o-", label=method, markersize=3, linewidth=1.2, alpha=0.8)
            ax.set_xlabel("Decoder Layer")
            ax.set_ylabel(label)
            ax.set_title(label)
            all_cross = []
            for method in methods:
                for l in cka_layers:
                    all_cross.append(cka_sens[l].get(method, {}).get(metric, 0))
            ylim = _auto_ylim(all_cross)
            ax.set_ylim(*ylim)
            ax.legend(fontsize=6, ncol=2)

        fig.suptitle("Cross-Method CKA Comparison (per layer)", fontsize=14, y=1.02)
        fig.tight_layout()
        _save(fig, out_dir, "s22_cka_cross_method")

    # ── Summary text figure ──
    if summary:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.axis("off")
        lines = [
            f"Vision avg abs_max:       {summary.get('avg_vision_abs_max', 'N/A')}",
            f"Text avg abs_max:         {summary.get('avg_text_abs_max', 'N/A')}",
            f"Vision/Text max ratio:    {summary.get('vision_to_text_max_ratio', 'N/A')}x",
            f"Vision avg outlier(3σ):   {summary.get('avg_vision_outlier_3sigma', 'N/A')}",
            f"Text avg outlier(3σ):     {summary.get('avg_text_outlier_3sigma', 'N/A')}",
            f"Vision/Text outlier ratio:{summary.get('vision_to_text_outlier_ratio', 'N/A')}x",
        ]
        cka_gap = summary.get("avg_cka_gap_vision_vs_text")
        if cka_gap is None:
            cka_gap = summary.get("avg_int4_cka_gap_vision_vs_text", "N/A")
        lines.append(f"Avg CKA gap (T-V):        {cka_gap}")

        per_method = summary.get("per_method_cka_gap", {})
        if per_method:
            lines.append("")
            lines.append("Per-Method CKA gap (T-V):")
            for method, gap in sorted(per_method.items()):
                lines.append(f"  {method:30s} {gap:.4f}" if isinstance(gap, (int, float)) else f"  {method:30s} {gap}")

        lines.append(f"")
        lines.append(f"Hypothesis:               {summary.get('conclusion', 'N/A')}")
        ax.text(0.02, 0.98, "\n".join(lines), transform=ax.transAxes,
                fontsize=10, verticalalignment="top", fontfamily="monospace",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
        ax.set_title("Stage 22 Summary", fontsize=14)
        fig.tight_layout()
        _save(fig, out_dir, "s22_summary")

    print(f"  Stage 22: {len(list(out_dir.glob('*.png')))} figures generated")


# ═══════════════════════════════════════════════════════════════════
#  Stage 23: Modality-Weighted Search
# ═══════════════════════════════════════════════════════════════════

def visualize_stage23(results_dir: Path, root: Optional[Path] = None):
    out_dir = results_dir / "figures"
    out_dir.mkdir(exist_ok=True)
    jf = _find_json(results_dir, "stage23_search_results")
    if jf is None:
        print("  [Stage 23] No results JSON found, skipping.")
        return
    with open(jf) as f:
        data = json.load(f)

    bw_results = data.get("bitwidth_results", {}).get("4", {})
    search_log = bw_results.get("search_log", [])
    assignments = bw_results.get("group_assignments", {})
    meta = data.get("metadata", {})
    wv = meta.get("vision_weight", 0.7)
    wt = meta.get("text_weight", 0.3)

    if not search_log:
        print("  [Stage 23] No search log found.")
        return

    s21_log = _load_s21_search_log(root) if root else None
    if s21_log:
        print("  [Stage 23] Loaded Stage 21 benchmark for comparison")
    else:
        print("  [Stage 23] No Stage 21 benchmark found (run Stage 21 first to enable)")

    # ── Fig 1: Algorithm selection heatmap (with S21 benchmark) ──
    n_rows = 4 if s21_log else 2
    fig, axes = plt.subplots(n_rows, 1,
                              figsize=(max(14, len(search_log)*0.18), 2.5 * n_rows),
                              sharex=True)
    _draw_algo_heatmap_with_benchmark(
        fig, axes, search_log,
        f"Stage 23: Algorithm Selection per Layer (w_v={wv}, w_t={wt})",
        s21_log=s21_log, out_dir=out_dir, filename="s23_algo_selection"
    )

    # ── Fig 2: Per-layer CKA breakdown (AUTO-SCALED) ──
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    layers_seen = []
    combined_scores = []
    v_scores = []
    t_scores = []
    a_scores = []

    for entry in search_log:
        best = entry["best_algo"]
        detail = entry.get("all_scores", {}).get(best, {})
        layers_seen.append(f"L{entry['layer_idx']}_{entry['group'][:1]}")
        combined_scores.append(detail.get("combined", 0))
        v_scores.append(detail.get("cka_vision", 0))
        t_scores.append(detail.get("cka_text", 0))
        a_scores.append(detail.get("cka_all", 0))

    x = np.arange(len(layers_seen))

    ax_top = axes[0]
    ax_top.plot(x, combined_scores, "k-", label="Combined", linewidth=2, alpha=0.8)
    ax_top.plot(x, a_scores, "^--", color=COLORS["full"], label="CKA_all", linewidth=1.2, alpha=0.7, markersize=3)
    ax_top.plot(x, t_scores, "s--", color=COLORS["text"], label="CKA_text", linewidth=1.2, alpha=0.7, markersize=3)
    ylim_top = _auto_ylim(combined_scores + a_scores + t_scores)
    ax_top.set_ylim(*ylim_top)
    ax_top.set_ylabel("CKA Score")
    ax_top.set_title("Per-Group Best Algorithm CKA Breakdown (Combined / All / Text)")
    ax_top.legend()

    ax_bot = axes[1]
    nonzero_v = [v for v in v_scores if v > 0]
    if nonzero_v:
        ax_bot.plot(x, v_scores, "o-", color=COLORS["vision"], label="CKA_vision",
                    linewidth=1.5, markersize=3)
        ylim_v = _auto_ylim(v_scores, margin_ratio=0.2)
        ax_bot.set_ylim(*ylim_v)
        ax_bot.set_title("CKA_vision per Group (separate scale)")
    else:
        ax_bot.text(0.5, 0.5, "CKA_vision = 0.0 for all entries\n"
                    "(calibration images may not be loaded — check image_path in calibration JSON)",
                    ha="center", va="center", transform=ax_bot.transAxes,
                    fontsize=12, color="red", fontweight="bold",
                    bbox=dict(facecolor="lightyellow", alpha=0.8))
        ax_bot.set_title("CKA_vision (NO DATA)")
    ax_bot.set_ylabel("CKA Score")
    if nonzero_v:
        ax_bot.legend()

    step = max(1, len(x) // 30)
    ax_bot.set_xticks(x[::step])
    ax_bot.set_xticklabels([layers_seen[i] for i in range(0, len(layers_seen), step)], rotation=45, ha="right")
    ax_bot.set_xlabel("Layer_Group")

    fig.tight_layout()
    _save(fig, out_dir, "s23_cka_breakdown")

    # ── Fig 3: Algorithm distribution ──
    all_algos = sorted(set(e["best_algo"] for e in search_log))
    algo_to_idx = {a: i for i, a in enumerate(all_algos)}
    cmap = matplotlib.colormaps.get_cmap("Set3").resampled(max(len(all_algos), 3))

    algo_counts = {}
    for a in assignments.values():
        algo_counts[a] = algo_counts.get(a, 0) + 1
    sorted_algos = sorted(algo_counts.items(), key=lambda x: -x[1])

    fig, ax = plt.subplots(figsize=(8, 5))
    names = [a for a, _ in sorted_algos]
    counts = [c for _, c in sorted_algos]
    bars = ax.barh(names, counts, color=[cmap(algo_to_idx.get(n, 0)) for n in names], edgecolor="white")
    ax.set_xlabel("Number of Sublayers")
    ax.set_title(f"Stage 23: Algorithm Distribution (w_v={wv}, w_t={wt})")
    for bar, c in zip(bars, counts):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                str(c), va="center", fontsize=10)
    fig.tight_layout()
    _save(fig, out_dir, "s23_algo_distribution")

    print(f"  Stage 23: {len(list(out_dir.glob('*.png')))} figures generated")


# ═══════════════════════════════════════════════════════════════════
#  Stage 24: Attention Fidelity
# ═══════════════════════════════════════════════════════════════════

def visualize_stage24(results_dir: Path, root: Optional[Path] = None):
    out_dir = results_dir / "figures"
    out_dir.mkdir(exist_ok=True)

    # Load data from multiple possible sources, preferring newer files
    attn_analysis = {}
    search_log = []
    assignments = {}
    saliency_info = {}

    # Source 1: combined results file (stage24_results_*.json)
    jf = _find_json(results_dir, "stage24_results")
    if jf is not None:
        with open(jf) as f:
            data = json.load(f)
        attn_analysis = data.get("attention_analysis", {})
        sr = data.get("search_results", {})
        search_log = sr.get("search_log", [])
        assignments = sr.get("group_assignments", {})
        saliency_info = data.get("saliency", {})

    # Source 2: separate Phase A file (stage24_phaseA_*.json) — override if newer/non-empty
    phaseA_file = _find_json(results_dir, "stage24_phaseA")
    if phaseA_file is not None:
        with open(phaseA_file) as f:
            pa = json.load(f)
        pa_data = pa.get("attention_analysis", {})
        if len(pa_data) > len(attn_analysis):
            attn_analysis = pa_data
            print(f"  [Stage 24] Using Phase A data from {phaseA_file.name} ({len(pa_data)} entries)")

    # Source 3: incremental search progress (search_progress_*.json) — override if more complete
    progress_file = _find_json(results_dir, "search_progress")
    if progress_file is not None:
        with open(progress_file) as f:
            prog = json.load(f)
        prog_log = prog.get("search_log", [])
        if len(prog_log) > len(search_log):
            search_log = prog_log
            assignments = prog.get("group_assignments", assignments)
            print(f"  [Stage 24] Using search progress from {progress_file.name} ({len(prog_log)} entries)")

    if not attn_analysis and not search_log:
        print("  [Stage 24] No results found in any data file, skipping.")
        return

    s21_log = _load_s21_search_log(root) if root else None
    if s21_log:
        print("  [Stage 24] Loaded Stage 21 benchmark for comparison")

    # ── Fig 1: Attention Divergence Metrics ──
    if attn_analysis:
        per_layer = {}
        for key, metrics in attn_analysis.items():
            parts = key.split("_")
            layer_id = int(parts[-1].replace("layer", ""))
            if layer_id not in per_layer:
                per_layer[layer_id] = {"kl": [], "js": [], "cos": []}
            per_layer[layer_id]["kl"].append(metrics["kl_divergence"])
            per_layer[layer_id]["js"].append(metrics["js_divergence"])
            per_layer[layer_id]["cos"].append(metrics["cosine_similarity"])

        sorted_layers = sorted(per_layer.keys())
        avg_kl = [np.mean(per_layer[l]["kl"]) for l in sorted_layers]
        avg_js = [np.mean(per_layer[l]["js"]) for l in sorted_layers]
        avg_cos = [np.mean(per_layer[l]["cos"]) for l in sorted_layers]

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        axes[0].bar(range(len(sorted_layers)), avg_kl, color="#E74C3C", alpha=0.8)
        axes[0].set_xlabel("Layer")
        axes[0].set_ylabel("KL Divergence")
        axes[0].set_title("Attention KL Divergence\n(FP16 → Int4)")
        axes[0].set_xticks(range(len(sorted_layers)))
        axes[0].set_xticklabels(sorted_layers)

        axes[1].bar(range(len(sorted_layers)), avg_js, color="#9B59B6", alpha=0.8)
        axes[1].set_xlabel("Layer")
        axes[1].set_ylabel("JS Divergence")
        axes[1].set_title("Attention JS Divergence\n(FP16 → Int4)")
        axes[1].set_xticks(range(len(sorted_layers)))
        axes[1].set_xticklabels(sorted_layers)

        axes[2].bar(range(len(sorted_layers)), avg_cos, color="#3498DB", alpha=0.8)
        axes[2].set_xlabel("Layer")
        axes[2].set_ylabel("Cosine Similarity")
        axes[2].set_title("Attention Cosine Similarity\n(FP16 vs Int4)")
        cos_ylim = _auto_ylim(avg_cos)
        axes[2].set_ylim(*cos_ylim)

        fig.suptitle("Stage 24: Attention Map Divergence Under Int4 Quantization", fontsize=14, y=1.02)
        fig.tight_layout()
        _save(fig, out_dir, "s24_attention_divergence")
    else:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.axis("off")
        ax.text(0.5, 0.5,
                "Phase A: Attention Divergence Analysis — NO DATA\n\n"
                "Possible causes:\n"
                "  1. Calibration images not found (check image_path in calibration JSON)\n"
                "  2. eager attention mode not returning attention weights\n"
                "  3. OOM during full forward pass with output_attentions=True\n\n"
                "Fix: ensure calibration image paths exist on this server,\n"
                "then re-run Stage 24.",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=11, color="#C0392B",
                bbox=dict(facecolor="lightyellow", alpha=0.8, boxstyle="round"))
        ax.set_title("Stage 24: Attention Divergence (Empty)")
        fig.tight_layout()
        _save(fig, out_dir, "s24_attention_divergence_empty")

    # ── Fig 2: Search score breakdown (AUTO-SCALED) ──
    if search_log:
        attn_entries = [e for e in search_log if e["group"] == "attn"]
        if attn_entries:
            fig, ax = plt.subplots(figsize=(14, 5))
            x = np.arange(len(attn_entries))
            layer_labels = [f"L{e['layer_idx']}" for e in attn_entries]

            best_cka = []
            best_attn_f = []
            best_combined = []
            for e in attn_entries:
                best = e["best_algo"]
                d = e.get("detail", {}).get(best, {})
                best_cka.append(d.get("cka", 0))
                best_attn_f.append(d.get("attn_fidelity", 0))
                best_combined.append(d.get("combined", 0))

            w = 0.25
            ax.bar(x - w, best_cka, w, color=COLORS["text"], label="CKA", alpha=0.85)
            ax.bar(x, best_attn_f, w, color=COLORS["attn"], label="Attn Fidelity", alpha=0.85)
            ax.bar(x + w, best_combined, w, color="#2C3E50", label="Combined", alpha=0.85)

            all_vals = best_cka + best_attn_f + best_combined
            ylim = _auto_ylim(all_vals)
            ax.set_ylim(*ylim)

            step = max(1, len(x) // 20)
            ax.set_xticks(x[::step])
            ax.set_xticklabels([layer_labels[i] for i in range(0, len(layer_labels), step)])
            ax.set_xlabel("Decoder Layer")
            ax.set_ylabel("Score")
            ax.set_title("Attention Group: CKA vs Attention Fidelity (best algorithm)")
            ax.legend()
            fig.tight_layout()
            _save(fig, out_dir, "s24_attn_score_breakdown")

    # ── Fig 3: Saliency Map ──
    saliency_dir = results_dir / "saliency_maps"
    saliency_files = sorted(saliency_dir.glob("saliency_sample*.npy")) if saliency_dir.exists() else []
    if saliency_files:
        n_maps = min(len(saliency_files), 6)
        fig, axes = plt.subplots(1, n_maps, figsize=(4 * n_maps, 4))
        if n_maps == 1:
            axes = [axes]
        for idx, sf in enumerate(saliency_files[:n_maps]):
            sal = np.load(sf)
            num_v = sal.shape[0]
            side = int(np.sqrt(num_v))
            if side * side == num_v:
                sal_2d = sal.reshape(side, side)
            else:
                side = int(np.ceil(np.sqrt(num_v)))
                padded = np.zeros(side * side)
                padded[:num_v] = sal
                sal_2d = padded.reshape(side, side)

            im = axes[idx].imshow(sal_2d, cmap="hot", interpolation="bilinear")
            axes[idx].set_title(f"Sample {idx}", fontsize=10)
            axes[idx].axis("off")
            plt.colorbar(im, ax=axes[idx], shrink=0.7)

        fig.suptitle("Stage 24: Vision Token Saliency Maps", fontsize=14)
        fig.tight_layout()
        _save(fig, out_dir, "s24_saliency_maps")

    # ── Fig 4: Algorithm selection (with S21 benchmark) ──
    if search_log:
        n_rows = 4 if s21_log else 2
        fig, axes = plt.subplots(n_rows, 1,
                                  figsize=(max(14, len(search_log)*0.18), 2.5 * n_rows),
                                  sharex=True)
        _draw_algo_heatmap_with_benchmark(
            fig, axes, search_log,
            "Stage 24: Attention-Fidelity Algorithm Selection",
            s21_log=s21_log, out_dir=out_dir, filename="s24_algo_selection"
        )

    print(f"  Stage 24: {len(list(out_dir.glob('*.png')))} figures generated")


# ═══════════════════════════════════════════════════════════════════
#  Stage 25: Hard-Sample Calibration
# ═══════════════════════════════════════════════════════════════════

def visualize_stage25(results_dir: Path, root: Optional[Path] = None):
    out_dir = results_dir / "figures"
    out_dir.mkdir(exist_ok=True)
    jf = _find_json(results_dir, "stage25_results")
    if jf is None:
        print("  [Stage 25] No results JSON found, skipping.")
        return
    with open(jf) as f:
        data = json.load(f)

    diff_jf = _find_json(results_dir, "difficulty_scores")
    diff_data = None
    if diff_jf:
        with open(diff_jf) as f:
            diff_data = json.load(f)

    comparison = data.get("comparison", {})
    random_search = data.get("random_search", {})
    hard_search = data.get("hard_search", {})

    s21_log = _load_s21_search_log(root) if root else None
    if s21_log:
        print("  [Stage 25] Loaded Stage 21 benchmark for comparison")

    # ── Fig 1: Perplexity distribution histogram ──
    if diff_data:
        difficulties = diff_data.get("difficulties", [])
        ppls = [d["perplexity"] for d in difficulties if d["perplexity"] < 1e5]
        if ppls:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            axes[0].hist(ppls, bins=50, color="#3498DB", alpha=0.7, edgecolor="white")
            axes[0].axvline(np.median(ppls), color="red", linestyle="--", label=f"Median={np.median(ppls):.1f}")
            axes[0].axvline(np.mean(ppls), color="orange", linestyle="--", label=f"Mean={np.mean(ppls):.1f}")
            axes[0].set_xlabel("Perplexity")
            axes[0].set_ylabel("Count")
            axes[0].set_title("Sample Difficulty Distribution")
            axes[0].legend()

            log_ppls = np.log10(np.array(ppls) + 1)
            axes[1].hist(log_ppls, bins=50, color="#9B59B6", alpha=0.7, edgecolor="white")
            axes[1].set_xlabel("log10(Perplexity + 1)")
            axes[1].set_ylabel("Count")
            axes[1].set_title("Sample Difficulty Distribution (log scale)")

            fig.suptitle("Stage 25: Calibration Sample Difficulty Scores", fontsize=14, y=1.02)
            fig.tight_layout()
            _save(fig, out_dir, "s25_perplexity_distribution")

    # ── Fig 2: Strategy comparison heatmap (with S21 benchmark) ──
    random_log = random_search.get("search_log", [])
    hard_log = hard_search.get("search_log", [])

    if random_log and hard_log:
        random_map = {(e["layer_idx"], e["group"]): e["best_algo"] for e in random_log}
        hard_map = {(e["layer_idx"], e["group"]): e["best_algo"] for e in hard_log}
        s21_map = {(e["layer_idx"], e["group"]): e["best_algo"] for e in s21_log} if s21_log else {}
        all_keys = sorted(set(random_map.keys()) | set(hard_map.keys()))

        all_algo_list = list(random_map.values()) + list(hard_map.values())
        if s21_map:
            all_algo_list += list(s21_map.values())
        all_algos = sorted(set(all_algo_list))
        algo_to_idx = {a: i for i, a in enumerate(all_algos)}

        attn_keys = [(l, g) for l, g in all_keys if g == "attn"]

        n_rows = 4 if not s21_map else 5
        fig, axes = plt.subplots(n_rows, 1,
                                  figsize=(max(14, len(attn_keys)*0.35), 2 * n_rows),
                                  sharex=True,
                                  gridspec_kw={"height_ratios": [1]*min(n_rows-1, 4) + [0.4]})
        cmap_disc = matplotlib.colormaps.get_cmap("Set3").resampled(max(len(all_algos), 3))

        row_specs = [
            (axes[0], attn_keys, "Attn (Random)", random_map),
            (axes[1], attn_keys, "Attn (Hard)", hard_map),
        ]
        if s21_map:
            row_specs.append((axes[2], attn_keys, "Attn (S21)", s21_map))

        for ax, keys, gname, smap in row_specs:
            if not keys:
                continue
            colors = [cmap_disc(algo_to_idx.get(smap.get(k, ""), 0)) for k in keys]
            ax.bar(range(len(keys)), [1]*len(keys), color=colors, edgecolor="white", linewidth=0.5)
            ax.set_ylabel(gname, fontsize=9)
            ax.set_yticks([])

        diff_ax = axes[-2]
        diff_colors = ["#E74C3C" if random_map.get(k) != hard_map.get(k) else "#2ECC71" for k in attn_keys]
        diff_ax.bar(range(len(attn_keys)), [1]*len(attn_keys), color=diff_colors, edgecolor="white", linewidth=0.5)
        diff_ax.set_ylabel("Diff", fontsize=9)
        diff_ax.set_yticks([])

        axes[-1].axis("off")
        legend_handles = [Patch(facecolor=cmap_disc(algo_to_idx[a]), label=a) for a in all_algos]
        legend_handles += [Patch(facecolor="#E74C3C", label="Different"),
                           Patch(facecolor="#2ECC71", label="Same")]
        axes[-1].legend(handles=legend_handles, loc="center", ncol=min(4, len(legend_handles)),
                       fontsize=8, title="Algorithm / Agreement")

        step = max(1, len(attn_keys) // 20)
        diff_ax.set_xticks(range(0, len(attn_keys), step))
        diff_ax.set_xticklabels([str(k[0]) for k in attn_keys[::step]])
        diff_ax.set_xlabel("Decoder Layer")

        fig.suptitle("Stage 25: Random vs Hard Calibration Strategy Comparison (Attention)", fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        _save(fig, out_dir, "s25_strategy_comparison_attn")

        # MLP comparison
        mlp_keys = [(l, g) for l, g in all_keys if g == "mlp"]
        if mlp_keys:
            n_rows_mlp = 4 if not s21_map else 5
            fig, axes = plt.subplots(n_rows_mlp, 1,
                                      figsize=(max(14, len(mlp_keys)*0.35), 2 * n_rows_mlp),
                                      sharex=True,
                                      gridspec_kw={"height_ratios": [1]*min(n_rows_mlp-1, 4) + [0.4]})

            mlp_specs = [
                (axes[0], "MLP (Random)", random_map),
                (axes[1], "MLP (Hard)", hard_map),
            ]
            if s21_map:
                mlp_specs.append((axes[2], "MLP (S21)", s21_map))

            for ax, gname, smap in mlp_specs:
                colors = [cmap_disc(algo_to_idx.get(smap.get(k, ""), 0)) for k in mlp_keys]
                ax.bar(range(len(mlp_keys)), [1]*len(mlp_keys), color=colors, edgecolor="white", linewidth=0.5)
                ax.set_ylabel(gname, fontsize=9)
                ax.set_yticks([])

            diff_ax = axes[-2]
            diff_colors = ["#E74C3C" if random_map.get(k) != hard_map.get(k) else "#2ECC71" for k in mlp_keys]
            diff_ax.bar(range(len(mlp_keys)), [1]*len(mlp_keys), color=diff_colors, edgecolor="white", linewidth=0.5)
            diff_ax.set_ylabel("Diff", fontsize=9)
            diff_ax.set_yticks([])
            axes[-1].axis("off")
            axes[-1].legend(handles=legend_handles, loc="center", ncol=min(4, len(legend_handles)),
                           fontsize=8, title="Algorithm / Agreement")
            step = max(1, len(mlp_keys) // 20)
            diff_ax.set_xticks(range(0, len(mlp_keys), step))
            diff_ax.set_xticklabels([str(k[0]) for k in mlp_keys[::step]])
            diff_ax.set_xlabel("Decoder Layer")
            fig.suptitle("Stage 25: Random vs Hard Calibration Strategy Comparison (MLP)", fontsize=13)
            fig.tight_layout(rect=[0, 0, 1, 0.95])
            _save(fig, out_dir, "s25_strategy_comparison_mlp")

    # ── Fig 3: Per-layer CKA comparison (AUTO-SCALED) ──
    if random_log and hard_log:
        random_cka = {(e["layer_idx"], e["group"]): e.get("best_cka", 0) for e in random_log}
        hard_cka = {(e["layer_idx"], e["group"]): e.get("best_cka", 0) for e in hard_log}

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for ax, gname in [(axes[0], "attn"), (axes[1], "mlp")]:
            keys = sorted([k for k in random_cka if k[1] == gname])
            if not keys:
                continue
            r_vals = [random_cka[k] for k in keys]
            h_vals = [hard_cka.get(k, 0) for k in keys]
            x = np.arange(len(keys))
            w = 0.35
            ax.bar(x - w/2, r_vals, w, color=COLORS["random"], label="Random calib", alpha=0.8)
            ax.bar(x + w/2, h_vals, w, color=COLORS["hard"], label="Hard calib", alpha=0.8)
            step = max(1, len(x) // 15)
            ax.set_xticks(x[::step])
            ax.set_xticklabels([str(k[0]) for k in keys[::step]])
            ax.set_xlabel("Decoder Layer")
            ax.set_ylabel("Best CKA")
            ax.set_title(f"{gname.upper()} Group")
            ax.legend()
            ylim = _auto_ylim(r_vals + h_vals)
            ax.set_ylim(*ylim)

        fig.suptitle("Stage 25: Best CKA Score — Random vs Hard Calibration", fontsize=14, y=1.02)
        fig.tight_layout()
        _save(fig, out_dir, "s25_cka_comparison")

    # ── Fig 4: Summary ──
    if comparison:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.axis("off")
        total = comparison.get("total_sublayers", 0)
        diff = comparison.get("different_sublayers", 0)
        rate = comparison.get("difference_rate", 0)
        stats = data.get("difficulty_stats", {})

        lines = [
            f"Total sublayers:      {total}",
            f"Different sublayers:  {diff}",
            f"Difference rate:      {rate*100:.1f}%",
            f"",
            f"All samples mean PPL:  {stats.get('all_mean_ppl', 'N/A')}",
            f"Hard samples mean PPL: {stats.get('hard_mean_ppl', 'N/A')}",
            f"Easy samples mean PPL: {stats.get('easy_mean_ppl', 'N/A')}",
        ]
        ax.text(0.05, 0.95, "\n".join(lines), transform=ax.transAxes,
                fontsize=12, verticalalignment="top", fontfamily="monospace",
                bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.5))
        ax.set_title("Stage 25 Summary", fontsize=14)
        fig.tight_layout()
        _save(fig, out_dir, "s25_summary")

    print(f"  Stage 25: {len(list(out_dir.glob('*.png')))} figures generated")


# ═══════════════════════════════════════════════════════════════════
#  Cross-Stage Comparison (includes S21 config if available)
# ═══════════════════════════════════════════════════════════════════

def visualize_cross_stage(root: Path):
    """对比不同 stage 产出的量化配置差异。"""
    config_dir = root / "configs"
    if not config_dir.exists():
        print("  [Cross-stage] No configs directory found.")
        return

    out_dir = root / "figures"
    out_dir.mkdir(exist_ok=True)

    configs = {}
    for jf in sorted(config_dir.glob("stage*_*.json")):
        tag = jf.stem
        try:
            with open(jf) as f:
                configs[tag] = json.load(f)
        except Exception:
            continue

    if len(configs) < 2:
        print("  [Cross-stage] Need at least 2 config files to compare.")
        return

    config_names = list(configs.keys())
    all_layers = sorted(set().union(*(set(c.keys()) for c in configs.values())))

    sig_map = {}
    for cn in config_names:
        sig_map[cn] = {l: _algo_signature(configs[cn][l]) for l in all_layers if l in configs[cn]}

    all_sigs = sorted(set(s for sm in sig_map.values() for s in sm.values()))
    sig_to_idx = {s: i for i, s in enumerate(all_sigs)}

    fig, axes = plt.subplots(len(config_names), 1,
                              figsize=(max(16, len(all_layers)*0.15), 2*len(config_names)),
                              sharex=True)
    if len(config_names) <= 1:
        axes = [axes]
    cmap_disc = matplotlib.colormaps.get_cmap("Set3").resampled(max(len(all_sigs), 3))

    for ax, cn in zip(axes, config_names):
        colors = [cmap_disc(sig_to_idx.get(sig_map[cn].get(l, "RTN"), 0)) for l in all_layers]
        ax.bar(range(len(all_layers)), [1]*len(all_layers), color=colors, edgecolor="white", linewidth=0.3)
        is_s21 = "stage21" in cn
        short_name = cn.replace("_w4a4_", "\n").replace("stage", "S")
        weight = "bold" if is_s21 else "normal"
        ax.set_ylabel(short_name, fontsize=7, rotation=0, ha="right", va="center", fontweight=weight)
        ax.set_yticks([])
        if is_s21:
            ax.patch.set_alpha(0.3)

    step = max(1, len(all_layers) // 30)
    axes[-1].set_xticks(range(0, len(all_layers), step))
    short_labels = [l.split(".")[-2] + "." + l.split(".")[-1] if "." in l else l for l in all_layers]
    axes[-1].set_xticklabels([short_labels[i] for i in range(0, len(all_layers), step)],
                              rotation=90, fontsize=6)

    legend_handles = [Patch(facecolor=cmap_disc(sig_to_idx[s]), label=s) for s in all_sigs]
    fig.legend(handles=legend_handles, loc="upper right", ncol=min(4, len(all_sigs)),
               fontsize=7, title="Algorithm")
    fig.suptitle("Cross-Stage Quantization Strategy Comparison", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, out_dir, "cross_stage_comparison")

    print(f"  Cross-stage: figures saved to {out_dir}")


# ═══════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="InternVL-U Quantization Visualization")
    parser.add_argument("--stage", type=str, required=True,
                        choices=["22", "23", "24", "25", "cross", "all"],
                        help="Which stage to visualize (or 'all')")
    parser.add_argument("--results_dir", type=str, default=None,
                        help="Path to stage results directory")
    parser.add_argument("--root", type=str, default="./quantization_outputs",
                        help="Root output directory (used for 'all' or 'cross')")
    args = parser.parse_args()

    root = Path(args.root)

    if args.stage == "22":
        d = Path(args.results_dir) if args.results_dir else root / "stage22_modality_analysis"
        print(f"\n{'='*60}\nVisualizing Stage 22: {d}\n{'='*60}")
        visualize_stage22(d)
    elif args.stage == "23":
        d = Path(args.results_dir) if args.results_dir else root / "stage23_modality_weighted"
        print(f"\n{'='*60}\nVisualizing Stage 23: {d}\n{'='*60}")
        visualize_stage23(d, root=root)
    elif args.stage == "24":
        d = Path(args.results_dir) if args.results_dir else root / "stage24_attention_fidelity"
        print(f"\n{'='*60}\nVisualizing Stage 24: {d}\n{'='*60}")
        visualize_stage24(d, root=root)
    elif args.stage == "25":
        d = Path(args.results_dir) if args.results_dir else root / "stage25_hard_sample"
        print(f"\n{'='*60}\nVisualizing Stage 25: {d}\n{'='*60}")
        visualize_stage25(d, root=root)
    elif args.stage == "cross":
        print(f"\n{'='*60}\nCross-stage comparison: {root}\n{'='*60}")
        visualize_cross_stage(root)
    elif args.stage == "all":
        for s, sub in [("22", "stage22_modality_analysis"),
                        ("23", "stage23_modality_weighted"),
                        ("24", "stage24_attention_fidelity"),
                        ("25", "stage25_hard_sample")]:
            d = root / sub
            if d.exists():
                print(f"\n{'='*60}\nVisualizing Stage {s}: {d}\n{'='*60}")
                func = globals()[f"visualize_stage{s}"]
                if s in ("23", "24", "25"):
                    func(d, root=root)
                else:
                    func(d)
        visualize_cross_stage(root)

    print("\nDone!")


if __name__ == "__main__":
    main()
