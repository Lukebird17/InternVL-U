#!/usr/bin/env python3
"""
Visualize MME & GenEval evaluation results across quantization configs.
Generates:
  1. MME grouped bar (Perception / Cognition stacked) — W4A4 only
  2. GenEval grouped bar — W4A4 only
  3. Cross-bit degradation line chart (MME Total & GenEval vs bit-width)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

# ============================================================
# Data
# ============================================================

BASELINE = {
    "label": "FP16",
    "mme_p": 1553.61, "mme_c": 507.50,
    "geneval": 0.83,
}

UNIFORM_W4 = [
    {"label": "SmoothQuant",  "mme_p": 1459.29, "mme_c": 441.07, "geneval": 0.86},
    {"label": "SVDQuant",     "mme_p": 1480.47, "mme_c": 413.93, "geneval": 0.86},
    {"label": "GPTQ",         "mme_p": 1417.08, "mme_c": 392.14, "geneval": 0.84},
    {"label": "AWQ",          "mme_p": 1301.61, "mme_c": 447.86, "geneval": 0.83},
]

MIXED_CONFIGS = {
    "CALM": {
        4: {"mme_p": 1398.31, "mme_c": 506.07, "geneval": 0.83},
        3: {"mme_p":  298.55, "mme_c":  33.57, "geneval": 0.61},
        2: {"mme_p":    0.00, "mme_c":   0.00, "geneval": 0.00},
    },
    "Exhaustive": {
        4: {"mme_p": 1382.51, "mme_c": 245.36, "geneval": 0.78},
        3: {"mme_p":  269.10, "mme_c":  20.36, "geneval": 0.68},
        2: {"mme_p":    0.00, "mme_c":   0.00, "geneval": 0.00},
    },
    "FuncGroup": {
        4: {"mme_p": 1170.62, "mme_c": 313.21, "geneval": 0.79},
        3: {"mme_p":  571.58, "mme_c": 173.21, "geneval": 0.73},
        2: {"mme_p":    0.00, "mme_c":   0.00, "geneval": 0.00},
    },
}

COLORS = {
    "FP16":         "#7f7f7f",
    "SmoothQuant":  "#82b8d9",
    "SVDQuant":     "#3878a0",
    "GPTQ":         "#e6c36a",
    "AWQ":          "#e89a6c",
    "CALM":         "#5aaa5a",
    "Exhaustive":   "#b07aa1",
    "FuncGroup":    "#d4845f",
}

# ============================================================
# Fig 1 — W4A4 MME stacked bar (Perception + Cognition)
# ============================================================

def plot_mme_w4(output_dir: Path, dpi=150):
    entries = [BASELINE] + UNIFORM_W4 + [
        {"label": lbl, **MIXED_CONFIGS[lbl][4]}
        for lbl in ["CALM", "Exhaustive", "FuncGroup"]
    ]

    labels = [e["label"] for e in entries]
    percs  = [e["mme_p"] for e in entries]
    cogns  = [e["mme_c"] for e in entries]
    totals = [p + c for p, c in zip(percs, cogns)]
    colors = [COLORS[l] for l in labels]

    n = len(labels)
    x = np.arange(n)
    w = 0.55

    fig, ax = plt.subplots(figsize=(max(n * 1.1, 9), 5))

    bars_p = ax.bar(x, percs, w, color=colors, alpha=0.85, label="Perception")
    bars_c = ax.bar(x, cogns, w, bottom=percs, color=colors, alpha=0.50, label="Cognition")

    for i, (xi, t) in enumerate(zip(x, totals)):
        ax.text(xi, t + 20, f"{t:.0f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    fp16_total = BASELINE["mme_p"] + BASELINE["mme_c"]
    ax.axhline(y=fp16_total, color="#7f7f7f", linewidth=1, linestyle="--", alpha=0.6)
    ax.text(n - 0.5, fp16_total + 15, f"FP16 = {fp16_total:.0f}", fontsize=7,
            color="#7f7f7f", ha="right")

    ax.axvline(x=0.5, color="#cccccc", linewidth=1, linestyle=":")
    ax.axvline(x=4.5, color="#cccccc", linewidth=1, linestyle=":")
    ax.text(0, -140, "Baseline", ha="center", fontsize=7, color="#999")
    ax.text(2.5, -140, "Uniform W4A4", ha="center", fontsize=7, color="#999")
    ax.text(6, -140, "Mixed-precision W4A4", ha="center", fontsize=7, color="#999")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, rotation=20, ha="right")
    ax.set_ylabel("MME Score", fontsize=11)
    ax.set_ylim(0, max(totals) * 1.12)
    ax.set_title("MME Benchmark — W4A4 Quantization Comparison", fontsize=13, fontweight="bold")

    legend_p = matplotlib.patches.Patch(facecolor="#888888", alpha=0.85, label="Perception")
    legend_c = matplotlib.patches.Patch(facecolor="#888888", alpha=0.45, label="Cognition")
    ax.legend(handles=[legend_p, legend_c], loc="upper right", fontsize=8)

    fig.tight_layout()
    out = output_dir / "eval_mme_w4a4.png"
    fig.savefig(str(out), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ============================================================
# Fig 2 — W4A4 GenEval grouped bar
# ============================================================

def plot_geneval_w4(output_dir: Path, dpi=150):
    entries = [BASELINE] + UNIFORM_W4 + [
        {"label": lbl, **MIXED_CONFIGS[lbl][4]}
        for lbl in ["CALM", "Exhaustive", "FuncGroup"]
    ]

    labels = [e["label"] for e in entries]
    scores = [e["geneval"] for e in entries]
    colors = [COLORS[l] for l in labels]

    n = len(labels)
    x = np.arange(n)
    w = 0.55

    fig, ax = plt.subplots(figsize=(max(n * 1.1, 9), 4.5))
    bars = ax.bar(x, scores, w, color=colors, alpha=0.85, edgecolor="white", linewidth=0.5)

    for xi, s in zip(x, scores):
        ax.text(xi, s + 0.008, f"{s:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.axhline(y=BASELINE["geneval"], color="#7f7f7f", linewidth=1, linestyle="--", alpha=0.6)

    ax.axvline(x=0.5, color="#cccccc", linewidth=1, linestyle=":")
    ax.axvline(x=4.5, color="#cccccc", linewidth=1, linestyle=":")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, rotation=20, ha="right")
    ax.set_ylabel("GenEval ALL Score", fontsize=11)
    ax.set_ylim(0, 1.02)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.set_title("GenEval Benchmark — W4A4 Quantization Comparison", fontsize=13, fontweight="bold")

    fig.tight_layout()
    out = output_dir / "eval_geneval_w4a4.png"
    fig.savefig(str(out), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ============================================================
# Fig 3 — Cross-bit degradation (MME Total + GenEval, dual axis)
# ============================================================

def plot_cross_bit(output_dir: Path, dpi=150):
    wbits = [4, 3, 2]
    fp16_mme = BASELINE["mme_p"] + BASELINE["mme_c"]
    fp16_ge  = BASELINE["geneval"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # --- Left: MME Total ---
    ax = axes[0]
    ax.axhline(y=fp16_mme, color="#7f7f7f", linewidth=1.5, linestyle="--", alpha=0.5, label="FP16")

    for name, style in [("CALM", "o-"), ("Exhaustive", "s-"), ("FuncGroup", "D-")]:
        vals = []
        for wb in wbits:
            d = MIXED_CONFIGS[name][wb]
            vals.append(d["mme_p"] + d["mme_c"])
        ax.plot(wbits, vals, style, color=COLORS[name], label=name,
                linewidth=2, markersize=7, alpha=0.9)

    for name, d in [("SmoothQuant", UNIFORM_W4[0]), ("SVDQuant", UNIFORM_W4[1]),
                     ("GPTQ", UNIFORM_W4[2]), ("AWQ", UNIFORM_W4[3])]:
        total = d["mme_p"] + d["mme_c"]
        ax.plot(4, total, "^", color=COLORS[name], markersize=9, alpha=0.85, label=f"{name} (uniform)")

    ax.set_xlabel("Weight Bits", fontsize=11)
    ax.set_ylabel("MME Total Score", fontsize=11)
    ax.set_title("MME Degradation across Bit-widths", fontsize=12, fontweight="bold")
    ax.set_xticks(wbits)
    ax.set_xlim(1.5, 4.5)
    ax.set_ylim(-50, fp16_mme * 1.15)
    ax.legend(fontsize=7.5, loc="upper left", ncol=2)
    ax.grid(axis="y", alpha=0.3)

    # --- Right: GenEval ---
    ax = axes[1]
    ax.axhline(y=fp16_ge, color="#7f7f7f", linewidth=1.5, linestyle="--", alpha=0.5, label="FP16")

    for name, style in [("CALM", "o-"), ("Exhaustive", "s-"), ("FuncGroup", "D-")]:
        vals = [MIXED_CONFIGS[name][wb]["geneval"] for wb in wbits]
        ax.plot(wbits, vals, style, color=COLORS[name], label=name,
                linewidth=2, markersize=7, alpha=0.9)

    for name, d in [("SmoothQuant", UNIFORM_W4[0]), ("SVDQuant", UNIFORM_W4[1]),
                     ("GPTQ", UNIFORM_W4[2]), ("AWQ", UNIFORM_W4[3])]:
        ax.plot(4, d["geneval"], "^", color=COLORS[name], markersize=9, alpha=0.85,
                label=f"{name} (uniform)")

    ax.set_xlabel("Weight Bits", fontsize=11)
    ax.set_ylabel("GenEval ALL Score", fontsize=11)
    ax.set_title("GenEval Degradation across Bit-widths", fontsize=12, fontweight="bold")
    ax.set_xticks(wbits)
    ax.set_xlim(1.5, 4.5)
    ax.set_ylim(-0.03, 1.02)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.legend(fontsize=7.5, loc="upper left", ncol=2)
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Quantization Quality vs. Bit-width", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    out = output_dir / "eval_cross_bit_degradation.png"
    fig.savefig(str(out), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ============================================================
# Fig 4 — W4A4 MME sub-category radar / heatmap
# ============================================================

def plot_mme_subcategory(output_dir: Path, dpi=150):
    """Heatmap of MME sub-category scores (normalized to FP16)."""
    categories = [
        "existence", "count", "position", "color", "posters",
        "celebrity", "scene", "landmark", "artwork", "OCR",
        "commonsense", "numerical", "translation", "code",
    ]
    cat_short = [
        "Exist", "Count", "Pos", "Color", "Poster",
        "Celeb", "Scene", "Land", "Art", "OCR",
        "Common", "Numer", "Trans", "Code",
    ]

    fp16_raw = [195.0, 158.33, 118.33, 170.0, 128.91, 158.53, 158.75, 148.25, 147.5, 170.0,
                115.0, 100.0, 192.5, 100.0]

    configs = {
        "SmoothQuant": [195.0, 136.67, 105.0, 160.0, 126.53, 150.59, 161.75, 137.0, 131.75, 155.0,
                        108.57, 100.0, 155.0, 77.5],
        "SVDQuant":    [190.0, 148.33, 128.33, 175.0, 112.59, 126.47, 156.5, 143.25, 145.0, 155.0,
                        116.43, 90.0, 132.5, 75.0],
        "GPTQ":        [195.0, 153.33, 96.67, 160.0, 95.92, 159.41, 150.5, 141.5, 139.75, 125.0,
                        107.14, 115.0, 65.0, 105.0],
        "AWQ":         [190.0, 148.33, 105.0, 145.0, 104.76, 86.76, 159.5, 123.25, 121.5, 117.5,
                        97.86, 92.5, 185.0, 72.5],
        "CALM":        [195.0, 128.33, 98.33, 158.33, 128.57, 118.24, 154.75, 123.75, 138.0, 155.0,
                        108.57, 120.0, 155.0, 122.5],
        "Exhaustive":  [195.0, 148.33, 123.33, 168.33, 111.56, 82.94, 156.75, 117.0, 131.75, 147.5,
                        97.86, 37.5, 80.0, 30.0],
        "FuncGroup":   [170.0, 118.33, 88.33, 165.0, 88.44, 51.76, 143.5, 67.25, 123.0, 155.0,
                        95.71, 47.5, 95.0, 75.0],
    }

    config_labels = list(configs.keys())
    n_configs = len(config_labels)
    n_cats = len(categories)

    ratios = np.zeros((n_configs, n_cats))
    for i, name in enumerate(config_labels):
        for j in range(n_cats):
            fp = fp16_raw[j]
            ratios[i, j] = configs[name][j] / fp if fp > 0 else 0.0

    fig, ax = plt.subplots(figsize=(12, 4.5))
    im = ax.imshow(ratios, cmap="RdYlGn", vmin=0.2, vmax=1.1, aspect="auto")

    for i in range(n_configs):
        for j in range(n_cats):
            val = ratios[i, j]
            color = "white" if val < 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7.5, color=color)

    ax.set_xticks(range(n_cats))
    ax.set_xticklabels(cat_short, fontsize=9, rotation=35, ha="right")
    ax.set_yticks(range(n_configs))
    ax.set_yticklabels(config_labels, fontsize=9)
    ax.set_title("W4A4 MME Sub-category Scores (ratio to FP16)", fontsize=13, fontweight="bold")

    ax.axvline(x=9.5, color="white", linewidth=2)
    ax.text(4.5, -0.8, "── Perception ──", ha="center", fontsize=8, color="#666")
    ax.text(12, -0.8, "── Cognition ──", ha="center", fontsize=8, color="#666")

    ax.axhline(y=3.5, color="white", linewidth=2)
    ax.text(-1.8, 1.5, "Uniform", ha="center", fontsize=8, color="#666", rotation=90)
    ax.text(-1.8, 5, "Mixed", ha="center", fontsize=8, color="#666", rotation=90)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Ratio to FP16", fontsize=9)

    fig.tight_layout()
    out = output_dir / "eval_mme_subcategory_w4a4.png"
    fig.savefig(str(out), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ============================================================
# Main
# ============================================================

def main():
    output_dir = Path(__file__).parent / "quantization_outputs" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(">>> MME W4A4 bar chart")
    plot_mme_w4(output_dir)

    print(">>> GenEval W4A4 bar chart")
    plot_geneval_w4(output_dir)

    print(">>> Cross-bit degradation")
    plot_cross_bit(output_dir)

    print(">>> MME sub-category heatmap")
    plot_mme_subcategory(output_dir)

    print(f"\nAll figures saved to: {output_dir}")


if __name__ == "__main__":
    main()
