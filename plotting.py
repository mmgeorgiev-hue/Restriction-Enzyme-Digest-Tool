"""
Plotting helpers each function returns a ``matplotlib.figure.Figure``
so the caller (Streamlit or a notebook) can display it however it likes.
"""

from __future__ import annotations

from typing import Dict, List, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from engine import row_label


def ranked_bar_chart(
    summary_rows: List[dict],
    value_key: str = "pct_usable_insertions",
    ylabel: str = "% usable insertions",
    title: str = "Ranked enzyme performance",
) -> plt.Figure:
    if not summary_rows:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig

    labels = [row_label(r) for r in summary_rows]
    values = [float(r[value_key]) for r in summary_rows]

    fig, ax = plt.subplots(figsize=(max(10, 0.28 * len(labels)), 5.5))
    bars = ax.bar(range(len(labels)), values, color="#4c78a8")
    ax.set_ylim(0, 100)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=75, ha="right", fontsize=7)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def top_violin(
    summary_rows: List[dict],
    insertion_sizes: Dict[str, List[int]],
    top_n: int = 12,
    jitter_max_points: int = 300,
) -> plt.Figure:
    import numpy as np

    if not summary_rows:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig

    top = summary_rows[:top_n]
    labels: List[str] = []
    data: List[List[int]] = []
    for r in top:
        lbl = row_label(r)
        sizes = [s for s in insertion_sizes.get(lbl, []) if s > 0]
        if not sizes:
            continue
        labels.append(lbl)
        data.append(sizes)

    if not data:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No insertion data", ha="center", va="center")
        return fig

    fig, ax = plt.subplots(figsize=(max(10, 0.7 * len(labels)), 5.5))

    parts = ax.violinplot(
        data, showmeans=False, showmedians=False, showextrema=False,
    )
    for body in parts.get("bodies", []):
        body.set_facecolor("#d3d3d3")
        body.set_edgecolor("black")
        body.set_linewidth(1.0)
        body.set_alpha(0.7)

    rng = np.random.default_rng(42)
    for i, d in enumerate(data, 1):
        arr = np.array(d, dtype=float)
        if len(arr) > jitter_max_points:
            arr = rng.choice(arr, jitter_max_points, replace=False)
        jitter = rng.uniform(-0.12, 0.12, size=len(arr))
        ax.scatter(
            i + jitter, arr, s=6, alpha=0.55, color="black", zorder=3,
            edgecolors="none",
        )

    ax.set_yscale("log")
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=8)
    ax.set_ylabel("Fragment size (bp, log)")
    ax.set_title(f"Insertion fragment distributions top {len(labels)} enzymes")
    fig.tight_layout()
    return fig


def best_mid_poor_histogram(
    summary_rows: List[dict],
    insertion_sizes: Dict[str, List[int]],
    useful_min: int = 500,
    useful_max: int = 5000,
) -> plt.Figure:
    import math

    if len(summary_rows) < 3:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Need >= 3 enzymes", ha="center", va="center")
        return fig

    picks = [
        ("Best", summary_rows[0]),
        ("Mid", summary_rows[len(summary_rows) // 2]),
        ("Poor", summary_rows[-1]),
    ]
    palette = {"Best": "#2ca25f", "Mid": "#3182bd", "Poor": "#de2d26"}

    all_sizes = [
        s for _, row in picks
        for s in insertion_sizes.get(row_label(row), []) if s > 0
    ]
    if not all_sizes:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig

    log_min = math.log10(max(1, min(all_sizes)))
    log_max = math.log10(max(all_sizes))
    n_bins = 35
    bins = [10 ** (log_min + (log_max - log_min) * i / n_bins) for i in range(n_bins + 1)]

    fig, ax = plt.subplots(figsize=(13, 8.5))
    for tag, row in picks:
        d = insertion_sizes.get(row_label(row), [])
        if not d:
            continue
        usable_pct = float(row["pct_usable_insertions"])
        lbl = f"{tag}: {row_label(row)} ({usable_pct:.1f}% usable)"
        weights = [100.0 / len(d)] * len(d)
        ax.hist(
            d, bins=bins, weights=weights, alpha=0.5,
            label=lbl, color=palette[tag], edgecolor="none",
        )

    ax.set_xscale("log")
    ax.set_ylim(0, 100)
    ax.axvline(useful_min, color="black", ls="--", lw=1.8)
    ax.axvline(useful_max, color="black", ls="--", lw=1.8)
    ax.axvspan(useful_min, useful_max, color="#b2df8a", alpha=0.12)
    ax.set_xlabel("Insertion-derived fragment size (bp, log scale)")
    ax.set_ylabel("% of simulated insertions in each size bin")
    ax.set_title("Best vs. mid vs. poor (ranked by % usable insertions)")
    ax.text(
        0.01, 0.98,
        f"Shaded region = usable iPCR window ({useful_min:,}\u2013{useful_max:,} bp)\n"
        "Bar height shows the % of insertions in each size bin.",
        transform=ax.transAxes, ha="left", va="top", fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"},
    )
    ax.legend(title="Overall usable insertion rate", fontsize=9)
    fig.tight_layout()
    return fig


def fragment_balance_bar(
    summary_rows: List[dict],
    insertion_sizes: Dict[str, List[int]],
    useful_min: int = 500,
    useful_max: int = 5000,
    top_n: int = 20,
) -> plt.Figure:
    """Horizontal stacked bar: % too-small / usable / too-large per enzyme."""
    if not summary_rows:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig

    top = summary_rows[:top_n]
    labels = [row_label(r) for r in top]
    pct_small, pct_useful, pct_large = [], [], []

    for r in top:
        d = insertion_sizes.get(row_label(r), [])
        n = len(d)
        if n == 0:
            pct_small.append(0.0)
            pct_useful.append(0.0)
            pct_large.append(0.0)
            continue
        ns = sum(1 for x in d if x < useful_min)
        nu = sum(1 for x in d if useful_min <= x <= useful_max)
        nl = sum(1 for x in d if x > useful_max)
        pct_small.append(100.0 * ns / n)
        pct_useful.append(100.0 * nu / n)
        pct_large.append(100.0 * nl / n)

    fig_h = max(6, 0.35 * len(labels))
    fig, ax = plt.subplots(figsize=(12, fig_h))
    ax.barh(labels, pct_small, color="#d95f02", label=f"< {useful_min:,} bp")
    ax.barh(labels, pct_useful, left=pct_small, color="#1b9e77",
            label=f"Usable ({useful_min:,}\u2013{useful_max:,} bp)")
    left_for_large = [s + u for s, u in zip(pct_small, pct_useful)]
    ax.barh(labels, pct_large, left=left_for_large, color="#7570b3",
            label=f"> {useful_max:,} bp")
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of simulated insertions")
    ax.set_title("Insertion outcome balance")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="x", ls=":", lw=0.8, alpha=0.6)
    fig.tight_layout()
    return fig


def sites_per_chromosome_heatmap(
    summary_rows: List[dict],
    motif_metrics: dict,
    top_n: int = 15,
) -> plt.Figure:
    """Simple heatmap of cut-site density per chromosome for top enzymes."""
    import numpy as np

    top = summary_rows[:top_n]
    if not top:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig

    all_chroms: list = []
    for r in top:
        m = motif_metrics[r["site"]]
        for c in m.sites_per_chromosome:
            if c not in all_chroms:
                all_chroms.append(c)
    all_chroms.sort()

    labels = [row_label(r) for r in top]
    matrix = []
    for r in top:
        m = motif_metrics[r["site"]]
        matrix.append([m.sites_per_chromosome.get(c, 0) for c in all_chroms])

    arr = np.array(matrix, dtype=float)

    fig, ax = plt.subplots(figsize=(max(8, 0.6 * len(all_chroms)), max(5, 0.4 * len(labels))))
    im = ax.imshow(arr, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(all_chroms)))
    ax.set_xticklabels(all_chroms, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title("Cut sites per chromosome")
    fig.colorbar(im, ax=ax, shrink=0.7, label="# sites")
    fig.tight_layout()
    return fig
