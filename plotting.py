"""
Plotting helpers â€” each function returns a ``matplotlib.figure.Figure``
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
    labels = [row_label(r) for r in top]
    data = [insertion_sizes.get(lbl, [1]) for lbl in labels]

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
    ax.set_title(f"Insertion fragment distributions â€” top {len(labels)} enzymes")
    fig.tight_layout()
    return fig


def best_mid_poor_histogram(
    summary_rows: List[dict],
    insertion_sizes: Dict[str, List[int]],
    useful_min: int = 500,
    useful_max: int = 5000,
) -> plt.Figure:
    if len(summary_rows) < 3:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Need >= 3 enzymes", ha="center", va="center")
        return fig

    best = summary_rows[0]
    mid = summary_rows[len(summary_rows) // 2]
    poor = summary_rows[-1]
    picks = [
        ("Best", best),
        ("Medium", mid),
        ("Poor", poor),
    ]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for tag, row in picks:
        lbl = f"{tag}: {row_label(row)}"
        d = insertion_sizes.get(row_label(row), [])
        if not d:
            continue
        weights = [100.0 / len(d)] * len(d)
        ax.hist(d, bins=80, alpha=0.4, label=lbl, weights=weights)

    ax.set_xscale("log")
    ax.set_ylim(0, 100)
    ax.axvline(useful_min, color="black", ls="--", lw=1.3)
    ax.axvline(useful_max, color="black", ls="--", lw=1.3)
    ax.set_xlabel("Fragment size (bp, log)")
    ax.set_ylabel("% of insertions")
    ax.set_title("Best vs. median vs. poor enzyme")
    ax.legend(fontsize=8)
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
