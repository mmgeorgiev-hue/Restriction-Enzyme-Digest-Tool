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
    cmap_name: str = "YlOrRd",
) -> plt.Figure:
    import numpy as np
    from matplotlib import cm
    from matplotlib.colors import Normalize

    if not summary_rows:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig

    labels = [row_label(r) for r in summary_rows]
    values = [float(r[value_key]) for r in summary_rows]

    cmap = cm.get_cmap(cmap_name)
    norm = Normalize(vmin=0, vmax=100)
    colors = [cmap(norm(v)) for v in values]

    fig, ax = plt.subplots(figsize=(max(10, 0.28 * len(labels)), 5.5))
    ax.bar(range(len(labels)), values, color=colors, edgecolor="black", linewidth=0.4)
    ax.set_ylim(0, 100)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=75, ha="right", fontsize=7)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array(np.asarray(values))
    cbar = fig.colorbar(sm, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label(ylabel)

    fig.tight_layout()
    return fig


def top_violin(
    summary_rows: List[dict],
    insertion_sizes: Dict[str, List[int]],
    top_n: int = 12,
    jitter_max_points: int = 300,
    useful_min: int = 500,
    useful_max: int = 5000,
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

    try:
        from scipy.stats import gaussian_kde
        _have_kde = True
    except ImportError:
        _have_kde = False

    rng = np.random.default_rng(42)
    width = 0.38

    for i, d in enumerate(data, 1):
        arr = np.asarray([x for x in d if x > 0], dtype=float)
        if arr.size == 0:
            continue

        log_arr = np.log10(arr)
        lo, hi = float(log_arr.min()), float(log_arr.max())
        pad = max(0.15, 0.05 * (hi - lo))
        ys_log = np.linspace(lo - pad, hi + pad, 256)

        if _have_kde and arr.size > 1 and (hi - lo) > 0:
            kde = gaussian_kde(log_arr, bw_method="scott")
            density = kde(ys_log)
        else:
            counts, edges = np.histogram(log_arr, bins=20)
            centers = 0.5 * (edges[:-1] + edges[1:])
            density = np.interp(ys_log, centers, counts.astype(float))

        if density.max() > 0:
            density = density / density.max() * width

        ys = 10 ** ys_log
        ax.fill_betweenx(
            ys, i - density, i + density,
            facecolor="#d3d3d3", edgecolor="black", linewidth=1.0,
            alpha=0.7, zorder=2,
        )

        sample = arr if arr.size <= jitter_max_points else rng.choice(
            arr, jitter_max_points, replace=False,
        )
        jitter = rng.uniform(-0.12, 0.12, size=sample.size)
        ax.scatter(
            i + jitter, sample, s=6, alpha=0.55, color="black",
            zorder=3, edgecolors="none",
        )

    ax.set_yscale("log")
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=8)
    ax.set_ylabel("Fragment size (bp, log scale)")
    ax.set_title(f"Insertion fragment distributions for top {len(labels)} enzymes")

    ax.axhspan(useful_min, useful_max, color="#b2df8a", alpha=0.15, zorder=0)
    ax.axhline(useful_min, color="#1b9e77", lw=0.8, ls="--", zorder=1)
    ax.axhline(useful_max, color="#1b9e77", lw=0.8, ls="--", zorder=1)

    n_per_enzyme = [
        len([s for s in insertion_sizes.get(lbl, []) if s > 0]) for lbl in labels
    ]
    n_text = (
        f"n = {n_per_enzyme[0]:,} simulated insertions per enzyme"
        if n_per_enzyme and len(set(n_per_enzyme)) == 1
        else f"n = {min(n_per_enzyme):,}\u2013{max(n_per_enzyme):,} per enzyme"
    )

    caption = (
        "Figure. Distribution of in silico iPCR-amplifiable fragment sizes for the "
        f"top {len(labels)} enzymes, ranked by % usable insertions. Each violin is a "
        "kernel density estimate of fragment lengths (computed in log10 space and "
        "mirrored to give a symmetric two-sided shape). Black dots are individual "
        "simulated insertions (jittered horizontally; subsampled to "
        f"{jitter_max_points:,} per enzyme for legibility). The shaded green band "
        f"marks the usable iPCR window ({useful_min:,}\u2013{useful_max:,} bp); "
        "fragments that fall inside this band can be amplified and sequenced. "
        f"{n_text}."
    )

    fig.tight_layout(rect=(0, 0.18, 1, 1))
    fig.text(
        0.02, 0.02, caption,
        ha="left", va="bottom", fontsize=8.5, wrap=True,
        bbox={"facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.9},
    )
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
    exclude_patterns: tuple = ("scaffold", "centromere", "unplaced", "chrUn"),
) -> plt.Figure:
    """Heatmap of cut-site density per chromosome for top enzymes.

    Contigs whose names contain any of the substrings in *exclude_patterns*
    (case-insensitive) are filtered out, so the figure shows only the
    primary chromosomes (e.g. Bd1-Bd5) and not centromere or scaffold
    contigs that would otherwise dominate the x-axis.
    """
    import numpy as np

    top = summary_rows[:top_n]
    if not top:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig

    def _keep(name: str) -> bool:
        n = name.lower()
        return not any(p.lower() in n for p in exclude_patterns)

    all_chroms: list = []
    for r in top:
        m = motif_metrics[r["site"]]
        for c in m.sites_per_chromosome:
            if c not in all_chroms and _keep(c):
                all_chroms.append(c)
    all_chroms.sort()

    if not all_chroms:
        fig, ax = plt.subplots()
        ax.text(
            0.5, 0.5,
            "All contigs were filtered out by exclude_patterns",
            ha="center", va="center",
        )
        return fig

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
