"""
Publication-quality visualisations for continual learning experiments.

All functions return a matplotlib Figure and optionally save it to disk.
Usage::

    from utils.visualization import plot_accuracy_matrix, plot_seed_stability
    fig = plot_accuracy_matrix(acc_matrix, save_path="figures/acc_matrix.pdf")
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def _save_or_show(fig: plt.Figure, save_path: Optional[Path | str]) -> plt.Figure:
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig


def plot_accuracy_matrix(
    acc_matrix: np.ndarray,
    task_names: Optional[list[str]] = None,
    title: str = "Accuracy Matrix",
    save_path: Optional[Path | str] = None,
) -> plt.Figure:
    """
    Heat-map of acc_matrix[i, j] = accuracy on task j after training task i.

    Standard CL visualisation (Lopez-Paz & Ranzato, 2017).
    NaN cells are shown in light grey.
    """
    n = acc_matrix.shape[0]
    labels = task_names or [f"T{i}" for i in range(n)]

    fig, ax = plt.subplots(figsize=(max(5, n), max(4, n - 1)))
    masked = np.ma.masked_invalid(acc_matrix)
    cmap = plt.cm.Blues.copy()
    cmap.set_bad(color="#eeeeee")

    im = ax.imshow(masked, vmin=0, vmax=100, cmap=cmap, aspect="auto")
    plt.colorbar(im, ax=ax, label="Accuracy (%)")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Evaluated on Task")
    ax.set_ylabel("After Training Task")
    ax.set_title(title)

    for i in range(n):
        for j in range(n):
            val = acc_matrix[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                        fontsize=8, color="white" if val > 60 else "black")

    fig.tight_layout()
    return _save_or_show(fig, save_path)


def plot_forgetting_curve(
    forgetting_per_method: dict[str, list[float]],
    title: str = "Average Forgetting per Task",
    save_path: Optional[Path | str] = None,
) -> plt.Figure:
    """
    Line plot of average forgetting vs number of tasks trained.

    Args:
        forgetting_per_method: {method_name: [forgetting after task 1, task 2, ...]}.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, values in forgetting_per_method.items():
        xs = range(1, len(values) + 1)
        ax.plot(xs, values, marker="o", label=name)

    ax.set_xlabel("Tasks Trained")
    ax.set_ylabel("Average Forgetting (%)")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return _save_or_show(fig, save_path)


def plot_causal_effect_distribution(
    causal_effects: list[float],
    threshold: float = 0.05,
    title: str = "ATE Distribution",
    save_path: Optional[Path | str] = None,
) -> plt.Figure:
    """
    Histogram of Average Treatment Effect (ATE) scores.

    Colour-codes regions as:
      - Beneficial (ATE < -threshold): green
      - Neutral: grey
      - Harmful (ATE > threshold): red
    """
    arr = np.array(causal_effects)

    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.linspace(arr.min() - 0.05, arr.max() + 0.05, 40)

    # Three colour bands
    for lo, hi, colour, label in [
        (arr.min() - 1, -threshold, "#4caf50", "Beneficial"),
        (-threshold, threshold, "#9e9e9e", "Neutral"),
        (threshold, arr.max() + 1, "#f44336", "Harmful"),
    ]:
        mask = (arr >= lo) & (arr < hi)
        if mask.any():
            ax.hist(arr[mask], bins=bins, color=colour, alpha=0.75, label=label)

    ax.axvline(-threshold, color="green", linestyle="--", linewidth=1)
    ax.axvline(threshold, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("ATE (causal effect size)")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return _save_or_show(fig, save_path)


def plot_compute_vs_accuracy(
    methods: list[dict],
    title: str = "Compute–Accuracy Pareto Frontier",
    save_path: Optional[Path | str] = None,
) -> plt.Figure:
    """
    Scatter/line plot of runtime vs Class-IL accuracy — Pareto frontier view.

    Args:
        methods: List of dicts with keys:
            name (str), runtime_min (float), class_il (float),
            optionally: std (float), colour (str).

    Example::

        methods = [
            {"name": "Vanilla DER++",   "runtime_min": 43,   "class_il": 22.33, "std": 0.77},
            {"name": "Heuristic",        "runtime_min": 43,   "class_il": 21.82},
            {"name": "Hybrid",           "runtime_min": 120,  "class_il": 22.8},
            {"name": "TRUE Causality",   "runtime_min": 780,  "class_il": 23.52, "std": 1.18},
            {"name": "Influence Fn.",    "runtime_min": 120,  "class_il": 23.1,  "std": 0.9},
        ]
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    colours = plt.cm.tab10.colors

    for i, m in enumerate(methods):
        c = m.get("colour", colours[i % len(colours)])
        ax.scatter(m["runtime_min"], m["class_il"], s=120, color=c, zorder=5)
        if "std" in m:
            ax.errorbar(m["runtime_min"], m["class_il"], yerr=m["std"],
                        fmt="none", color=c, capsize=4)
        ax.annotate(
            m["name"], (m["runtime_min"], m["class_il"]),
            textcoords="offset points", xytext=(8, 4), fontsize=9,
        )

    ax.set_xlabel("Runtime per seed (minutes, RTX 5090)")
    ax.set_ylabel("Class-IL Accuracy (%)")
    ax.set_title(title)
    ax.set_xscale("log")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return _save_or_show(fig, save_path)


def plot_seed_stability(
    seed_results: dict[str, list[float]],
    title: str = "Seed Stability",
    save_path: Optional[Path | str] = None,
) -> plt.Figure:
    """
    Box plot comparing seed-to-seed variance across methods.

    Args:
        seed_results: {method_name: [seed1_acc, seed2_acc, ...]}.
    """
    fig, ax = plt.subplots(figsize=(max(5, len(seed_results) * 1.5), 5))

    names = list(seed_results.keys())
    data = [seed_results[n] for n in names]

    bp = ax.boxplot(data, patch_artist=True, medianprops={"color": "black", "linewidth": 2})
    colours = plt.cm.tab10.colors
    for patch, colour in zip(bp["boxes"], colours):
        patch.set_facecolor(colour)
        patch.set_alpha(0.7)

    # Overlay individual seed points
    for i, (name, scores) in enumerate(seed_results.items(), start=1):
        xs = np.random.default_rng(i).uniform(i - 0.15, i + 0.15, size=len(scores))
        ax.scatter(xs, scores, color="black", s=20, zorder=5, alpha=0.7)

    ax.set_xticks(range(1, len(names) + 1))
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("Class-IL Accuracy (%)")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return _save_or_show(fig, save_path)


if __name__ == "__main__":
    import tempfile

    print("Generating example figures...")

    # Seed stability from paper results
    fig = plot_seed_stability(
        {
            "Vanilla DER++": [22.6, 22.87, 21.15, 23.12, 21.9],
            "TRUE Causality": [24.04, 25.09, 23.03, 21.73, 23.72],
        },
        save_path="figures/seed_stability.pdf",
    )
    plt.close(fig)

    # Pareto frontier
    fig = plot_compute_vs_accuracy(
        [
            {"name": "Vanilla DER++", "runtime_min": 43, "class_il": 22.33, "std": 0.77},
            {"name": "TRUE Causality", "runtime_min": 780, "class_il": 23.52, "std": 1.18},
        ],
        save_path="figures/pareto.pdf",
    )
    plt.close(fig)

    print("Done — saved to figures/")
