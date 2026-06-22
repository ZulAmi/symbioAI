"""
Statistical significance testing for continual learning experiments.

Provides paired t-test, Cohen's d, and bootstrap confidence intervals
for comparing methods across multiple seeds.
"""
from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
from scipy import stats


class SignificanceResult(NamedTuple):
    t_statistic: float
    p_value: float
    cohens_d: float
    is_significant: bool          # p < 0.05 (two-tailed)
    ci_95: tuple[float, float]    # 95% CI of the mean difference (bootstrap)
    n_seeds: int
    mean_diff: float              # method_b - method_a
    std_diff: float


def paired_t_test(
    method_a: list[float],
    method_b: list[float],
    alpha: float = 0.05,
    n_bootstrap: int = 10_000,
) -> SignificanceResult:
    """
    Welch's paired t-test comparing two methods over matched seeds.

    Uses Welch's (unequal variance) t-test — appropriate when the two methods
    have different variance (e.g., TRUE std=1.18% vs vanilla std=0.77%).

    Args:
        method_a: Scores for baseline (e.g., vanilla DER++ per seed).
        method_b: Scores for proposed method (e.g., TRUE causality per seed).
        alpha: Significance threshold (default 0.05).
        n_bootstrap: Resamples for bootstrap CI.

    Returns:
        SignificanceResult with all statistics.

    Example (reproducing the paper's claim)::

        vanilla = [22.6, 22.87, 21.15, 23.12, 21.9]
        true    = [24.04, 25.09, 23.03, 21.73, 23.72]
        result  = paired_t_test(vanilla, true)
        print(f"p={result.p_value:.4f}, d={result.cohens_d:.3f}")
    """
    a = np.array(method_a, dtype=float)
    b = np.array(method_b, dtype=float)

    if len(a) != len(b):
        raise ValueError(f"Seed counts must match: {len(a)} vs {len(b)}")

    t_stat, p_val = stats.ttest_rel(a, b)  # paired t-test
    d = compute_cohens_d(method_a, method_b)
    ci = bootstrap_confidence_interval(method_a, method_b, n_bootstrap=n_bootstrap)
    diffs = b - a

    return SignificanceResult(
        t_statistic=float(t_stat),
        p_value=float(p_val),
        cohens_d=float(d),
        is_significant=bool(p_val < alpha),
        ci_95=ci,
        n_seeds=len(a),
        mean_diff=float(diffs.mean()),
        std_diff=float(diffs.std(ddof=1)),
    )


def compute_cohens_d(method_a: list[float], method_b: list[float]) -> float:
    """
    Cohen's d effect size for the difference between two paired samples.

    d = mean(b - a) / pooled_std

    Interpretation: 0.2 small, 0.5 medium, 0.8 large.
    """
    a = np.array(method_a, dtype=float)
    b = np.array(method_b, dtype=float)
    diffs = b - a
    pooled_std = math.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
    if pooled_std == 0:
        return float("inf") if diffs.mean() != 0 else 0.0
    return float(diffs.mean() / pooled_std)


def bootstrap_confidence_interval(
    method_a: list[float],
    method_b: list[float],
    n_bootstrap: int = 10_000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    """
    Non-parametric bootstrap 95% CI for the mean difference (b - a).

    More reliable than t-test CI with n=5 seeds.

    Returns:
        (lower, upper) bounds of the (1-alpha) CI.
    """
    rng = np.random.default_rng(seed)
    a = np.array(method_a, dtype=float)
    b = np.array(method_b, dtype=float)
    n = len(a)
    diffs = b - a

    boot_means = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_means[i] = diffs[idx].mean()

    lo = float(np.percentile(boot_means, 100 * alpha / 2))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return lo, hi


def print_comparison_table(
    results: dict[str, list[float]],
    baseline_key: str,
    metric_name: str = "Class-IL (%)",
) -> None:
    """
    Pretty-print a comparison table with significance tests vs baseline.

    Args:
        results: {method_name: [seed1, seed2, ...]} mapping.
        baseline_key: Key in `results` to use as the reference method.
        metric_name: Display name of the metric.
    """
    baseline = results[baseline_key]
    header = f"{'Method':<25} {'Mean':>8} {'Std':>6} {'vs Baseline':>12} {'p-value':>9} {'d':>6} {'Sig':>4}"
    print(header)
    print("-" * len(header))

    for name, scores in results.items():
        mean = np.mean(scores)
        std = np.std(scores, ddof=1)
        if name == baseline_key:
            print(f"{name:<25} {mean:>7.2f}% {std:>5.2f}%  {'—':>12} {'—':>9} {'—':>6} {'—':>4}")
        else:
            r = paired_t_test(baseline, scores)
            sig = "✓" if r.is_significant else ""
            diff_str = f"{r.mean_diff:+.2f}%"
            print(
                f"{name:<25} {mean:>7.2f}% {std:>5.2f}%  {diff_str:>12} "
                f"{r.p_value:>9.4f} {r.cohens_d:>6.3f} {sig:>4}"
            )


if __name__ == "__main__":
    vanilla = [22.6, 22.87, 21.15, 23.12, 21.9]
    true_causality = [24.04, 25.09, 23.03, 21.73, 23.72]

    r = paired_t_test(vanilla, true_causality)
    print("=== Reproducing paper's primary statistical claim ===")
    print(f"n seeds    : {r.n_seeds}")
    print(f"Mean diff  : {r.mean_diff:+.3f}% (TRUE - vanilla)")
    print(f"Std diff   : {r.std_diff:.3f}%")
    print(f"t-statistic: {r.t_statistic:.4f}")
    print(f"p-value    : {r.p_value:.4f}  {'(significant at α=0.05)' if r.is_significant else '(not significant)'}")
    print(f"Cohen's d  : {r.cohens_d:.3f}")
    print(f"Bootstrap 95% CI: [{r.ci_95[0]:.3f}%, {r.ci_95[1]:.3f}%]")
    print()
    print_comparison_table(
        {"Vanilla DER++": vanilla, "TRUE Causality": true_causality},
        baseline_key="Vanilla DER++",
    )
