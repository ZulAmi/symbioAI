"""
Automated ablation study runner.

Reads an ablation YAML config, sweeps the ablate.param over ablate.values,
runs each variant, and saves a summary CSV + visualisation.

Usage
-----
    # Ablate over causal_eval_interval
    python scripts/run_ablation.py configs/ablation/eval_interval.yaml

    # Ablate over micro_steps, 3 seeds
    python scripts/run_ablation.py configs/ablation/micro_steps.yaml --seeds 1 2 3

    # Dry-run: print commands without executing
    python scripts/run_ablation.py configs/ablation/candidates.yaml --dry_run
"""
from __future__ import annotations

import argparse
import csv
import logging
import re
import subprocess
import sys
import time
from pathlib import Path

from omegaconf import OmegaConf

logger = logging.getLogger(__name__)


# ─── Result parsing ────────────────────────────────────────────────────────────

def _parse_log(log_path: Path) -> dict[str, float]:
    """Extract final Class-IL and Task-IL accuracies from a Mammoth log file."""
    class_il, task_il = None, None
    class_re = re.compile(r"Class-IL.*?(\d+\.\d+)%", re.IGNORECASE)
    task_re = re.compile(r"Task-IL.*?(\d+\.\d+)%", re.IGNORECASE)

    if not log_path.exists():
        return {}

    for line in log_path.read_text().splitlines():
        m = class_re.search(line)
        if m:
            class_il = float(m.group(1))
        m = task_re.search(line)
        if m:
            task_il = float(m.group(1))

    result = {}
    if class_il is not None:
        result["class_il"] = class_il
    if task_il is not None:
        result["task_il"] = task_il
    return result


# ─── Core runner ───────────────────────────────────────────────────────────────

def build_command(cfg: dict, param: str, value, seed: int) -> list[str]:
    """Assemble the Mammoth launch command for one ablation variant."""
    cmd = [
        sys.executable, "run_optimized_true_causality.py",
        f"--{param}", str(value),
        "--seed", str(seed),
        "--n_epochs", str(cfg.get("n_epochs", 5)),
        "--buffer_size", str(cfg.get("buffer_size", 500)),
    ]
    # Forward any extra top-level keys that are not ablation/seeds/n_epochs
    skip = {"ablate", "seeds", "n_epochs", "buffer_size", "defaults"}
    for k, v in cfg.items():
        if k not in skip:
            cmd += [f"--{k}", str(v)]
    return cmd


def run_variant(
    cmd: list[str],
    log_path: Path,
    dry_run: bool = False,
    timeout_sec: int = 3600 * 14,
) -> dict[str, float]:
    """Run one ablation variant, stream output to log_path, return metrics."""
    log_path.parent.mkdir(parents=True, exist_ok=True)

    if dry_run:
        print("  DRY RUN:", " ".join(cmd))
        return {}

    logger.info("Running: %s", " ".join(cmd))
    t0 = time.time()

    with log_path.open("w") as fh:
        proc = subprocess.run(
            cmd,
            stdout=fh,
            stderr=subprocess.STDOUT,
            timeout=timeout_sec,
            check=False,
        )

    elapsed = (time.time() - t0) / 60
    if proc.returncode != 0:
        logger.warning("Command exited with code %d (see %s)", proc.returncode, log_path)
    else:
        logger.info("Done in %.1f min → %s", elapsed, log_path)

    return _parse_log(log_path)


# ─── Reporting ─────────────────────────────────────────────────────────────────

def save_csv(rows: list[dict], out_path: Path) -> None:
    if not rows:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    logger.info("Results saved → %s", out_path)


def print_summary(param: str, rows: list[dict]) -> None:
    from collections import defaultdict
    import numpy as np

    by_value: dict = defaultdict(list)
    for r in rows:
        by_value[r[param]].append(r.get("class_il", float("nan")))

    header = f"{'Value':>12}  {'Mean Class-IL':>14}  {'Std':>6}  n"
    print(f"\nAblation: {param}")
    print(header)
    print("-" * len(header))
    for val in sorted(by_value):
        scores = [s for s in by_value[val] if not (s != s)]  # drop NaN
        if scores:
            print(f"{str(val):>12}  {np.mean(scores):>13.2f}%  {np.std(scores, ddof=1) if len(scores) > 1 else 0:>5.2f}%  {len(scores)}")
        else:
            print(f"{str(val):>12}  {'n/a':>13}  {'n/a':>5}  0")


def plot_ablation(param: str, rows: list[dict], out_path: Path) -> None:
    """Save a simple bar/line chart of Class-IL vs ablation parameter value."""
    try:
        import numpy as np
        import matplotlib.pyplot as plt
        from collections import defaultdict

        by_value: dict = defaultdict(list)
        for r in rows:
            by_value[r[param]].append(r.get("class_il", float("nan")))

        xs = sorted(by_value)
        means = [float(np.nanmean(by_value[x])) for x in xs]
        stds = [float(np.nanstd(by_value[x], ddof=1)) if len(by_value[x]) > 1 else 0.0 for x in xs]

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.errorbar([str(x) for x in xs], means, yerr=stds, marker="o", capsize=5)
        ax.set_xlabel(param)
        ax.set_ylabel("Class-IL (%)")
        ax.set_title(f"Ablation: {param}")
        ax.grid(alpha=0.3)
        fig.tight_layout()

        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Plot saved → %s", out_path)
    except ImportError:
        logger.warning("matplotlib not available — skipping plot")


# ─── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    ap = argparse.ArgumentParser(description="Run an ablation sweep from a YAML config.")
    ap.add_argument("config", help="Path to ablation YAML (e.g. configs/ablation/micro_steps.yaml)")
    ap.add_argument("--seeds", nargs="+", type=int, help="Override seeds from config")
    ap.add_argument("--dry_run", action="store_true", help="Print commands without running")
    ap.add_argument("--output_dir", default="runs/ablation", help="Where to write logs and CSV")
    args = ap.parse_args()

    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    ablate = cfg.get("ablate", {})
    param = ablate.get("param")
    values = ablate.get("values", [])
    seeds = args.seeds or cfg.get("seeds", [1])

    if not param or not values:
        ap.error("Config must have ablate.param and ablate.values")

    out_dir = Path(args.output_dir) / param
    rows: list[dict] = []

    for val in values:
        for seed in seeds:
            tag = f"{param}_{val}_seed{seed}"
            log_path = out_dir / f"{tag}.log"

            cmd = build_command(cfg, param, val, seed)
            metrics = run_variant(cmd, log_path, dry_run=args.dry_run)

            row = {param: val, "seed": seed, **metrics}
            rows.append(row)
            if metrics:
                logger.info("  %s: Class-IL=%.2f%%", tag, metrics.get("class_il", float("nan")))

    if not args.dry_run:
        save_csv(rows, out_dir / "results.csv")
        print_summary(param, rows)
        plot_ablation(param, rows, out_dir / "ablation_plot.pdf")


if __name__ == "__main__":
    main()
