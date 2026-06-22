#!/usr/bin/env python3
"""
SymbioAI — TRUE Interventional Causality Experiment Runner
============================================================

Wraps Mammoth's utils/main.py with sensible defaults for causal experiments.
Must be run from the Mammoth root directory, with PYTHONPATH including symbioAI.

Setup (RunPod):
    cd /workspace/mammoth
    export PYTHONPATH=/workspace/symbioAI:/workspace/mammoth
    python /workspace/symbioAI/run_optimized_true_causality.py [options]

Modes (--use_causal_sampling):
    0  Vanilla DER++          ~43 min / seed
    1  Graph heuristic        ~43 min / seed
    2  Hybrid                 ~2 h / seed
    3  TRUE causality         ~3-5 h / seed  (optimised from ~13 h)
    4  Influence functions    ~2 h / seed (est.)

Examples:
    # Vanilla baseline, seed 1
    python run_optimized_true_causality.py --use_causal_sampling 0 --seed 1

    # TRUE causality, 5 epochs, seed 1, log to W&B
    python run_optimized_true_causality.py --use_causal_sampling 3 --seed 1 --wandb

    # 50-epoch run (literature standard)
    python run_optimized_true_causality.py --use_causal_sampling 3 --n_epochs 50 --seed 1
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def check_environment() -> None:
    try:
        import torch
        cuda = torch.cuda.is_available()
        device = torch.cuda.get_device_name(0) if cuda else "CPU"
        print(f"PyTorch {torch.__version__} | CUDA: {cuda} | Device: {device}")
    except ImportError:
        print("WARNING: PyTorch not installed")


def find_mammoth_main() -> Path:
    """Find Mammoth's utils/main.py relative to cwd."""
    candidates = [
        Path("utils/main.py"),           # run from mammoth root
        Path("mammoth/utils/main.py"),   # run from workspace
    ]
    for p in candidates:
        if p.exists():
            return p
    print(
        "ERROR: Cannot find utils/main.py.\n"
        "Run this script from the Mammoth root directory:\n"
        "  cd /workspace/mammoth && python /workspace/symbioAI/run_optimized_true_causality.py"
    )
    sys.exit(1)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run SymbioAI causal continual learning experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Core experiment ──────────────────────────────────────────────────────
    parser.add_argument("--use_causal_sampling", type=int, default=3,
                        help="0=vanilla, 1=heuristic, 2=hybrid, 3=TRUE, 4=influence (default: 3)")
    parser.add_argument("--n_epochs", type=int, default=5,
                        help="Epochs per task (default: 5; literature standard: 50)")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--buffer_size", type=int, default=500)

    # ── DER++ hyperparameters ────────────────────────────────────────────────
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.03)

    # ── Causal tuning ────────────────────────────────────────────────────────
    parser.add_argument("--true_micro_steps", type=int, default=1,
                        help="Gradient steps per intervention (default: 1)")
    parser.add_argument("--causal_hybrid_candidates", type=int, default=100,
                        help="Candidates evaluated per round (default: 100)")
    parser.add_argument("--causal_eval_interval", type=int, default=5,
                        help="Reuse causal ranking for N steps (default: 5)")

    # ── Output / logging ─────────────────────────────────────────────────────
    parser.add_argument("--output_dir", default="validation/results/runs")
    parser.add_argument("--output_name", default=None)
    parser.add_argument("--wandb", action="store_true",
                        help="Enable W&B logging (requires WANDB_API_KEY)")
    parser.add_argument("--debug", type=int, default=0)

    args = parser.parse_args()

    check_environment()
    mammoth_main = find_mammoth_main()

    mode_labels = {0: "vanilla", 1: "heuristic", 2: "hybrid", 3: "true", 4: "influence"}
    mode_name = mode_labels.get(args.use_causal_sampling, f"mode{args.use_causal_sampling}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_name = args.output_name or f"{mode_name}_seed{args.seed}_{args.n_epochs}ep.log"
    log_path = out_dir / log_name

    print(f"\nMode: {mode_name} (use_causal_sampling={args.use_causal_sampling})")
    print(f"Seed: {args.seed} | Epochs: {args.n_epochs} | Log: {log_path}\n")

    cmd = [
        sys.executable, str(mammoth_main),
        "--model", "derpp-causal",
        "--dataset", "seq-cifar100",
        "--buffer_size", str(args.buffer_size),
        "--alpha", str(args.alpha),
        "--beta", str(args.beta),
        "--n_epochs", str(args.n_epochs),
        "--batch_size", str(args.batch_size),
        "--lr", str(args.lr),
        "--lr_scheduler", "multisteplr",
        "--lr_milestones", "3", "4",
        "--use_causal_sampling", str(args.use_causal_sampling),
        "--temperature", "2.0",
        "--true_micro_steps", str(args.true_micro_steps),
        "--true_temp_lr", "0.05",
        "--causal_hybrid_candidates", str(args.causal_hybrid_candidates),
        "--causal_eval_interval", str(args.causal_eval_interval),
        "--causal_num_interventions", "50",
        "--causal_effect_threshold", "0.05",
        "--causal_blend_ratio", "0.3",
        "--causal_batch_size", "16",
        "--causal_warmup_tasks", "2",   # was 5; with 10 tasks, 5 is too conservative
        "--use_batched_causality", "1",
        "--debug", str(args.debug),
        "--seed", str(args.seed),
    ]

    if args.wandb:
        wandb_key = os.environ.get("WANDB_API_KEY", "")
        if not wandb_key:
            print("WARNING: --wandb passed but WANDB_API_KEY not set; logging may fail")
        cmd += ["--wandb", "True", "--wandb_project", "causal-continual-learning",
                "--wandb_name", f"{mode_name}_seed{args.seed}"]

    print("Command:")
    print("  " + " \\\n  ".join(cmd))
    print()

    with log_path.open("w") as fh:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        for line in proc.stdout:
            print(line, end="")
            fh.write(line)
            fh.flush()
        proc.wait()

    print(f"\nExit code: {proc.returncode}")
    print(f"Log saved: {log_path}")

    # Extract final result
    try:
        for line in reversed(log_path.read_text().splitlines()):
            if "Class-IL" in line and "%" in line:
                print(f"Result: {line.strip()}")
                break
    except Exception:
        pass

    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
