#!/usr/bin/env bash
set -euo pipefail

# RunPod remote setup for SymbioAI via SSH
# Usage: ./scripts/runpod_remote_setup.sh <POD_ID> [KEY_PATH]
# Example:
#   ./scripts/runpod_remote_setup.sh xa01tgm2j389fj-64411a7b ~/.ssh/id_ed25519

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <POD_ID> [KEY_PATH]" >&2
  exit 1
fi

POD_ID="$1"
KEY_PATH="${2:-$HOME/.ssh/id_ed25519}"

if [[ ! -f "$KEY_PATH" ]]; then
  echo "[ERROR] SSH key not found at $KEY_PATH" >&2
  exit 2
fi

REMOTE_CMD='bash -lc '
REMOTE_CMD+=$'"\n'
REMOTE_CMD+=$'set -euo pipefail\n'
REMOTE_CMD+=$'echo \"[INFO] Using Python:\" && python3 -V || true\n'
REMOTE_CMD+=$'cd /workspace\n'
REMOTE_CMD+=$'if [ -d symbioAI ]; then\n'
REMOTE_CMD+=$'  echo \"[INFO] Repo exists — pulling latest...\"\n'
REMOTE_CMD+=$'  cd symbioAI && git pull --rebase\n'
REMOTE_CMD+=$'else\n'
REMOTE_CMD+=$'  echo \"[INFO] Cloning repo...\"\n'
REMOTE_CMD+=$'  git clone https://github.com/ZulAmi/symbioAI.git\n'
REMOTE_CMD+=$'  cd symbioAI\n'
REMOTE_CMD+=$'fi\n'
REMOTE_CMD+=$'python3 -m pip install --upgrade pip\n'
REMOTE_CMD+=$'python3 -m pip install -r requirements.txt\n'
REMOTE_CMD+=$'echo \"[INFO] Running verify script...\"\n'
REMOTE_CMD+=$'python3 verify_runpod_setup.py || true\n'
REMOTE_CMD+=$'echo \"[INFO] Running core benchmark sanity test...\"\n'
REMOTE_CMD+=$'python3 test_core_benchmark.py || true\n'
REMOTE_CMD+=$'if ! command -v screen >/dev/null 2>&1; then\n'
REMOTE_CMD+=$'  echo \"[INFO] Installing screen...\"\n'
REMOTE_CMD+=$'  apt-get update -y && apt-get install -y screen\n'
REMOTE_CMD+=$'fi\n'
REMOTE_CMD+=$'mkdir -p logs\n'
REMOTE_CMD+=$'echo \"[INFO] Starting benchmarks in detached screen...\"\n'
REMOTE_CMD+=$'screen -dmS benchmarks bash -lc \"cd /workspace/symbioAI && python3 run_benchmarks.py --mode full >> logs/runpod_benchmarks.log 2>&1\"\n'
REMOTE_CMD+=$'echo \"[INFO] Active screen sessions:\"\n'
REMOTE_CMD+=$'screen -ls || true\n'
REMOTE_CMD+=$'echo \"[INFO] Tail logs with:\\n  tail -f /workspace/symbioAI/logs/runpod_benchmarks.log\"\n'
REMOTE_CMD+=$'"'

# Force TTY (-tt) to avoid PTY issues when running remote commands
ssh -tt "${POD_ID}@ssh.runpod.io" -i "$KEY_PATH" -- "$REMOTE_CMD"