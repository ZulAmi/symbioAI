# ✅ Symbio AI - Ready to Run!

## 🎉 Setup Complete

Your Symbio AI system is **fully configured** and ready to use.

### What's Installed

- ✅ Python 3.11.14 (Homebrew)
- ✅ Virtual environment (`.venv`)
- ✅ Core AI/ML dependencies (PyTorch, Transformers, etc.)
- ✅ Z3 Theorem Prover
- ✅ All demos tested and working

---

## 🚀 Quick Commands

### Run Demos

```bash
# Activate venv first (do this in every new terminal)
source .venv/bin/activate

# Theorem Proving (5 min)
python examples/theorem_proving_demo.py

# All demos
python quickstart.py all

# System health check
python quickstart.py health
```

### Individual Demos

```bash
source .venv/bin/activate

# Fast demos (~1-2 minutes each)
python examples/theorem_proving_demo.py
python examples/recursive_self_improvement_demo.py
python examples/cross_task_transfer_demo.py

# Longer demos (~3-5 minutes each)
python examples/metacognitive_causal_demo.py
python examples/neural_symbolic_demo.py
python examples/compositional_concept_demo.py
```

### Run Full System

```bash
source .venv/bin/activate

# Main orchestrator
python main.py

# API Gateway (in separate terminal)
python quickstart.py api
# Then visit: http://127.0.0.1:8080/admin/metrics
```

---

## 📊 What Just Happened

1. **Installed Python 3.11** via Homebrew (pre-compiled, no build issues)
2. **Created virtual environment** with Python 3.11
3. **Installed 40+ packages** including:
   - PyTorch 2.1.2 (CPU)
   - Transformers 4.36.2
   - Z3 Solver 4.12.2
   - Pydantic, PyYAML, pytest, and more
4. **Verified system** with health check - all green ✅

---

## 🎯 What Can You Do Now?

### 1. Run Automated Theorem Proving

```bash
source .venv/bin/activate
python examples/theorem_proving_demo.py
```

**Output**: 7 comprehensive demos showing:

- Safety property verification
- Correctness verification
- Automatic lemma generation
- Proof repair
- Multi-prover integration
- Safety-critical system verification
- Statistics and analytics

### 2. Explore Other Advanced Features

```bash
# Recursive self-improvement (meta-evolution)
python examples/recursive_self_improvement_demo.py

# Cross-task transfer learning (automatic curriculum)
python examples/cross_task_transfer_demo.py

# Self-aware AI with causal diagnosis
python examples/metacognitive_causal_demo.py
```

### 3. Read Documentation

- `QUICKSTART.md` - Complete setup guide (you're here!)
- `docs/automated_theorem_proving.md` - Technical guide
- `docs/theorem_proving_quick_reference.md` - Quick API reference
- `AUTOMATED_THEOREM_PROVING_COMPLETE.md` - Full completion report

### 4. Run Tests

```bash
source .venv/bin/activate
pytest -v tests/
```

---

## 💡 Pro Tips

### Always Activate the Virtual Environment

```bash
source .venv/bin/activate
```

Do this **every time** you open a new terminal window.

### Install Additional Packages

```bash
source .venv/bin/activate
pip install <package-name>
```

### Full Dependency Install (Optional)

If you need database support, cloud providers, etc.:

```bash
source .venv/bin/activate

# Install selectively (avoid problematic packages)
grep -v -E '^(cupy-cuda|psycopg2-binary|asyncpg)' requirements.txt > requirements-filtered.txt
pip install -r requirements-filtered.txt

# Or install specific feature sets
pip install sqlalchemy alembic  # Database
pip install boto3 azure-storage-blob  # Cloud
pip install mlflow wandb  # Experiment tracking
```

---

## 🔧 Troubleshooting

### "Command not found: python"

```bash
# Always activate venv first
source .venv/bin/activate
```

### "ModuleNotFoundError"

```bash
# Make sure you're in the venv and package is installed
source .venv/bin/activate
pip install <missing-package>
```

### Start Fresh

```bash
# Remove and recreate venv
rm -rf .venv
/opt/homebrew/opt/python@3.11/bin/python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements-core.txt
```

---

## 📚 Documentation Structure

```
Symbio AI/
├── QUICKSTART.md                       ← Setup guide (this file)
├── README.md                           ← Project overview
├── requirements-core.txt               ← Core dependencies (installed ✅)
├── requirements.txt                    ← Full dependencies
│
├── docs/
│   ├── automated_theorem_proving.md    ← Theorem proving technical guide
│   ├── theorem_proving_quick_reference.md
│   ├── architecture.md
│   └── ...
│
├── examples/                           ← All demos here
│   ├── theorem_proving_demo.py         ← Start here!
│   ├── recursive_self_improvement_demo.py
│   ├── cross_task_transfer_demo.py
│   └── ...
│
└── training/                           ← Core AI modules
    ├── automated_theorem_proving.py    ← Formal verification engine
    ├── recursive_self_improvement.py
    ├── cross_task_transfer.py
    └── ...
```

---

## 🎓 Next Steps

1. ✅ **Run your first demo** (theorem proving)
2. ✅ **Read the technical docs** (`docs/automated_theorem_proving.md`)
3. ✅ **Explore other demos** (recursive improvement, transfer learning)
4. ✅ **Run the full system** (`python main.py`)
5. ✅ **Check out the API** (`python quickstart.py api`)

---

## 📞 Quick Reference Card

```bash
# === Always do this first ===
source .venv/bin/activate

# === Common Commands ===
python quickstart.py health          # Check system
python quickstart.py all             # Run all demos
python examples/theorem_proving_demo.py  # Single demo
python main.py                       # Full orchestrator
pytest -v                            # Run tests

# === Documentation ===
open docs/automated_theorem_proving.md
open QUICKSTART.md
```

---

## ⏱️ Setup Time

**Total time**: ~5 minutes

- Homebrew Python install: ~2 min
- Dependency install: ~2 min
- Verification: ~1 min

**First demo run**: ~30 seconds

---

**You're all set! 🚀**

Start with: `source .venv/bin/activate && python examples/theorem_proving_demo.py`
