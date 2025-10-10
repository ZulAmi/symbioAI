# Symbio AI - Quick Start Guide (macOS)

## 🚀 Recommended Setup (5 minutes)

This is the **fastest and most reliable** way to run Symbio AI on macOS without build issues.

### Prerequisites

- macOS with Homebrew installed
- ~2GB free disk space

---

## Step 1: Install Python 3.11 via Homebrew

```bash
# Install Python 3.11 (pre-compiled, no build needed)
brew install python@3.11

# Verify installation
/opt/homebrew/bin/python3.11 --version
```

---

## Step 2: Create Virtual Environment

```bash
cd "/Users/zulhilmirahmat/Development/programming/Symbio AI"

# Create venv with Python 3.11
/opt/homebrew/bin/python3.11 -m venv .venv

# Activate it
source .venv/bin/activate

# Confirm you're using Python 3.11
python --version  # Should show Python 3.11.x
```

---

## Step 3: Install Core Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Create macOS-safe requirements (excludes GPU-only packages)
cat > requirements-core.txt << 'EOF'
# Core AI/ML (no GPU deps)
torch==2.1.2
transformers==4.36.2
numpy==1.26.2
pandas==2.1.4
scikit-learn==1.3.2

# Async framework
aiohttp==3.9.1
asyncio==3.4.3

# Configuration
pydantic==2.5.2
PyYAML==6.0.1

# Testing
pytest==7.4.3
pytest-asyncio==0.23.2

# Monitoring
psutil==5.9.6

# Formal verification (optional but recommended)
z3-solver==4.12.2
EOF

# Install core dependencies
pip install -r requirements-core.txt
```

**Why this works**: Avoids problematic packages like `psycopg2-binary` (requires PostgreSQL headers), `cupy-cuda` (GPU-only), and other macOS build issues.

---

## Step 4: Run Your First Demo 🎉

```bash
# Quick test - Automated Theorem Proving
python examples/theorem_proving_demo.py
```

**Expected output**: 7 comprehensive demos showing safety verification, lemma generation, proof repair, etc.

---

## Step 5: Run the Full System (Optional)

### Option A: Main System Orchestrator

```bash
python main.py
```

### Option B: API Gateway (Service Mesh)

```bash
python - << 'PY'
import asyncio
from api.gateway import ServiceMesh

config = {
    'gateway': {},
    'redis_url': 'redis://localhost:6379'  # Falls back to in-memory if Redis not running
}

async def run():
    mesh = ServiceMesh(config)
    await mesh.run(host='127.0.0.1', port=8080)

asyncio.run(run())
PY
```

Then visit: http://127.0.0.1:8080/admin/metrics

---

## More Demos

```bash
# Recursive self-improvement
python examples/recursive_self_improvement_demo.py

# Cross-task transfer learning
python examples/cross_task_transfer_demo.py

# Metacognitive monitoring + causal diagnosis
python examples/metacognitive_causal_demo.py

# Neural-symbolic architecture
python examples/neural_symbolic_demo.py

# Compositional concept learning
python examples/compositional_concept_demo.py
```

---

## Optional: Full Dependency Install

If you need **all** features (databases, cloud providers, etc.), install selectively:

```bash
# Install with macOS-safe filtering
grep -v -E '^(cupy-cuda|psycopg2-binary|asyncpg)' requirements.txt > requirements-filtered.txt
pip install -r requirements-filtered.txt

# If you need PostgreSQL support
brew install postgresql@16
pip install psycopg2-binary
```

---

## Troubleshooting

### Issue: `pip install` fails with "pg_config not found"

**Solution**: Skip PostgreSQL packages or install PostgreSQL first:

```bash
brew install postgresql@16
pip install psycopg2-binary
```

### Issue: `cupy-cuda` fails on macOS

**Solution**: Already excluded in requirements-core.txt (GPU package, not needed on macOS)

### Issue: Python 3.13 compatibility issues

**Solution**: Use Python 3.11 (most stable for these pinned dependencies)

### Issue: Redis connection errors

**Solution**: API Gateway falls back to in-memory mode. To use Redis:

```bash
brew install redis
brew services start redis
```

---

## Testing

```bash
# Run tests
pytest -q

# If no tests discovered, create pytest.ini:
cat > pytest.ini << 'EOF'
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
asyncio_mode = auto
EOF

# Try again
pytest -v
```

---

## Next Steps

1. ✅ Read the documentation: `docs/automated_theorem_proving.md`
2. ✅ Explore the API: `api/gateway.py`
3. ✅ Check completion reports: `AUTOMATED_THEOREM_PROVING_COMPLETE.md`
4. ✅ Review architecture: `docs/architecture.md`

---

## Summary: Why This Approach?

✅ **No build issues** - Pre-compiled Python 3.11 from Homebrew  
✅ **Fast setup** - Core deps install in ~2 minutes  
✅ **All demos work** - Tested with theorem proving, self-improvement, transfer learning  
✅ **Production ready** - Can add full deps later as needed  
✅ **macOS optimized** - Avoids GPU and PostgreSQL complications

---

**Time to first demo: ~5 minutes** 🚀
