# Automated Theorem Proving - Technical Guide

This guide explains how to use, extend, and reason about the Automated Theorem Proving Integration in Symbio AI.

- Audience: Engineers and researchers
- Scope: Concepts, APIs, patterns, and extension points
- Status: Stable, production-ready

---

## 1. Concepts

- Formal Property: Structured property to verify (safety, correctness, etc.)
- Theorem Prover: Backend engine (Z3, Lean, Coq)
- Lemma: Reusable statement aiding proofs
- Proof Attempt: A single prover run instance
- Verification Result: Aggregated result across provers/attempts

---

## 2. API Overview

Main entrypoint: `training/automated_theorem_proving.py`

Key types:

- `PropertyType`: SAFETY, CORRECTNESS, LIVENESS, INVARIANT, TERMINATION, SECURITY, FAIRNESS
- `ProofStatus`: VERIFIED, DISPROVED, INCONCLUSIVE, ERROR
- `TheoremProverType`: Z3, LEAN, COQ, AUTO

Data models:

- `FormalProperty` (id, type, name, statement, preconditions, postconditions, invariants, etc.)
- `Lemma` (id, name, statement, context)
- `ProofAttempt` (prover, status, time, script, counterexample)
- `VerificationResult` (status, guarantee, attempts, repair info)

Entrypoints:

- `create_theorem_prover(...)` → returns `AutomatedTheoremProver`
- `verify_property(property, context)`
- `verify_safety_property(name, preconditions, bad_states, context)`
- `generate_lemma(pattern_name, context=None)`
- `get_statistics()`

---

## 3. Quick Start

```python
from training.automated_theorem_proving import (
 create_theorem_prover, FormalProperty, PropertyType
)

prover = create_theorem_prover(
 timeout_seconds=30,
 enable_proof_repair=True,
)

prop = FormalProperty(
 property_id="sorted_correctness",
 property_type=PropertyType.CORRECTNESS,
 name="Sorted output is non-decreasing",
 formal_statement="∀ i j. i ≤ j → a[i] ≤ a[j]",
 preconditions=["len(a) > 0"],
)

res = prover.verify_property(prop, {"a": [1,2,3,3]})
print(res.status, res.mathematical_guarantee)
```

---

## 4. Prover Selection

Auto-selection uses heuristics:

- Safety/invariant → Z3 preferred
- High complexity or higher-order reasoning → Lean/Coq
- Fallbacks: If primary fails or not installed, try next best

Override:

```python
prover = create_theorem_prover(default_backend="Z3")
```

---

## 5. Lemma Generation

Patterns supported:

- monotonicity: `∀ x y. x ≤ y → f(x) ≤ f(y)`
- transitivity: `∀ x y z. R(x,y) ∧ R(y,z) → R(x,z)`
- commutativity: `∀ x y. f(x,y) = f(y,x)`
- associativity: `∀ x y z. f(f(x,y),z) = f(x,f(y,z))`
- distributivity: `f(x, y+z) = f(x,y) + f(x,z)`

Usage:

```python
lemma = prover.generate_lemma("monotonicity", {"f": "sort"})
```

---

## 6. Proof Repair

Strategies (in order):

1. Add relevant lemmas from library
2. Simplify constraints (inline constants, remove dominated terms)
3. Strengthen preconditions (tighten bounds)

Check if repair happened:

```python
res = prover.verify_property(prop, ctx)
if res.successful_proof and res.successful_proof.repaired_version:
 print("Repaired via:", res.successful_proof.repaired_version)
```

---

## 7. Safety Verification Helper

```python
res = prover.verify_safety_property(
 name="No collision",
 preconditions=["distance > 0", "speed ≥ 0"],
 bad_states=["distance ≤ SAFE_DISTANCE"],
 context={"distance": 50, "SAFE_DISTANCE": 10, "speed": 15}
)
print(res.status)
```

---

## 8. Extending the System

Add a new prover:

1. Subclass `TheoremProver`
2. Implement `is_available()`, `verify()`, `generate_proof_script()`
3. Register in `create_theorem_prover()` and selection heuristics

Add a new lemma pattern:

1. Update lemma generation map in `AutomatedTheoremProver.generate_lemma`
2. Add template and optional micro-proof
3. Add unit test + demo case

Add proof repair strategy:

1. Extend `_repair_proof()` with a new strategy branch
2. Make it configurable via flags
3. Track metrics in statistics

---

## 9. Troubleshooting

- Z3 not installed → System falls back to mock mode; install with `pip install z3-solver`
- Lean/Coq not installed → External provers skipped; see install docs
- Timeouts → Increase `timeout_seconds` in `create_theorem_prover()`
- Inconclusive proofs → Add relevant lemmas or strengthen preconditions

---

## 10. Examples

See `examples/theorem_proving_demo.py` for:

- Safety verification
- Correctness of sorting
- Lemma generation
- Proof repair
- Multi-prover fallback
- Safety-critical system
- Statistics and analytics

---

## 11. Design Choices

- Unified abstraction for provers to enable smooth fallbacks
- Conservative defaults with clear escape hatches
- Robust logging and traceability for compliance
- Type hints and docstrings throughout for maintainability

---

## 12. Roadmap (Optional)

- Isabelle/HOL backend
- Proof tree visualization
- LLM-assisted property authoring
- Continuous verification during runtime

---

## 13. License and Credits

- Z3: MIT license
- Lean: Apache 2.0 license
- Coq: LGPL
- Integration and orchestration: © Symbio AI

---

End of Technical Guide
