L# Automated Theorem Proving - Quick Reference

A concise guide to using the theorem proving integration.

---

## Install (optional but recommended)

- Z3: `pip install z3-solver`
- Lean: https://lean-lang.org/lean4/doc/setup.html
- Coq: https://coq.inria.fr/download

System works without them (mock mode), but real proofs need install.

---

## Create a prover

```python
from training.automated_theorem_proving import create_theorem_prover
prover = create_theorem_prover(timeout_seconds=30, enable_proof_repair=True)
```

Backend options:

- default_backend: "AUTO" | "Z3" | "LEAN" | "COQ"

---

## Verify a property

```python
from training.automated_theorem_proving import FormalProperty, PropertyType

prop = FormalProperty(
  property_id="safety_bounds",
  property_type=PropertyType.SAFETY,
  name="Bounds safety",
  formal_statement="x >= 0 and x < MAX",
  preconditions=["MAX > 0"],
)

res = prover.verify_property(prop, {"x": 3, "MAX": 5})
print(res.status, res.mathematical_guarantee)
```

Status values: VERIFIED | DISPROVED | INCONCLUSIVE | ERROR

---

## Safety helper

```python
res = prover.verify_safety_property(
  name="No collision",
  preconditions=["distance > 0", "speed >= 0"],
  bad_states=["distance <= SAFE_DISTANCE"],
  context={"distance": 50, "SAFE_DISTANCE": 10, "speed": 15},
)
print(res.status)
```

---

## Lemma generation

```python
lemma = prover.generate_lemma("transitivity", {"R": "<="})
print(lemma.name, lemma.statement)
```

Patterns: monotonicity, transitivity, commutativity, associativity, distributivity

---

## Proof repair

```python
res = prover.verify_property(prop, ctx)
if not res.mathematical_guarantee:
  # repair runs automatically if enabled
  print(res.status, res.total_attempts)
```

---

## Stats

```python
stats = prover.get_statistics()
print(stats)
```

---

## Demos

Run: `python examples/theorem_proving_demo.py`

Demos included:

- Safety
- Correctness
- Lemmas
- Repair
- Multi-prover
- Safety-critical system
- Stats

---

End of Quick Reference
