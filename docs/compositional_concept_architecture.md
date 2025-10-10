# Compositional Concept Learning - Architecture Overview

**Visual guide to the system architecture**

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                 COMPOSITIONAL CONCEPT LEARNER                       │
│                    (Main Orchestrator)                              │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  │
        ┌─────────────────────────┼─────────────────────────┐
        │                         │                         │
        ▼                         ▼                         ▼
┌───────────────┐         ┌──────────────┐        ┌─────────────────┐
│   PERCEPTION  │         │  KNOWLEDGE   │        │    REASONING    │
│               │         │              │        │                 │
│ • Slot        │         │ • Concepts   │        │ • Relations     │
│   Attention   │────────▶│ • Hierarchies│◀───────│ • Composition   │
│ • Object      │         │ • Library    │        │ • Abstract      │
│   Encoder     │         │              │        │   Reasoning     │
└───────────────┘         └──────────────┘        └─────────────────┘
        │                         │                         │
        │                         │                         │
        └─────────────────────────┼─────────────────────────┘
                                  │
                                  ▼
                        ┌─────────────────┐
                        │ DISENTANGLEMENT │
                        │                 │
                        │ • β-VAE         │
                        │ • Factor        │
                        │   Manipulation  │
                        └─────────────────┘
```

## Component Details

### 1. Perception Layer

```
┌────────────────────────────────────────────────┐
│            OBJECT-CENTRIC PERCEPTION           │
└────────────────────────────────────────────────┘
                      │
                      ▼
        ┌─────────────────────────────┐
        │   Perceptual Input          │
        │   (Image, Embedding, etc.)  │
        └─────────────────────────────┘
                      │
                      ▼
        ┌─────────────────────────────┐
        │    SlotAttentionModule      │
        │                             │
        │  ┌─────────────────────┐   │
        │  │ 1. Initialize Slots │   │
        │  └─────────────────────┘   │
        │            │                │
        │            ▼                │
        │  ┌─────────────────────┐   │
        │  │ 2. Attention Iters  │   │
        │  │    (K iterations)   │   │
        │  └─────────────────────┘   │
        │            │                │
        │            ▼                │
        │  ┌─────────────────────┐   │
        │  │ 3. GRU Update       │   │
        │  └─────────────────────┘   │
        └─────────────────────────────┘
                      │
                      ▼
        ┌─────────────────────────────┐
        │     Object Slots            │
        │  [slot_1, slot_2, ..., slot_K]│
        └─────────────────────────────┘
                      │
                      ▼
        ┌─────────────────────────────┐
        │  ObjectRepresentation       │
        │  • object_id                │
        │  • slots: List[Slot]        │
        │  • binding_type             │
        │  • features                 │
        │  • concept_assignments      │
        └─────────────────────────────┘
```

### 2. Knowledge Representation

```
┌────────────────────────────────────────────────┐
│            CONCEPT LIBRARY                     │
└────────────────────────────────────────────────┘
                      │
        ┌─────────────┼─────────────┐
        │             │             │
        ▼             ▼             ▼
┌──────────┐  ┌──────────┐  ┌──────────┐
│ PRIMITIVE│  │COMPOSITE │  │ ABSTRACT │
│          │  │          │  │          │
│ • red    │  │• red_    │  │• vehicle │
│ • circle │  │  circle  │  │• animal  │
│ • large  │  │• scene   │  │• shape   │
└──────────┘  └──────────┘  └──────────┘
                      │
                      ▼
        ┌─────────────────────────────┐
        │      Concept Structure      │
        │                             │
        │  concept_id: str            │
        │  name: str                  │
        │  concept_type: enum         │
        │  embedding: ndarray         │
        │  confidence: float          │
        │  examples: List             │
        │  parent_concepts: List[str] │
        │  child_concepts: List[str]  │
        │  relations: Dict            │
        │  abstraction_level: int     │
        └─────────────────────────────┘
```

### 3. Concept Hierarchy

```
                    ┌─────────────┐
                    │   scene     │  Level 2 (Abstract)
                    │ (abstract)  │
                    └─────────────┘
                          │
              ┌───────────┴───────────┐
              │                       │
        ┌──────────┐           ┌──────────┐
        │red_circle│           │blue_     │  Level 1 (Composite)
        │(composite│           │square    │
        └──────────┘           └──────────┘
              │                       │
        ┌─────┴─────┐           ┌─────┴─────┐
        │           │           │           │
    ┌───┴───┐   ┌───┴───┐   ┌───┴───┐   ┌───┴───┐
    │  red  │   │circle │   │ blue  │   │square │  Level 0 (Primitive)
    │(attr) │   │(obj)  │   │(attr) │   │(obj)  │
    └───────┘   └───────┘   └───────┘   └───────┘
```

### 4. Relation Discovery

```
┌────────────────────────────────────────────────┐
│            RELATION NETWORK                    │
└────────────────────────────────────────────────┘
                      │
                      ▼
        ┌─────────────────────────────┐
        │   Input: Concept Pairs      │
        │   (concept_i, concept_j)    │
        └─────────────────────────────┘
                      │
                      ▼
        ┌─────────────────────────────┐
        │   Pairwise Encoder          │
        │   concat(emb_i, emb_j)      │
        └─────────────────────────────┘
                      │
                      ▼
        ┌─────────────────────────────┐
        │   Multi-Head Attention      │
        │   Aggregate Evidence        │
        └─────────────────────────────┘
                      │
                      ▼
        ┌─────────────────────────────┐
        │   Relation Classifier       │
        │   • Type                    │
        │   • Strength                │
        │   • Confidence              │
        └─────────────────────────────┘
                      │
                      ▼
        ┌─────────────────────────────┐
        │   ConceptRelation           │
        │   • relation_id             │
        │   • relation_type           │
        │   • source_concept_id       │
        │   • target_concept_id       │
        │   • strength                │
        │   • confidence              │
        └─────────────────────────────┘
```

### 5. Composition Engine

```
┌────────────────────────────────────────────────┐
│         COMPOSITION FUNCTION                   │
└────────────────────────────────────────────────┘
                      │
                      ▼
        ┌─────────────────────────────┐
        │   Input: Component IDs      │
        │   [concept_1, concept_2,... │
        └─────────────────────────────┘
                      │
                      ▼
        ┌─────────────────────────────┐
        │   Composition Operation     │
        │                             │
        │   • attribute_binding       │
        │   • conjunction             │
        │   • disjunction             │
        │   • negation                │
        │   • abstraction             │
        └─────────────────────────────┘
                      │
                      ▼
        ┌─────────────────────────────┐
        │   Neural Composition        │
        │                             │
        │   1. Encode components      │
        │   2. Select operation       │
        │   3. Compute composite      │
        │   4. Generate embedding     │
        └─────────────────────────────┘
                      │
                      ▼
        ┌─────────────────────────────┐
        │   Composite Concept         │
        │   • concept_id (new)        │
        │   • components: [ids]       │
        │   • operation: str          │
        │   • abstraction_level: +1   │
        └─────────────────────────────┘
```

### 6. Disentanglement System

```
┌────────────────────────────────────────────────┐
│         DISENTANGLED VAE (β-VAE)               │
└────────────────────────────────────────────────┘
                      │
                      ▼
        ┌─────────────────────────────┐
        │   Encoder Network           │
        │   input → [μ, log_σ²]       │
        └─────────────────────────────┘
                      │
                      ▼
        ┌─────────────────────────────┐
        │   Reparameterization        │
        │   z = μ + σ * ε             │
        └─────────────────────────────┘
                      │
                      ▼
        ┌─────────────────────────────┐
        │   Latent Factors            │
        │   [z₀, z₁, ..., z_L]        │
        │                             │
        │   z₀: size                  │
        │   z₁: color                 │
        │   z₂: shape                 │
        │   z₃: position_x            │
        │   z₄: position_y            │
        │   z₅: rotation              │
        │   ...                       │
        └─────────────────────────────┘
                      │
                      ▼
        ┌─────────────────────────────┐
        │   Decoder Network           │
        │   z → reconstruction        │
        └─────────────────────────────┘
                      │
                      ▼
        ┌─────────────────────────────┐
        │   Loss Function             │
        │   L = recon + β * KL        │
        │   (β=4.0 for disentangle)   │
        └─────────────────────────────┘
```

### 7. Abstract Reasoning

```
┌────────────────────────────────────────────────┐
│         ABSTRACT REASONING ENGINE              │
└────────────────────────────────────────────────┘
                      │
                      ▼
        ┌─────────────────────────────┐
        │   Query Types               │
        │                             │
        │   • Commonality             │
        │   • Analogy                 │
        │   • Completion              │
        │   • Causal                  │
        └─────────────────────────────┘
                      │
                      ▼
        ┌─────────────────────────────┐
        │   Reasoning Process         │
        │                             │
        │   1. Parse query            │
        │   2. Retrieve concepts      │
        │   3. Apply reasoning type   │
        │   4. Generate trace         │
        │   5. Produce result         │
        └─────────────────────────────┘
                      │
                      ▼
        ┌─────────────────────────────┐
        │   Result + Trace            │
        │                             │
        │   • reasoning: str          │
        │   • confidence: float       │
        │   • steps: List[str]        │
        │   • evidence: Dict          │
        └─────────────────────────────┘
```

## Data Flow

### Learning Primitive Concept

```
Input Examples
      │
      ▼
┌──────────────┐
│  Aggregate   │
│  Features    │
└──────────────┘
      │
      ▼
┌──────────────┐
│  Compute     │
│  Embedding   │
└──────────────┘
      │
      ▼
┌──────────────┐
│  Calculate   │
│  Confidence  │
└──────────────┘
      │
      ▼
┌──────────────┐
│  Store in    │
│  Library     │
└──────────────┘
```

### Composing Concepts

```
Component IDs
      │
      ▼
┌──────────────┐
│  Retrieve    │
│  Concepts    │
└──────────────┘
      │
      ▼
┌──────────────┐
│  Composition │
│  Function    │
└──────────────┘
      │
      ▼
┌──────────────┐
│  Generate    │
│  Composite   │
│  Embedding   │
└──────────────┘
      │
      ▼
┌──────────────┐
│  Create      │
│  Concept     │
│  Structure   │
└──────────────┘
      │
      ▼
┌──────────────┐
│  Add to      │
│  Hierarchy   │
└──────────────┘
```

### Discovering Relations

```
Concept Pairs
      │
      ▼
┌──────────────┐
│  Pairwise    │
│  Encoding    │
└──────────────┘
      │
      ▼
┌──────────────┐
│  Relation    │
│  Network     │
└──────────────┘
      │
      ▼
┌──────────────┐
│  Classify    │
│  Relation    │
│  Type        │
└──────────────┘
      │
      ▼
┌──────────────┐
│  Compute     │
│  Strength &  │
│  Confidence  │
└──────────────┘
      │
      ▼
┌──────────────┐
│  Store       │
│  Relation    │
└──────────────┘
```

## Integration Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    SYMBIO AI ECOSYSTEM                      │
└─────────────────────────────────────────────────────────────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        │                      │                      │
        ▼                      ▼                      ▼
┌───────────────┐    ┌──────────────────┐   ┌────────────────┐
│Agent          │    │Cross-Task        │   │Neural-Symbolic │
│Orchestrator   │    │Transfer Engine   │   │Reasoner        │
└───────────────┘    └──────────────────┘   └────────────────┘
        │                      │                      │
        │                      │                      │
        └──────────────────────┼──────────────────────┘
                               │
                               ▼
                ┌──────────────────────────┐
                │ Compositional Concept    │
                │ Learning System          │
                │                          │
                │ • Object Perception      │
                │ • Concept Library        │
                │ • Relation Discovery     │
                │ • Hierarchies            │
                │ • Disentanglement        │
                │ • Abstract Reasoning     │
                └──────────────────────────┘
                               │
                               │
        ┌──────────────────────┼──────────────────────┐
        │                      │                      │
        ▼                      ▼                      ▼
┌───────────────┐    ┌──────────────────┐   ┌────────────────┐
│Metacognitive  │    │Causal Self-      │   │Recursive Self- │
│Monitoring     │    │Diagnosis         │   │Improvement     │
└───────────────┘    └──────────────────┘   └────────────────┘
```

## Processing Pipeline

### End-to-End Example: "What is this scene?"

```
Step 1: PERCEPTION
┌────────────────┐
│ Input: Image   │
│ of 3 objects   │
└────────────────┘
        │
        ▼
  Slot Attention
        │
        ▼
┌────────────────┐
│ 3 Object Slots │
└────────────────┘

Step 2: CONCEPT LEARNING
        │
        ▼
┌────────────────┐
│ Learn concepts:│
│ • red          │
│ • blue         │
│ • circle       │
│ • square       │
└────────────────┘

Step 3: COMPOSITION
        │
        ▼
┌────────────────┐
│ Compose:       │
│ • red_circle   │
│ • blue_square  │
└────────────────┘

Step 4: RELATIONS
        │
        ▼
┌────────────────┐
│ Discover:      │
│ • spatial_     │
│   proximity    │
│ • semantic_    │
│   similarity   │
└────────────────┘

Step 5: HIERARCHY
        │
        ▼
┌────────────────┐
│ Build tree:    │
│ scene          │
│ ├─red_circle   │
│ └─blue_square  │
└────────────────┘

Step 6: REASONING
        │
        ▼
┌────────────────┐
│ Explain:       │
│ "This scene    │
│  contains a    │
│  red circle    │
│  and a blue    │
│  square in     │
│  proximity"    │
└────────────────┘
```

## Performance Characteristics

### Complexity Analysis

| Component          | Time       | Space   |
| ------------------ | ---------- | ------- |
| Slot Attention     | O(K×I×N)   | O(K×D)  |
| Concept Learning   | O(M×D)     | O(C×D)  |
| Composition        | O(N×D)     | O(D)    |
| Relation Discovery | O(C²×D)    | O(C²×R) |
| Hierarchy          | O(C log C) | O(C)    |
| Disentanglement    | O(E×M×D)   | O(M×L)  |

**Legend**:

- K = slots, I = iterations, N = input size
- M = examples, D = embedding dim
- C = concepts, R = relation dim
- E = epochs, L = latent dim

### Scalability

```
Concepts: 1 ──────────────────────────────────── 1,000+
          │                                      │
Relations: 0 ───────────────────────────────── 10,000+
          │                                      │
Hierarchy: 0 levels ─────────────────────── 10+ levels
          │                                      │
Objects:   1 ─────────────────────────────── 10+ objects
          │                                      │
          Small                                  Large
```

## Configuration Examples

### High Quality (Slow)

```python
learner = CompositionalConceptLearner(
    num_slots=10,          # More objects
    slot_dim=128,          # More capacity
    num_iterations=5,      # Better binding
    relation_dim=256,      # Richer relations
    latent_dim=128,        # More factors
    beta=10.0              # Strong disentangle
)
```

### Balanced (Recommended)

```python
learner = CompositionalConceptLearner(
    num_slots=7,           # Default
    slot_dim=64,           # Default
    num_iterations=3,      # Default
    relation_dim=128,      # Default
    latent_dim=64,         # Default
    beta=4.0               # Default
)
```

### Fast (Quick Testing)

```python
learner = CompositionalConceptLearner(
    num_slots=5,           # Fewer objects
    slot_dim=32,           # Less capacity
    num_iterations=2,      # Faster binding
    relation_dim=64,       # Simpler relations
    latent_dim=32,         # Fewer factors
    beta=1.0               # Minimal disentangle
)
```

## Key Innovations

### 1. Slot Attention for Object Binding

```
Traditional: Dense features (entangled objects)
Symbio AI:   Slot-based (discrete object binding)

Benefit: Clean object decomposition
```

### 2. Neural-Symbolic Composition

```
Traditional: End-to-end neural (black box)
Symbio AI:   Neural operators + symbolic structure

Benefit: Interpretable composition rules
```

### 3. Hierarchical Organization

```
Traditional: Flat concept space
Symbio AI:   Multi-level hierarchy

Benefit: Human-understandable organization
```

### 4. Disentangled Factors

```
Traditional: Entangled representations
Symbio AI:   Independent, manipulatable factors

Benefit: Counterfactual reasoning
```

### 5. Compositional Generalization

```
Traditional: Learn each combination separately
Symbio AI:   Compose primitives → exponential coverage

Benefit: 10 primitives → 100+ composites
```

---

**See Also**:

- Full documentation: `docs/compositional_concept_learning.md`
- Quick reference: `docs/compositional_concept_quick_reference.md`
- Demo: `examples/compositional_concept_demo.py`
- Implementation: `training/compositional_concept_learning.py`

**Last Updated**: January 2025
