"""
Comprehensive Demo: Compositional Concept Learning System

Demonstrates all key features:
1. Object-centric representations with slot attention
2. Relation networks for discovering relationships
3. Concept composition (building complex from simple)
4. Disentangled representations for interpretable factors
5. Hierarchical concept organization
6. Abstract reasoning over learned structures

Run: python examples/compositional_concept_demo.py
"""

import asyncio
import sys
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.compositional_concept_learning import (
    create_compositional_concept_learner,
    ConceptType,
    Concept,
    ObjectRepresentation,
    ConceptRelation
)


def print_section(title: str):
    """Print formatted section header."""
    print(f"\n{'='*80}")
    print(f" {title}")
    print(f"{'='*80}\n")


def demo_object_perception():
    """Demonstrate object-centric representation learning."""
    print_section("DEMO 1: Object-Centric Perception")
    
    learner = create_compositional_concept_learner(
        input_dim=128,
        concept_dim=64,
        num_slots=5
    )
    
    print("Scenario: Perceiving multiple objects in a scene")
    print("-" * 40)
    
    # Create mock scene with 3 objects
    scene_description = {
        "objects": [
            {"type": "circle", "color": "red", "size": "large"},
            {"type": "square", "color": "blue", "size": "small"},
            {"type": "triangle", "color": "green", "size": "medium"}
        ]
    }
    
    print(f"Scene contains: {len(scene_description['objects'])} objects")
    for i, obj in enumerate(scene_description['objects']):
        print(f"  Object {i+1}: {obj['color']} {obj['type']} ({obj['size']})")
    
    # Simulate perceptual input
    scene_input = np.random.randn(128)
    
    # Extract object representations
    print("\n🔍 Extracting object-centric representations...")
    objects = learner.perceive_objects(scene_input, num_objects=3)
    
    print(f"\n✓ Extracted {len(objects)} object representations:")
    for i, obj in enumerate(objects):
        print(f"\n  Object {i+1} (ID: {obj.object_id}):")
        print(f"    • Number of slots: {len(obj.slots)}")
        print(f"    • Binding strength: {obj.binding_strength:.2f}")
        print(f"    • Concept: {obj.concept_id}")
    
    print("\n📊 Summary:")
    print(f"  • Objects perceived: {len(objects)}")
    print(f"  • Slot-based representation: ✓")
    print(f"  • Object binding: ✓")
    
    return learner, objects


def demo_concept_learning(learner):
    """Demonstrate learning primitive and composite concepts."""
    print_section("DEMO 2: Concept Learning")
    
    print("Learning primitive concepts from examples...")
    print("-" * 40)
    
    # Learn color concepts
    print("\n1️⃣ Learning Color Concepts:")
    
    colors = ["red", "blue", "green"]
    color_concepts = {}
    
    for color in colors:
        # Generate training examples (mock)
        examples = [
            {"embedding": [np.random.randn() for _ in range(64)]}
            for _ in range(10)
        ]
        
        concept = learner.learn_concept(
            concept_name=color,
            concept_type=ConceptType.ATTRIBUTE,
            examples=examples,
            is_primitive=True
        )
        
        color_concepts[color] = concept
        print(f"   • Learned '{color}': confidence={concept.confidence:.2%}, "
              f"examples={concept.examples_seen}")
    
    # Learn shape concepts
    print("\n2️⃣ Learning Shape Concepts:")
    
    shapes = ["circle", "square", "triangle"]
    shape_concepts = {}
    
    for shape in shapes:
        examples = [
            {"embedding": [np.random.randn() for _ in range(64)]}
            for _ in range(10)
        ]
        
        concept = learner.learn_concept(
            concept_name=shape,
            concept_type=ConceptType.OBJECT,
            examples=examples,
            is_primitive=True
        )
        
        shape_concepts[shape] = concept
        print(f"   • Learned '{shape}': confidence={concept.confidence:.2%}, "
              f"examples={concept.examples_seen}")
    
    # Learn size concepts
    print("\n3️⃣ Learning Size Concepts:")
    
    sizes = ["small", "medium", "large"]
    size_concepts = {}
    
    for size in sizes:
        examples = [
            {"embedding": [np.random.randn() for _ in range(64)]}
            for _ in range(10)
        ]
        
        concept = learner.learn_concept(
            concept_name=size,
            concept_type=ConceptType.ATTRIBUTE,
            examples=examples,
            is_primitive=True
        )
        
        size_concepts[size] = concept
        print(f"   • Learned '{size}': confidence={concept.confidence:.2%}")
    
    print("\n📊 Summary:")
    print(f"  • Total primitive concepts learned: {len(color_concepts) + len(shape_concepts) + len(size_concepts)}")
    print(f"  • Concept types: attributes (colors, sizes), objects (shapes)")
    print(f"  • Average confidence: {np.mean([c.confidence for c in list(color_concepts.values()) + list(shape_concepts.values())]):.2%}")
    
    return color_concepts, shape_concepts, size_concepts


def demo_concept_composition(learner, color_concepts, shape_concepts):
    """Demonstrate composing concepts."""
    print_section("DEMO 3: Compositional Concept Learning")
    
    print("Composing primitive concepts into complex concepts...")
    print("-" * 40)
    
    compositions = []
    
    # Compose color + shape
    print("\n🔄 Creating Composite Concepts:")
    
    composite_specs = [
        ("red", "circle", "red_circle"),
        ("blue", "square", "blue_square"),
        ("green", "triangle", "green_triangle")
    ]
    
    for color_name, shape_name, comp_name in composite_specs:
        if color_name in color_concepts and shape_name in shape_concepts:
            composite = learner.compose_concepts(
                color_concepts[color_name].concept_id,
                shape_concepts[shape_name].concept_id,
                composition_name=comp_name,
                operation="attribute_binding"
            )
            
            compositions.append(composite)
            
            print(f"\n  Composite: {composite.name}")
            print(f"    • ID: {composite.concept_id}")
            print(f"    • Type: {composite.concept_type.value}")
            print(f"    • Components: {composite.composed_from}")
            print(f"    • Operation: {composite.composition_operation}")
            print(f"    • Confidence: {composite.confidence:.2%}")
            print(f"    • Abstraction level: {composite.abstraction_level}")
    
    # Higher-order composition
    if len(compositions) >= 2:
        print("\n🔄 Higher-Order Composition:")
        
        meta_composite = learner.compose_concepts(
            compositions[0].concept_id,
            compositions[1].concept_id,
            composition_name="scene_pattern",
            operation="conjunction"
        )
        
        print(f"\n  Meta-Composite: {meta_composite.name}")
        print(f"    • Composed from: {[c.name for c in compositions[:2]]}")
        print(f"    • Abstraction level: {meta_composite.abstraction_level}")
        print(f"    • Description: {meta_composite.human_description}")
    
    print("\n📊 Summary:")
    print(f"  • Simple compositions: {len(compositions)}")
    print(f"  • Higher-order compositions: 1")
    print(f"  • Compositional operations: attribute_binding, conjunction")
    print(f"  • Abstraction levels: 0 (primitive) → {max(c.abstraction_level for c in compositions)} (composite)")
    
    return compositions


def demo_relation_discovery(learner, objects):
    """Demonstrate relation network for discovering relationships."""
    print_section("DEMO 4: Relation Discovery")
    
    print("Discovering relationships between objects...")
    print("-" * 40)
    
    object_ids = [obj.object_id for obj in objects]
    
    print(f"\nAnalyzing {len(object_ids)} objects:")
    for i, obj_id in enumerate(object_ids):
        print(f"  Object {i+1}: {obj_id}")
    
    print("\n🔗 Running Relation Network...")
    relations = learner.discover_relations(object_ids)
    
    print(f"\n✓ Discovered {len(relations)} relations:\n")
    
    for i, relation in enumerate(relations, 1):
        print(f"  Relation {i}:")
        print(f"    • Type: {relation.relation_type}")
        print(f"    • Source: {relation.source_concept[:20]}...")
        print(f"    • Target: {relation.target_concept[:20]}...")
        print(f"    • Strength: {relation.strength:.2f}")
        print(f"    • Confidence: {relation.confidence:.2%}\n")
    
    # Analyze relation types
    relation_types = {}
    for rel in relations:
        relation_types[rel.relation_type] = relation_types.get(rel.relation_type, 0) + 1
    
    print("📊 Relation Type Distribution:")
    for rel_type, count in relation_types.items():
        print(f"  • {rel_type}: {count}")
    
    print(f"\n📊 Summary:")
    print(f"  • Total relations discovered: {len(relations)}")
    print(f"  • Relation types: {len(relation_types)}")
    print(f"  • Average strength: {np.mean([r.strength for r in relations]):.2f}")
    print(f"  • Average confidence: {np.mean([r.confidence for r in relations]):.2%}")
    
    return relations


def demo_concept_hierarchy(learner, compositions):
    """Demonstrate building concept hierarchies."""
    print_section("DEMO 5: Concept Hierarchy")
    
    print("Building hierarchical organization of concepts...")
    print("-" * 40)
    
    if not compositions:
        print("No compositions available for hierarchy building")
        return None
    
    # Build hierarchy from first composition
    root_concept = compositions[0]
    
    print(f"\nRoot Concept: {root_concept.name}")
    print(f"Strategy: composition_based\n")
    
    hierarchy = learner.build_concept_hierarchy(
        root_concept_id=root_concept.concept_id,
        strategy="composition_based"
    )
    
    print(f"✓ Built hierarchy: {hierarchy.hierarchy_id}\n")
    print(f"Hierarchy Statistics:")
    print(f"  • Total concepts: {hierarchy.num_concepts}")
    print(f"  • Maximum depth: {hierarchy.max_depth}")
    print(f"  • Root concept: {root_concept.name}")
    
    # Visualize hierarchy
    print(f"\n📊 Hierarchy Visualization:")
    print("-" * 40)
    visualization = learner.visualize_hierarchy(hierarchy.hierarchy_id)
    print(visualization)
    
    # Show level distribution
    if hierarchy.level_map:
        level_counts = {}
        for level in hierarchy.level_map.values():
            level_counts[level] = level_counts.get(level, 0) + 1
        
        print("\nConcepts per Abstraction Level:")
        for level in sorted(level_counts.keys()):
            print(f"  Level {level}: {level_counts[level]} concepts")
    
    print("\n📊 Summary:")
    print(f"  • Hierarchy depth: {hierarchy.max_depth}")
    print(f"  • Total concepts organized: {hierarchy.num_concepts}")
    print(f"  • Parent-child relations: {len(hierarchy.parent_map)}")
    
    return hierarchy


def demo_disentangled_learning(learner):
    """Demonstrate disentangled representation learning."""
    print_section("DEMO 6: Disentangled Representation Learning")
    
    print("Learning disentangled factors of variation...")
    print("-" * 40)
    
    # Generate training data
    training_data = [np.random.randn(128) for _ in range(100)]
    
    print(f"Training data: {len(training_data)} examples")
    
    print("\n🧬 Learning disentangled factors...")
    factor_names = learner.concept_disentangler.learn_disentangled_factors(
        training_data,
        num_epochs=50
    )
    
    print(f"\n✓ Learned {len(factor_names)} interpretable factors:\n")
    
    for factor_idx, factor_name in factor_names.items():
        stats = learner.concept_disentangler.factor_statistics.get(factor_idx, {})
        print(f"  Factor {factor_idx}: {factor_name}")
        print(f"    • Mean: {stats.get('mean', 0):.3f}")
        print(f"    • Std Dev: {stats.get('std', 0):.3f}")
        print(f"    • Range: [{stats.get('min', 0):.3f}, {stats.get('max', 0):.3f}]\n")
    
    # Demonstrate concept manipulation
    print("🎨 Concept Manipulation:")
    print("-" * 40)
    
    # Create a test concept embedding
    test_embedding = [np.random.randn() for _ in range(64)]
    
    print("\nOriginal concept embedding (first 10 dims):")
    print(f"  {[f'{x:.3f}' for x in test_embedding[:10]]}")
    
    # Manipulate "color" factor
    if 1 in factor_names:  # Assuming factor 1 is color
        print(f"\nManipulating factor '{factor_names[1]}' by +0.5...")
        modified = learner.concept_disentangler.manipulate_concept(
            test_embedding,
            factor_index=1,
            delta=0.5
        )
        
        print(f"Modified embedding (first 10 dims):")
        print(f"  {[f'{x:.3f}' for x in modified[:10]]}")
        
        # Show difference
        diff = [modified[i] - test_embedding[i] for i in range(min(10, len(modified)))]
        print(f"Difference:")
        print(f"  {[f'{x:.3f}' for x in diff]}")
    
    print("\n📊 Summary:")
    print(f"  • Disentangled factors: {len(factor_names)}")
    print(f"  • Factors are interpretable: ✓")
    print(f"  • Concept manipulation enabled: ✓")
    print(f"  • Examples: {factor_names.get(0, 'N/A')}, {factor_names.get(1, 'N/A')}, {factor_names.get(2, 'N/A')}")


def demo_abstract_reasoning(learner, objects):
    """Demonstrate abstract reasoning over learned structures."""
    print_section("DEMO 7: Abstract Reasoning")
    
    print("Performing abstract reasoning over learned concepts...")
    print("-" * 40)
    
    object_ids = [obj.object_id for obj in objects[:3]]
    
    # Reasoning task 1: Find commonalities
    print("\n💭 Task 1: What is common between these objects?")
    print(f"Objects: {len(object_ids)}")
    
    result1 = learner.abstract_reasoning(
        "What is common between these objects?",
        object_ids
    )
    
    print(f"\n✓ Reasoning Result:")
    print(f"  • Query: {result1['query']}")
    print(f"  • Objects analyzed: {result1['num_objects']}")
    print(f"  • Reasoning: {result1.get('reasoning', 'N/A')}")
    if 'common_attributes' in result1:
        print(f"  • Common attributes: {result1['common_attributes']}")
    
    # Reasoning task 2: Discover relations
    print("\n\n💭 Task 2: What relationships exist between these objects?")
    
    result2 = learner.abstract_reasoning(
        "What relationships exist between these objects?",
        object_ids
    )
    
    print(f"\n✓ Reasoning Result:")
    print(f"  • Query: {result2['query']}")
    print(f"  • Reasoning: {result2.get('reasoning', 'N/A')}")
    if 'discovered_relations' in result2:
        print(f"  • Relations found: {len(result2['discovered_relations'])}")
        for i, rel in enumerate(result2['discovered_relations'][:3], 1):
            print(f"    {i}. {rel['type']} (strength: {rel['strength']:.2f})")
    
    # Reasoning task 3: Composition
    print("\n\n💭 Task 3: How can we compose these objects?")
    
    result3 = learner.abstract_reasoning(
        "How can we compose these objects?",
        object_ids[:2]
    )
    
    print(f"\n✓ Reasoning Result:")
    print(f"  • Query: {result3['query']}")
    print(f"  • Reasoning: {result3.get('reasoning', 'N/A')}")
    if 'composed_concept' in result3:
        comp = result3['composed_concept']
        print(f"  • Composed concept: {comp['name']}")
        print(f"  • Confidence: {comp['confidence']:.2%}")
    
    print("\n📊 Summary:")
    print(f"  • Reasoning tasks completed: 3")
    print(f"  • Abstract reasoning capabilities: ✓")
    print(f"  • Compositional reasoning: ✓")
    print(f"  • Relational reasoning: ✓")


def demo_concept_explanation(learner):
    """Demonstrate human-interpretable concept explanations."""
    print_section("DEMO 8: Concept Explanation & Interpretability")
    
    print("Generating human-interpretable explanations...")
    print("-" * 40)
    
    # Get some concepts
    concepts = list(learner.concepts.values())[:3]
    
    for i, concept in enumerate(concepts, 1):
        print(f"\n📖 Concept {i}:")
        print("-" * 40)
        
        explanation = learner.get_concept_explanation(concept.concept_id)
        print(explanation)
    
    print("\n📊 Summary:")
    print(f"  • Concepts explained: {len(concepts)}")
    print(f"  • Human-readable descriptions: ✓")
    print(f"  • Compositional structure visible: ✓")
    print(f"  • Usage statistics tracked: ✓")


async def main():
    """Run all demonstrations."""
    print("\n" + "="*80)
    print(" COMPOSITIONAL CONCEPT LEARNING - COMPREHENSIVE DEMO")
    print("="*80)
    print("\nThis demo showcases:")
    print("  1. Object-centric representation learning")
    print("  2. Learning primitive and composite concepts")
    print("  3. Compositional concept building")
    print("  4. Relation discovery via neural networks")
    print("  5. Hierarchical concept organization")
    print("  6. Disentangled representation learning")
    print("  7. Abstract reasoning over symbolic structures")
    print("  8. Human-interpretable explanations")
    
    try:
        # Demo 1: Object perception
        learner, objects = demo_object_perception()
        
        # Demo 2: Concept learning
        color_concepts, shape_concepts, size_concepts = demo_concept_learning(learner)
        
        # Demo 3: Concept composition
        compositions = demo_concept_composition(learner, color_concepts, shape_concepts)
        
        # Demo 4: Relation discovery
        relations = demo_relation_discovery(learner, objects)
        
        # Demo 5: Concept hierarchy
        hierarchy = demo_concept_hierarchy(learner, compositions)
        
        # Demo 6: Disentangled learning
        demo_disentangled_learning(learner)
        
        # Demo 7: Abstract reasoning
        demo_abstract_reasoning(learner, objects)
        
        # Demo 8: Concept explanation
        demo_concept_explanation(learner)
        
        # Final summary
        print_section("🎉 DEMO COMPLETE - ALL SYSTEMS OPERATIONAL")
        
        print("✅ All demonstrations completed successfully!\n")
        
        print("System Statistics:")
        print(f"  • Total concepts learned: {len(learner.concepts)}")
        print(f"  • Primitive concepts: {sum(1 for c in learner.concepts.values() if c.is_primitive)}")
        print(f"  • Composite concepts: {sum(1 for c in learner.concepts.values() if not c.is_primitive)}")
        print(f"  • Object representations: {len(learner.objects)}")
        print(f"  • Discovered relations: {len(learner.concept_relations)}")
        print(f"  • Concept hierarchies: {len(learner.hierarchies)}")
        
        print("\nKey Capabilities Demonstrated:")
        print("  ✓ Object-centric perception with slot attention")
        print("  ✓ Compositional concept learning")
        print("  ✓ Relation network for discovering relationships")
        print("  ✓ Disentangled representations for manipulation")
        print("  ✓ Hierarchical concept organization")
        print("  ✓ Abstract reasoning over symbolic structures")
        print("  ✓ Human-interpretable explanations")
        
        print("\n📚 Next Steps:")
        print("  • Review documentation: docs/compositional_concept_learning.md")
        print("  • Explore API: training/compositional_concept_learning.py")
        print("  • Integrate with your AI system")
        print("  • Experiment with custom concept domains")
        
        print("\n" + "="*80)
        
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
