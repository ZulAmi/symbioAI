"""
Causal Self-Diagnosis System - Symbio AI

Provides causal reasoning capabilities for AI systems to diagnose their own
failures, identify root causes, and plan interventions using counterfactual analysis.

This system enables AI to:
- Perform causal inference on its own behavior
- Identify root causes of failures
- Reason about counterfactuals ("what if?")
- Plan targeted interventions
- Learn causal relationships over time
"""

import asyncio
import logging
import random
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Set, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime
from collections import defaultdict, deque
from enum import Enum

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Mock classes
    class nn:
        class Module:
            pass
    class GCNConv:
        def __init__(self, *args, **kwargs): pass

from deployment.observability import OBSERVABILITY


class CausalNodeType(Enum):
    """Types of nodes in causal graph."""
    INPUT = "input"  # Input features
    HIDDEN = "hidden"  # Hidden representations
    OUTPUT = "output"  # Output predictions
    PARAMETER = "parameter"  # Model parameters
    HYPERPARAMETER = "hyperparameter"  # Training hyperparameters
    ENVIRONMENT = "environment"  # External factors


class FailureMode(Enum):
    """Types of failures that can occur."""
    ACCURACY_DROP = "accuracy_drop"
    HALLUCINATION = "hallucination"
    BIAS = "bias"
    INSTABILITY = "instability"
    OVERFITTING = "overfitting"
    UNDERFITTING = "underfitting"
    CATASTROPHIC_FORGETTING = "catastrophic_forgetting"
    ADVERSARIAL_VULNERABILITY = "adversarial_vulnerability"
    DISTRIBUTION_SHIFT = "distribution_shift"


class InterventionStrategy(Enum):
    """Types of intervention strategies."""
    RETRAIN = "retrain"
    FINE_TUNE = "fine_tune"
    ADJUST_HYPERPARAMETERS = "adjust_hyperparameters"
    ADD_REGULARIZATION = "add_regularization"
    COLLECT_MORE_DATA = "collect_more_data"
    CHANGE_ARCHITECTURE = "change_architecture"
    APPLY_PATCH = "apply_patch"
    RESET_COMPONENT = "reset_component"


@dataclass
class CausalNode:
    """Node in causal graph."""
    node_id: str
    node_type: CausalNodeType
    name: str
    
    # Value tracking
    current_value: Any = None
    expected_value: Any = None
    deviation: float = 0.0
    
    # Causal properties
    is_root_cause: bool = False
    causal_strength: float = 0.0  # How much this node affects outcomes
    
    # Relationships
    parents: List[str] = field(default_factory=list)  # What causes this
    children: List[str] = field(default_factory=list)  # What this causes
    
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CausalEdge:
    """Edge in causal graph representing causal relationship."""
    source: str  # Source node ID
    target: str  # Target node ID
    
    # Causal strength
    causal_effect: float = 0.0  # Direct causal effect (-1 to 1)
    confidence: float = 0.0  # Confidence in this relationship
    
    # Evidence
    observational_evidence: float = 0.0  # Correlation strength
    interventional_evidence: float = 0.0  # From interventions
    counterfactual_evidence: float = 0.0  # From counterfactuals
    
    # Type of relationship
    relationship_type: str = "causes"  # "causes", "prevents", "moderates"
    
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FailureDiagnosis:
    """Complete diagnosis of a failure."""
    diagnosis_id: str
    failure_mode: FailureMode
    
    # Causal analysis
    root_causes: List[str] = field(default_factory=list)  # Node IDs
    contributing_factors: List[str] = field(default_factory=list)
    causal_path: List[str] = field(default_factory=list)  # Path from root to failure
    
    # Confidence
    diagnosis_confidence: float = 0.0
    
    # Evidence
    supporting_evidence: List[Dict[str, Any]] = field(default_factory=list)
    contradicting_evidence: List[Dict[str, Any]] = field(default_factory=list)
    
    # Recommendations
    recommended_interventions: List[Tuple[InterventionStrategy, float]] = field(default_factory=list)
    
    # Counterfactuals
    counterfactuals: List['Counterfactual'] = field(default_factory=list)
    
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class Counterfactual:
    """Counterfactual analysis: "What if X was different?"""
    counterfactual_id: str
    
    # The change
    changed_node: str
    original_value: Any
    counterfactual_value: Any
    
    # Predicted outcome
    predicted_outcome: Any
    outcome_change: float  # How much outcome would change
    
    # Plausibility
    plausibility: float = 0.0  # How realistic this counterfactual is
    
    # Actionability
    is_actionable: bool = False  # Can we actually make this change?
    intervention_required: Optional[InterventionStrategy] = None
    
    description: str = ""
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class InterventionPlan:
    """Plan for intervening to fix a failure."""
    plan_id: str
    target_failure: FailureMode
    
    # Interventions
    interventions: List[Tuple[InterventionStrategy, Dict[str, Any]]] = field(default_factory=list)
    
    # Expected outcomes
    expected_improvement: float = 0.0
    confidence: float = 0.0
    estimated_cost: float = 0.0  # Computational/time cost
    
    # Risk assessment
    risks: List[str] = field(default_factory=list)
    side_effects: List[str] = field(default_factory=list)
    
    # Validation
    validation_metrics: List[str] = field(default_factory=list)
    
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class CausalGraph:
    """
    Causal graph representing causal relationships in the AI system.
    Uses graph neural networks to learn and reason about causality.
    """
    
    def __init__(self, learning_rate: float = 0.001, num_variables: int = 5):
        self.nodes: Dict[str, CausalNode] = {}
        self.edges: Dict[Tuple[str, str], CausalEdge] = {}
        
        self.learning_rate = learning_rate
        self.num_variables = num_variables
        
        # Initialize nodes for variables
        for i in range(num_variables):
            self.add_node(
                node_id=f"var_{i}",
                node_type=CausalNodeType.INPUT,
                name=f"Variable {i}",
                current_value=0.0
            )
        
        # Adjacency information
        self.adjacency: Dict[str, Set[str]] = defaultdict(set)
        
        self.logger = logging.getLogger(__name__)
    
    def add_node(
        self,
        node_id: str,
        node_type: CausalNodeType,
        name: str,
        **kwargs
    ) -> CausalNode:
        """Add a node to the causal graph."""
        node = CausalNode(
            node_id=node_id,
            node_type=node_type,
            name=name,
            **kwargs
        )
        self.nodes[node_id] = node
        return node
    
    def add_edge(
        self,
        source: str,
        target: str,
        causal_effect: float = 0.0,
        **kwargs
    ) -> CausalEdge:
        """Add a causal edge between nodes."""
        if source not in self.nodes or target not in self.nodes:
            raise ValueError(f"Both nodes must exist in graph")
        
        edge = CausalEdge(
            source=source,
            target=target,
            causal_effect=causal_effect,
            **kwargs
        )
        
        self.edges[(source, target)] = edge
        
        # Update adjacency
        self.adjacency[source].add(target)
        
        # Update node relationships
        self.nodes[source].children.append(target)
        self.nodes[target].parents.append(source)
        
        return edge
    
    def find_causal_path(
        self,
        source: str,
        target: str
    ) -> Optional[List[str]]:
        """Find causal path from source to target using BFS."""
        if source not in self.nodes or target not in self.nodes:
            return None
        
        # BFS
        queue = deque([(source, [source])])
        visited = {source}
        
        while queue:
            current, path = queue.popleft()
            
            if current == target:
                return path
            
            for neighbor in self.adjacency.get(current, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        
        return None
    
    def compute_causal_strength(
        self,
        node_id: str,
        target_outcome: str
    ) -> float:
        """
        Compute how much a node causally affects a target outcome.
        Uses path analysis and edge strengths.
        """
        if node_id not in self.nodes or target_outcome not in self.nodes:
            return 0.0
        
        # Find all paths
        paths = self._find_all_paths(node_id, target_outcome)
        
        if not paths:
            return 0.0
        
        # Compute total causal strength across all paths
        total_strength = 0.0
        for path in paths:
            # Compute path strength (product of edge effects)
            path_strength = 1.0
            for i in range(len(path) - 1):
                edge = self.edges.get((path[i], path[i+1]))
                if edge:
                    path_strength *= edge.causal_effect
            
            total_strength += path_strength
        
        return total_strength
    
    def _find_all_paths(
        self,
        source: str,
        target: str,
        max_length: int = 10
    ) -> List[List[str]]:
        """Find all paths from source to target."""
        if source == target:
            return [[source]]
        
        paths = []
        
        def dfs(current: str, path: List[str], visited: Set[str]):
            if len(path) > max_length:
                return
            
            if current == target:
                paths.append(path.copy())
                return
            
            for neighbor in self.adjacency.get(current, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    path.append(neighbor)
                    dfs(neighbor, path, visited)
                    path.pop()
                    visited.remove(neighbor)
        
        dfs(source, [source], {source})
        return paths
    
    def identify_root_causes(
        self,
        failure_node: str,
        threshold: float = 0.3
    ) -> List[str]:
        """
        Identify root causes of a failure.
        Root causes are nodes with strong causal effect and few parents.
        """
        if failure_node not in self.nodes:
            return []
        
        root_causes = []
        
        for node_id, node in self.nodes.items():
            if node_id == failure_node:
                continue
            
            # Compute causal strength to failure
            strength = self.compute_causal_strength(node_id, failure_node)
            
            # Root causes have:
            # 1. Strong causal effect
            # 2. Few or no parents (they're "root")
            # 3. High deviation from expected
            
            if (strength > threshold and
                len(node.parents) <= 2 and
                node.deviation > 0.2):
                root_causes.append(node_id)
                node.is_root_cause = True
                node.causal_strength = strength
        
        # Sort by causal strength
        root_causes.sort(
            key=lambda nid: self.nodes[nid].causal_strength,
            reverse=True
        )
        
        return root_causes
    
    def export_graph(self, output_path: Path) -> None:
        """Export causal graph to JSON."""
        data = {
            "nodes": [
                {
                    "id": node.node_id,
                    "type": node.node_type.value,
                    "name": node.name,
                    "is_root_cause": node.is_root_cause,
                    "causal_strength": node.causal_strength,
                    "deviation": node.deviation
                }
                for node in self.nodes.values()
            ],
            "edges": [
                {
                    "source": edge.source,
                    "target": edge.target,
                    "causal_effect": edge.causal_effect,
                    "confidence": edge.confidence,
                    "relationship_type": edge.relationship_type
                }
                for edge in self.edges.values()
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        self.logger.info(f"Exported causal graph to {output_path}")
    
    def learn_structure(self, data: np.ndarray) -> Dict[str, Any]:
        """Learn causal structure from observational data."""
        n_samples, n_vars = data.shape
        
        # Mock structure learning - in production would use PC algorithm, GES, etc.
        learned_edges = []
        
        # Simple correlation-based structure discovery
        correlation_matrix = np.corrcoef(data.T)
        
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                corr = abs(correlation_matrix[i, j])
                if corr > 0.5:  # Threshold for significant correlation
                    # Assume temporal order determines causality
                    if i < j:
                        learned_edges.append((f"var_{i}", f"var_{j}"))
                        self.add_edge(f"var_{i}", f"var_{j}", causal_effect=corr)
        
        return {
            'edges': learned_edges,
            'dag': learned_edges,
            'confidence': 0.7
        }
    
    def discover_from_interventions(self, interventions: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
        """Discover causal edges from interventional data."""
        discovered_edges = []
        
        for intervention in interventions:
            variable = intervention['variable']
            observed_changes = intervention['observed_changes']
            
            source_var = f"var_{variable}"
            
            for target_var_idx, change in observed_changes.items():
                if abs(change) > 0.1:  # Significant change threshold
                    target_var = f"var_{target_var_idx}"
                    discovered_edges.append((source_var, target_var))
                    
                    # Add edge to graph
                    self.add_edge(
                        source=source_var,
                        target=target_var,
                        causal_effect=change,
                        interventional_evidence=1.0
                    )
        
        return discovered_edges
    
    def get_markov_blanket(self, variable_id: int) -> Set[str]:
        """Get Markov blanket of a variable (parents, children, and co-parents)."""
        var_name = f"var_{variable_id}"
        if var_name not in self.nodes:
            return set()
        
        node = self.nodes[var_name]
        markov_blanket = set()
        
        # Add parents
        markov_blanket.update(node.parents)
        
        # Add children
        markov_blanket.update(node.children)
        
        # Add co-parents (parents of children)
        for child in node.children:
            if child in self.nodes:
                markov_blanket.update(self.nodes[child].parents)
        
        # Remove the variable itself
        markov_blanket.discard(var_name)
        
        return markov_blanket
    
    def add_edge(self, source: str, target: str, causal_effect: float = 0.0, **kwargs) -> CausalEdge:
        """Add a causal edge between nodes (override to handle string/int conversion)."""
        # Handle numeric source/target
        if isinstance(source, int):
            source = f"var_{source}"
        if isinstance(target, int):
            target = f"var_{target}"
        
        # Ensure nodes exist
        if source not in self.nodes:
            self.add_node(source, CausalNodeType.INPUT, f"Variable {source}")
        if target not in self.nodes:
            self.add_node(target, CausalNodeType.INPUT, f"Variable {target}")
        
        edge = CausalEdge(
            source=source,
            target=target,
            causal_effect=causal_effect,
            **kwargs
        )
        
        self.edges[(source, target)] = edge
        
        # Update adjacency
        self.adjacency[source].add(target)
        
        # Update node relationships
        self.nodes[source].children.append(target)
        self.nodes[target].parents.append(source)
        
        return edge


class CounterfactualReasoner:
    """
    Reasons about counterfactuals: "What would have happened if...?"
    Uses causal graph to simulate alternative scenarios.
    """
    
    def __init__(self, causal_graph: CausalGraph = None, num_variables: int = 5):
        if causal_graph is None:
            causal_graph = CausalGraph(num_variables=num_variables)
        self.causal_graph = causal_graph
        self.num_variables = num_variables
        self.logger = logging.getLogger(__name__)
    
    def generate_counterfactual(
        self,
        node_id: str,
        counterfactual_value: Any,
        target_outcome: str
    ) -> Counterfactual:
        """
        Generate counterfactual: "What if node_id had value X?"
        
        Args:
            node_id: Node to change
            counterfactual_value: Value to change to
            target_outcome: Outcome node to predict
        
        Returns:
            Counterfactual analysis
        """
        node = self.causal_graph.nodes.get(node_id)
        if not node:
            raise ValueError(f"Node {node_id} not found")
        
        original_value = node.current_value
        
        # Simulate counterfactual by propagating change through graph
        predicted_outcome, outcome_change = self._simulate_intervention(
            node_id,
            counterfactual_value,
            target_outcome
        )
        
        # Assess plausibility
        plausibility = self._assess_plausibility(
            node_id,
            original_value,
            counterfactual_value
        )
        
        # Determine if actionable
        is_actionable, intervention = self._determine_actionability(node)
        
        # Generate description
        description = self._generate_counterfactual_description(
            node,
            original_value,
            counterfactual_value,
            outcome_change
        )
        
        return Counterfactual(
            counterfactual_id=f"cf_{node_id}_{datetime.utcnow().timestamp()}",
            changed_node=node_id,
            original_value=original_value,
            counterfactual_value=counterfactual_value,
            predicted_outcome=predicted_outcome,
            outcome_change=outcome_change,
            plausibility=plausibility,
            is_actionable=is_actionable,
            intervention_required=intervention,
            description=description
        )
    
    def _simulate_intervention(
        self,
        node_id: str,
        new_value: Any,
        target_outcome: str
    ) -> Tuple[Any, float]:
        """Simulate intervening on a node and predict outcome."""
        # Get causal strength
        causal_strength = self.causal_graph.compute_causal_strength(
            node_id,
            target_outcome
        )
        
        # Estimate value change
        node = self.causal_graph.nodes[node_id]
        if isinstance(new_value, (int, float)) and isinstance(node.current_value, (int, float)):
            value_change = abs(new_value - node.current_value) / (abs(node.current_value) + 1e-10)
        else:
            value_change = 1.0 if new_value != node.current_value else 0.0
        
        # Predict outcome change
        outcome_change = causal_strength * value_change
        
        # Mock predicted outcome (in production, would use actual simulation)
        target_node = self.causal_graph.nodes.get(target_outcome)
        if target_node and target_node.current_value is not None:
            if isinstance(target_node.current_value, (int, float)):
                predicted_outcome = target_node.current_value + outcome_change
            else:
                predicted_outcome = "improved" if outcome_change > 0 else "unchanged"
        else:
            predicted_outcome = None
        
        return predicted_outcome, outcome_change
    
    def _assess_plausibility(
        self,
        node_id: str,
        original_value: Any,
        counterfactual_value: Any
    ) -> float:
        """Assess how plausible the counterfactual is."""
        # Factors:
        # 1. How different from original (less difference = more plausible)
        # 2. Whether value is within reasonable range
        # 3. Historical variability of this node
        
        if isinstance(original_value, (int, float)) and isinstance(counterfactual_value, (int, float)):
            # Numeric values
            difference = abs(counterfactual_value - original_value) / (abs(original_value) + 1e-10)
            plausibility = max(0.0, 1.0 - difference)
        else:
            # Categorical values
            plausibility = 0.5 if original_value != counterfactual_value else 1.0
        
        return plausibility
    
    def _determine_actionability(
        self,
        node: CausalNode
    ) -> Tuple[bool, Optional[InterventionStrategy]]:
        """Determine if we can actually intervene on this node."""
        # Parameters and hyperparameters are actionable
        if node.node_type == CausalNodeType.PARAMETER:
            return True, InterventionStrategy.FINE_TUNE
        
        if node.node_type == CausalNodeType.HYPERPARAMETER:
            return True, InterventionStrategy.ADJUST_HYPERPARAMETERS
        
        # Environment factors may require data collection
        if node.node_type == CausalNodeType.ENVIRONMENT:
            return True, InterventionStrategy.COLLECT_MORE_DATA
        
        # Hidden states and outputs are not directly actionable
        return False, None
    
    def _generate_counterfactual_description(
        self,
        node: CausalNode,
        original_value: Any,
        counterfactual_value: Any,
        outcome_change: float
    ) -> str:
        """Generate natural language description of counterfactual."""
        direction = "improve" if outcome_change > 0 else "worsen"
        magnitude = abs(outcome_change)
        
        if magnitude > 0.5:
            impact = "significantly"
        elif magnitude > 0.2:
            impact = "moderately"
        else:
            impact = "slightly"
        
        return (
            f"If {node.name} was {counterfactual_value} instead of {original_value}, "
            f"the outcome would {impact} {direction} by {magnitude:.2%}"
        )
    
    def find_best_counterfactuals(
        self,
        target_outcome: str,
        num_counterfactuals: int = 5,
        require_actionable: bool = True
    ) -> List[Counterfactual]:
        """
        Find best counterfactuals to improve target outcome.
        
        Args:
            target_outcome: Outcome to improve
            num_counterfactuals: Number to return
            require_actionable: Only return actionable counterfactuals
        
        Returns:
            List of best counterfactuals
        """
        counterfactuals = []
        
        # Generate counterfactuals for all nodes
        for node_id, node in self.causal_graph.nodes.items():
            if node_id == target_outcome:
                continue
            
            # Generate counterfactual with improved value
            # Mock: in production, would use domain knowledge
            if isinstance(node.current_value, (int, float)):
                # Try increasing by 10%, 20%, 50%
                for factor in [1.1, 1.2, 1.5]:
                    cf_value = node.current_value * factor
                    try:
                        cf = self.generate_counterfactual(
                            node_id,
                            cf_value,
                            target_outcome
                        )
                        if not require_actionable or cf.is_actionable:
                            counterfactuals.append(cf)
                    except Exception as e:
                        self.logger.warning(f"Failed to generate counterfactual: {e}")
        
        # Sort by outcome change (descending)
        counterfactuals.sort(key=lambda cf: cf.outcome_change, reverse=True)
        
        return counterfactuals[:num_counterfactuals]


class CausalSelfDiagnosis:
    """
    Main causal self-diagnosis system.
    
    Integrates causal graph, counterfactual reasoning, and intervention planning
    to diagnose failures and recommend fixes.
    """
    
    def __init__(self):
        self.causal_graph = CausalGraph()
        self.counterfactual_reasoner = CounterfactualReasoner(self.causal_graph)
        
        # Diagnosis history
        self.diagnoses: List[FailureDiagnosis] = []
        
        # Intervention history
        self.interventions: List[InterventionPlan] = []
        
        self.logger = logging.getLogger(__name__)
    
    def build_causal_model(
        self,
        system_components: Dict[str, Any],
        observational_data: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """
        Build causal graph from system components and data.
        
        Args:
            system_components: Dict describing system structure
            observational_data: Optional historical data for learning edges
        """
        # Add nodes for system components
        for component_id, component in system_components.items():
            self.causal_graph.add_node(
                node_id=component_id,
                node_type=CausalNodeType[component.get("type", "HIDDEN").upper()],
                name=component.get("name", component_id),
                current_value=component.get("value"),
                expected_value=component.get("expected_value")
            )
        
        # Learn causal edges from structure and data
        self._learn_causal_edges(system_components, observational_data)
        
        self.logger.info(
            f"Built causal model with {len(self.causal_graph.nodes)} nodes "
            f"and {len(self.causal_graph.edges)} edges"
        )
    
    def _learn_causal_edges(
        self,
        system_components: Dict[str, Any],
        observational_data: Optional[List[Dict[str, Any]]]
    ) -> None:
        """Learn causal edges from structure and data."""
        # Add edges based on known structure
        for component_id, component in system_components.items():
            parents = component.get("parents", [])
            for parent in parents:
                if parent in self.causal_graph.nodes:
                    # Estimate causal effect
                    effect = self._estimate_causal_effect(
                        parent,
                        component_id,
                        observational_data
                    )
                    
                    self.causal_graph.add_edge(
                        source=parent,
                        target=component_id,
                        causal_effect=effect,
                        confidence=0.7,  # Mock
                        observational_evidence=0.6
                    )
    
    def _estimate_causal_effect(
        self,
        source: str,
        target: str,
        data: Optional[List[Dict[str, Any]]]
    ) -> float:
        """Estimate causal effect between nodes."""
        # Mock: In production, would use causal discovery algorithms
        # (e.g., PC algorithm, GES, or interventional data)
        return random.uniform(0.3, 0.9)
    
    def diagnose_failure(
        self,
        failure_description: Dict[str, Any],
        failure_mode: FailureMode
    ) -> FailureDiagnosis:
        """
        Diagnose a failure using causal analysis.
        
        Args:
            failure_description: Description of the failure
            failure_mode: Type of failure
        
        Returns:
            FailureDiagnosis
        """
        # Create failure node if not exists
        failure_node = "failure_outcome"
        if failure_node not in self.causal_graph.nodes:
            self.causal_graph.add_node(
                node_id=failure_node,
                node_type=CausalNodeType.OUTPUT,
                name="Failure Outcome",
                current_value=failure_description.get("severity", 1.0)
            )
        
        # Update node values from failure description
        for component_id, value in failure_description.get("component_values", {}).items():
            if component_id in self.causal_graph.nodes:
                node = self.causal_graph.nodes[component_id]
                node.current_value = value
                
                # Calculate deviation
                if node.expected_value is not None:
                    if isinstance(value, (int, float)) and isinstance(node.expected_value, (int, float)):
                        node.deviation = abs(value - node.expected_value) / (abs(node.expected_value) + 1e-10)
                    else:
                        node.deviation = 1.0 if value != node.expected_value else 0.0
        
        # Identify root causes
        root_causes = self.causal_graph.identify_root_causes(failure_node)
        
        # Find causal paths
        causal_paths = []
        for root in root_causes[:3]:  # Top 3 root causes
            path = self.causal_graph.find_causal_path(root, failure_node)
            if path:
                causal_paths.append(path)
        
        # Generate counterfactuals
        counterfactuals = self.counterfactual_reasoner.find_best_counterfactuals(
            target_outcome=failure_node,
            num_counterfactuals=5,
            require_actionable=True
        )
        
        # Recommend interventions based on counterfactuals
        interventions = self._recommend_interventions(
            root_causes,
            counterfactuals,
            failure_mode
        )
        
        # Create diagnosis
        diagnosis = FailureDiagnosis(
            diagnosis_id=f"diagnosis_{datetime.utcnow().timestamp()}",
            failure_mode=failure_mode,
            root_causes=root_causes,
            contributing_factors=[
                nid for nid, node in self.causal_graph.nodes.items()
                if node.deviation > 0.1 and nid not in root_causes
            ],
            causal_path=causal_paths[0] if causal_paths else [],
            diagnosis_confidence=0.8,  # Mock
            counterfactuals=counterfactuals,
            recommended_interventions=interventions
        )
        
        self.diagnoses.append(diagnosis)
        
        # Emit telemetry
        OBSERVABILITY.emit_counter("causal_diagnosis.performed")
        OBSERVABILITY.emit_gauge(
            "causal_diagnosis.root_causes",
            len(root_causes)
        )
        
        return diagnosis
    
    def _recommend_interventions(
        self,
        root_causes: List[str],
        counterfactuals: List[Counterfactual],
        failure_mode: FailureMode
    ) -> List[Tuple[InterventionStrategy, float]]:
        """Recommend interventions based on diagnosis."""
        interventions = []
        
        # Based on counterfactuals
        for cf in counterfactuals:
            if cf.is_actionable and cf.intervention_required:
                confidence = cf.plausibility * cf.outcome_change
                interventions.append((cf.intervention_required, confidence))
        
        # Based on failure mode
        mode_specific = self._get_mode_specific_interventions(failure_mode)
        interventions.extend(mode_specific)
        
        # Remove duplicates and sort by confidence
        seen = set()
        unique_interventions = []
        for strategy, conf in sorted(interventions, key=lambda x: x[1], reverse=True):
            if strategy not in seen:
                seen.add(strategy)
                unique_interventions.append((strategy, conf))
        
        return unique_interventions[:5]  # Top 5
    
    def _get_mode_specific_interventions(
        self,
        failure_mode: FailureMode
    ) -> List[Tuple[InterventionStrategy, float]]:
        """Get interventions specific to failure mode."""
        interventions_map = {
            FailureMode.ACCURACY_DROP: [
                (InterventionStrategy.FINE_TUNE, 0.8),
                (InterventionStrategy.COLLECT_MORE_DATA, 0.7)
            ],
            FailureMode.OVERFITTING: [
                (InterventionStrategy.ADD_REGULARIZATION, 0.9),
                (InterventionStrategy.COLLECT_MORE_DATA, 0.7)
            ],
            FailureMode.UNDERFITTING: [
                (InterventionStrategy.CHANGE_ARCHITECTURE, 0.8),
                (InterventionStrategy.ADJUST_HYPERPARAMETERS, 0.75)
            ],
            FailureMode.HALLUCINATION: [
                (InterventionStrategy.FINE_TUNE, 0.85),
                (InterventionStrategy.ADD_REGULARIZATION, 0.7)
            ],
            FailureMode.CATASTROPHIC_FORGETTING: [
                (InterventionStrategy.APPLY_PATCH, 0.9),
                (InterventionStrategy.RETRAIN, 0.6)
            ],
            FailureMode.DISTRIBUTION_SHIFT: [
                (InterventionStrategy.COLLECT_MORE_DATA, 0.85),
                (InterventionStrategy.FINE_TUNE, 0.75)
            ]
        }
        
        return interventions_map.get(failure_mode, [
            (InterventionStrategy.RETRAIN, 0.5)
        ])
    
    def create_intervention_plan(
        self,
        diagnosis: FailureDiagnosis,
        constraints: Optional[Dict[str, Any]] = None
    ) -> InterventionPlan:
        """
        Create detailed intervention plan from diagnosis.
        
        Args:
            diagnosis: Failure diagnosis
            constraints: Optional constraints (budget, time, etc.)
        
        Returns:
            InterventionPlan
        """
        constraints = constraints or {}
        
        # Select interventions within constraints
        max_cost = constraints.get("max_cost", float('inf'))
        max_time = constraints.get("max_time", float('inf'))
        
        selected_interventions = []
        total_cost = 0.0
        
        for strategy, confidence in diagnosis.recommended_interventions:
            # Estimate cost
            cost = self._estimate_intervention_cost(strategy)
            
            if total_cost + cost <= max_cost:
                # Add intervention details
                details = self._get_intervention_details(
                    strategy,
                    diagnosis.root_causes
                )
                selected_interventions.append((strategy, details))
                total_cost += cost
        
        # Calculate expected improvement
        expected_improvement = self._estimate_total_improvement(
            selected_interventions,
            diagnosis
        )
        
        # Assess risks
        risks = self._assess_intervention_risks(selected_interventions)
        
        plan = InterventionPlan(
            plan_id=f"plan_{datetime.utcnow().timestamp()}",
            target_failure=diagnosis.failure_mode,
            interventions=selected_interventions,
            expected_improvement=expected_improvement,
            confidence=diagnosis.diagnosis_confidence,
            estimated_cost=total_cost,
            risks=risks,
            validation_metrics=self._get_validation_metrics(diagnosis.failure_mode)
        )
        
        self.interventions.append(plan)
        
        return plan
    
    def _estimate_intervention_cost(
        self,
        strategy: InterventionStrategy
    ) -> float:
        """Estimate cost of intervention."""
        cost_map = {
            InterventionStrategy.RETRAIN: 1.0,
            InterventionStrategy.FINE_TUNE: 0.3,
            InterventionStrategy.ADJUST_HYPERPARAMETERS: 0.1,
            InterventionStrategy.ADD_REGULARIZATION: 0.2,
            InterventionStrategy.COLLECT_MORE_DATA: 0.5,
            InterventionStrategy.CHANGE_ARCHITECTURE: 0.8,
            InterventionStrategy.APPLY_PATCH: 0.15,
            InterventionStrategy.RESET_COMPONENT: 0.05
        }
        return cost_map.get(strategy, 0.5)
    
    def _get_intervention_details(
        self,
        strategy: InterventionStrategy,
        root_causes: List[str]
    ) -> Dict[str, Any]:
        """Get specific details for intervention."""
        details = {
            "strategy": strategy.value,
            "target_components": root_causes[:3],
            "estimated_duration": "varies"
        }
        
        if strategy == InterventionStrategy.FINE_TUNE:
            details.update({
                "learning_rate": 1e-5,
                "num_epochs": 10,
                "target_layers": root_causes
            })
        elif strategy == InterventionStrategy.ADJUST_HYPERPARAMETERS:
            details.update({
                "parameters_to_adjust": root_causes,
                "optimization_method": "grid_search"
            })
        elif strategy == InterventionStrategy.ADD_REGULARIZATION:
            details.update({
                "regularization_type": "l2",
                "regularization_strength": 0.01
            })
        
        return details
    
    def _estimate_total_improvement(
        self,
        interventions: List[Tuple[InterventionStrategy, Dict[str, Any]]],
        diagnosis: FailureDiagnosis
    ) -> float:
        """Estimate total improvement from interventions."""
        # Sum improvements from counterfactuals
        total = 0.0
        for cf in diagnosis.counterfactuals:
            if cf.is_actionable:
                total += cf.outcome_change * 0.8  # Discount for uncertainty
        
        # Add synergistic effects
        if len(interventions) > 1:
            total *= 1.1  # 10% boost from multiple interventions
        
        return min(1.0, total)
    
    def _assess_intervention_risks(
        self,
        interventions: List[Tuple[InterventionStrategy, Dict[str, Any]]]
    ) -> List[str]:
        """Assess risks of intervention plan."""
        risks = []
        
        strategies = [s for s, _ in interventions]
        
        if InterventionStrategy.RETRAIN in strategies:
            risks.append("Risk of catastrophic forgetting")
            risks.append("High computational cost")
        
        if InterventionStrategy.CHANGE_ARCHITECTURE in strategies:
            risks.append("May require retraining from scratch")
            risks.append("Compatibility issues with existing components")
        
        if len(interventions) > 3:
            risks.append("Complex intervention plan may be difficult to validate")
        
        return risks
    
    def _get_validation_metrics(
        self,
        failure_mode: FailureMode
    ) -> List[str]:
        """Get metrics to validate intervention success."""
        metrics_map = {
            FailureMode.ACCURACY_DROP: ["accuracy", "f1_score", "precision", "recall"],
            FailureMode.OVERFITTING: ["validation_loss", "train_val_gap", "generalization_score"],
            FailureMode.HALLUCINATION: ["factuality_score", "faithfulness", "consistency"],
            FailureMode.CATASTROPHIC_FORGETTING: ["backward_transfer", "average_accuracy"],
            FailureMode.DISTRIBUTION_SHIFT: ["domain_accuracy", "transfer_performance"]
        }
        
        return metrics_map.get(failure_mode, ["accuracy", "loss"])
    
    def get_diagnosis_summary(self) -> Dict[str, Any]:
        """Get summary of all diagnoses."""
        if not self.diagnoses:
            return {"total_diagnoses": 0}
        
        # Aggregate statistics
        failure_modes = defaultdict(int)
        root_causes_freq = defaultdict(int)
        
        for diagnosis in self.diagnoses:
            failure_modes[diagnosis.failure_mode.value] += 1
            for cause in diagnosis.root_causes:
                root_causes_freq[cause] += 1
        
        return {
            "total_diagnoses": len(self.diagnoses),
            "failure_modes": dict(failure_modes),
            "most_common_root_causes": sorted(
                root_causes_freq.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10],
            "total_interventions_planned": len(self.interventions),
            "causal_graph_size": {
                "nodes": len(self.causal_graph.nodes),
                "edges": len(self.causal_graph.edges)
            }
        }
    
    def export_diagnosis_data(self, output_path: Path) -> None:
        """Export all diagnosis data."""
        data = {
            "diagnoses": [asdict(d) for d in self.diagnoses],
            "interventions": [asdict(p) for p in self.interventions],
            "summary": self.get_diagnosis_summary(),
            "exported_at": datetime.utcnow().isoformat()
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        self.logger.info(f"Exported diagnosis data to {output_path}")
    
    def identify_root_causes(self, failure_data: Dict[str, Any]) -> List[str]:
        """Identify root causes from failure data."""
        # Create failure diagnosis based on the data
        failure_mode = FailureMode.ACCURACY_DROP  # Default
        
        # Detect failure mode from data
        if 'model_accuracy' in failure_data and failure_data['model_accuracy'] < 0.5:
            failure_mode = FailureMode.ACCURACY_DROP
        
        # Update causal graph with failure data
        for var_name, value in failure_data.items():
            # Add node if it doesn't exist
            if var_name not in self.causal_graph.nodes:
                self.causal_graph.add_node(
                    node_id=var_name,
                    node_type=CausalNodeType.INPUT,
                    name=var_name.replace('_', ' ').title(),
                    current_value=value
                )
            else:
                self.causal_graph.nodes[var_name].current_value = value
        
        # Perform diagnosis
        diagnosis = self.diagnose_failure(failure_data, failure_mode)
        
        return diagnosis.root_causes
    
    def recommend_interventions(self, diagnosis: FailureDiagnosis) -> List[Tuple[InterventionStrategy, float]]:
        """Recommend interventions for a diagnosis."""
        return diagnosis.recommended_interventions
    
    def detect_degradation(self, performance_data: Dict[str, Any]) -> bool:
        """Detect performance degradation."""
        # Simple degradation detection
        for metric_name, value in performance_data.items():
            if 'accuracy' in metric_name.lower() and value < 0.7:
                return True
            if 'loss' in metric_name.lower() and value > 1.0:
                return True
        return False
    
    def track_intervention_impact(self, intervention_id: str, before_metrics: Dict[str, Any], after_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Track the impact of an intervention."""
        impact = {}
        
        for metric_name in before_metrics:
            if metric_name in after_metrics:
                before_val = before_metrics[metric_name]
                after_val = after_metrics[metric_name]
                
                if isinstance(before_val, (int, float)) and isinstance(after_val, (int, float)):
                    change = after_val - before_val
                    relative_change = change / (abs(before_val) + 1e-10)
                    impact[metric_name] = {
                        'absolute_change': change,
                        'relative_change': relative_change,
                        'improvement': change > 0 if 'accuracy' in metric_name.lower() else change < 0
                    }
        
        return {
            'intervention_id': intervention_id,
            'impact': impact,
            'overall_improvement': sum(1 for m in impact.values() if m['improvement']) / len(impact) if impact else 0.0
        }
    
    def find_explanations(self, query: Dict[str, Any]) -> List[str]:
        """Find explanations for observed phenomena."""
        explanations = []
        
        # Simple explanation generation
        for node_id, node in self.causal_graph.nodes.items():
            if node.is_root_cause:
                explanations.append(f"{node.name} is a root cause with strength {node.causal_strength:.2f}")
        
        # Add causal path explanations
        if len(explanations) == 0:
            explanations.append("No clear causal explanations found")
        
        return explanations


def create_causal_diagnosis_system(num_variables: int = 10, num_interventions: int = 5) -> CausalSelfDiagnosis:
    """
    Factory function to create a configured Causal Self-Diagnosis System.
    
    Args:
        num_variables: Number of variables in causal graph
        num_interventions: Number of interventions to consider
    
    Returns:
        Configured CausalSelfDiagnosis
    """
    system = CausalSelfDiagnosis()
    system.causal_graph = CausalGraph(num_variables=num_variables)
    system.counterfactual_reasoner = CounterfactualReasoner(system.causal_graph, num_variables)
    
    logging.info(f"Created Causal Self-Diagnosis System with {num_variables} variables")
    
    return system
