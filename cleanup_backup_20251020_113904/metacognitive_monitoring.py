"""
Metacognitive Monitoring System - Symbio AI

Provides self-awareness capabilities for AI systems to monitor and understand
their own cognitive processes, performance, limitations, and decision-making.

This system enables AI to:
- Monitor its own performance and confidence
- Detect uncertainty and knowledge gaps
- Track reasoning processes and decision paths
- Identify when to seek help or defer decisions
- Learn from self-reflection and metacognitive analysis
"""

import asyncio
import logging
import random
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Callable, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime
from collections import defaultdict, deque
from enum import Enum

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Mock torch
    class torch:
        class nn:
            class Module:
                pass
            class LSTM:
                def __init__(self, *args, **kwargs): pass
            class Linear:
                def __init__(self, *args, **kwargs): pass
        @staticmethod
        def tensor(x): return x
        @staticmethod
        def sigmoid(x): return 1 / (1 + np.exp(-np.array(x))) if hasattr(x, '__iter__') else 1 / (1 + np.exp(-x))
    
    class F:
        @staticmethod
        def softmax(x, dim=-1): return x

from deployment.observability import OBSERVABILITY


class CognitiveState(Enum):
    """States of cognitive processing."""
    CONFIDENT = "confident"  # High confidence, reliable
    UNCERTAIN = "uncertain"  # Low confidence, uncertain
    CONFUSED = "confused"  # Contradictory information
    LEARNING = "learning"  # Actively learning/adapting
    STABLE = "stable"  # Stable, consistent performance
    DEGRADING = "degrading"  # Performance declining
    RECOVERING = "recovering"  # Improving from degradation


class MetacognitiveSignal(Enum):
    """Types of metacognitive signals."""
    CONFIDENCE = "confidence"  # Prediction confidence
    UNCERTAINTY = "uncertainty"  # Epistemic uncertainty
    ATTENTION = "attention"  # Attention distribution
    SURPRISE = "surprise"  # Unexpected outcomes
    FAMILIARITY = "familiarity"  # Input familiarity
    COMPLEXITY = "complexity"  # Task complexity
    ERROR_LIKELIHOOD = "error_likelihood"  # Predicted error probability


class InterventionType(Enum):
    """Types of metacognitive interventions."""
    DEFER_TO_HUMAN = "defer_to_human"
    REQUEST_MORE_DATA = "request_more_data"
    SEEK_EXPERT = "seek_expert"
    INCREASE_COMPUTE = "increase_compute"
    SIMPLIFY_TASK = "simplify_task"
    ACTIVATE_FALLBACK = "activate_fallback"
    TRIGGER_LEARNING = "trigger_learning"
    NO_INTERVENTION = "no_intervention"


@dataclass
class MetacognitiveState:
    """Complete metacognitive state at a point in time."""
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    # Cognitive state
    cognitive_state: CognitiveState = CognitiveState.STABLE
    
    # Confidence metrics
    prediction_confidence: float = 0.0  # Confidence in current prediction
    calibration_score: float = 0.0  # How well-calibrated confidence is
    
    # Uncertainty metrics
    epistemic_uncertainty: float = 0.0  # Model uncertainty
    aleatoric_uncertainty: float = 0.0  # Data uncertainty
    total_uncertainty: float = 0.0
    
    # Attention and focus
    attention_entropy: float = 0.0  # Distribution of attention
    focus_score: float = 0.0  # How focused vs. scattered
    
    # Performance awareness
    perceived_performance: float = 0.0  # Self-assessed performance
    actual_performance: float = 0.0  # Ground truth performance
    performance_gap: float = 0.0  # Difference (awareness accuracy)
    
    # Reasoning process
    reasoning_steps: int = 0
    reasoning_complexity: float = 0.0
    decision_path: List[str] = field(default_factory=list)
    
    # Knowledge state
    knowledge_coverage: float = 0.0  # How much relevant knowledge
    knowledge_gaps: List[str] = field(default_factory=list)
    
    # Metacognitive signals
    signals: Dict[MetacognitiveSignal, float] = field(default_factory=dict)
    
    # Intervention recommendations
    recommended_intervention: InterventionType = InterventionType.NO_INTERVENTION
    intervention_confidence: float = 0.0


@dataclass
class CognitiveEvent:
    """Significant cognitive event requiring attention."""
    event_id: str
    event_type: str  # "high_uncertainty", "performance_drop", "knowledge_gap", etc.
    severity: float  # 0-1
    
    # Context
    task_context: Dict[str, Any] = field(default_factory=dict)
    metacognitive_state: Optional[MetacognitiveState] = None
    
    # Analysis
    root_cause: Optional[str] = None
    contributing_factors: List[str] = field(default_factory=list)
    
    # Response
    intervention_taken: Optional[InterventionType] = None
    outcome: Optional[str] = None
    
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class ReflectionInsight:
    """Insight gained from metacognitive reflection."""
    insight_id: str
    insight_type: str  # "pattern", "limitation", "strength", "improvement_opportunity"
    
    description: str
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    
    # Actionable recommendations
    recommendations: List[str] = field(default_factory=list)
    expected_impact: float = 0.0
    
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class ConfidenceEstimator(nn.Module):
    """
    Neural network that estimates prediction confidence and uncertainty.
    Learns to predict when the model is likely to be wrong.
    """
    
    def __init__(self, input_dim: int = 128, hidden_dim: int = 64):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Confidence prediction
        self.confidence_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        ) if TORCH_AVAILABLE else lambda x: torch.sigmoid(x)
        
        # Uncertainty estimation
        self.uncertainty_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),  # [epistemic, aleatoric]
            nn.Softplus()
        ) if TORCH_AVAILABLE else lambda x: x
        
        # Calibration layer
        self.calibration = nn.Linear(3, 1) if TORCH_AVAILABLE else lambda x: x
    
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Predict confidence and uncertainty from features.
        
        Args:
            features: Model features or representations
        
        Returns:
            Dict with confidence, uncertainties
        """
        if TORCH_AVAILABLE:
            confidence = self.confidence_net(features)
            uncertainties = self.uncertainty_net(features)
            
            # Calibrate confidence
            calibration_input = torch.cat([
                confidence,
                uncertainties[:, 0:1],  # epistemic
                uncertainties[:, 1:2]   # aleatoric
            ], dim=-1)
            calibrated_confidence = torch.sigmoid(self.calibration(calibration_input))
            
            return {
                "confidence": confidence,
                "calibrated_confidence": calibrated_confidence,
                "epistemic_uncertainty": uncertainties[:, 0],
                "aleatoric_uncertainty": uncertainties[:, 1],
                "total_uncertainty": uncertainties.sum(dim=-1)
            }
        else:
            # Mock implementation
            return {
                "confidence": torch.tensor([random.uniform(0.5, 0.95)]),
                "calibrated_confidence": torch.tensor([random.uniform(0.5, 0.9)]),
                "epistemic_uncertainty": torch.tensor([random.uniform(0.1, 0.3)]),
                "aleatoric_uncertainty": torch.tensor([random.uniform(0.1, 0.3)]),
                "total_uncertainty": torch.tensor([random.uniform(0.2, 0.6)])
            }


class AttentionMonitor:
    """
    Monitors attention patterns to understand what the model focuses on
    and identify potential issues with attention distribution.
    """
    
    def __init__(self):
        self.attention_history: deque = deque(maxlen=100)
        self.logger = logging.getLogger(__name__)
    
    def analyze_attention(
        self,
        attention_weights: np.ndarray
    ) -> Dict[str, float]:
        """
        Analyze attention distribution.
        
        Args:
            attention_weights: Attention weights [seq_len, seq_len] or similar
        
        Returns:
            Dict with attention metrics
        """
        # Ensure numpy array
        if not isinstance(attention_weights, np.ndarray):
            attention_weights = np.array(attention_weights)
        
        # Flatten if multi-dimensional
        if len(attention_weights.shape) > 1:
            attention_weights = attention_weights.flatten()
        
        # Normalize to probability distribution
        attention_weights = attention_weights / (attention_weights.sum() + 1e-10)
        
        # Calculate entropy (measure of attention spread)
        entropy = -np.sum(
            attention_weights * np.log(attention_weights + 1e-10)
        )
        max_entropy = np.log(len(attention_weights))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        # Focus score (inverse of entropy)
        focus_score = 1.0 - normalized_entropy
        
        # Identify attention peaks
        mean_attention = attention_weights.mean()
        std_attention = attention_weights.std()
        peak_threshold = mean_attention + 2 * std_attention
        num_peaks = (attention_weights > peak_threshold).sum()
        
        # Attention uniformity
        uniformity = 1.0 - std_attention / (mean_attention + 1e-10)
        
        metrics = {
            "entropy": float(entropy),
            "normalized_entropy": float(normalized_entropy),
            "focus_score": float(focus_score),
            "num_peaks": int(num_peaks),
            "uniformity": float(uniformity),
            "max_attention": float(attention_weights.max()),
            "mean_attention": float(mean_attention)
        }
        
        self.attention_history.append(metrics)
        
        return metrics
    
    def detect_attention_anomalies(self) -> List[str]:
        """Detect anomalies in attention patterns."""
        if len(self.attention_history) < 10:
            return []
        
        anomalies = []
        
        # Get recent attention metrics
        recent = list(self.attention_history)[-10:]
        
        # Check for extremely low focus
        avg_focus = np.mean([m["focus_score"] for m in recent])
        if avg_focus < 0.3:
            anomalies.append("low_focus_pattern")
        
        # Check for too uniform attention (might miss important info)
        avg_uniformity = np.mean([m["uniformity"] for m in recent])
        if avg_uniformity > 0.9:
            anomalies.append("overly_uniform_attention")
        
        # Check for too many peaks (scattered attention)
        avg_peaks = np.mean([m["num_peaks"] for m in recent])
        if avg_peaks > 5:
            anomalies.append("scattered_attention")
        
        return anomalies


class ReasoningTracer:
    """
    Traces and analyzes the reasoning process to understand
    decision paths and identify potential reasoning errors.
    """
    
    def __init__(self):
        self.reasoning_traces: List[Dict[str, Any]] = []
        self.logger = logging.getLogger(__name__)
    
    def trace_reasoning_step(
        self,
        step_id: str,
        step_type: str,
        inputs: Any,
        outputs: Any,
        confidence: float = 0.0,
        metadata: Dict[str, Any] = None
    ) -> None:
        """Record a reasoning step."""
        step = {
            "step_id": step_id,
            "step_type": step_type,
            "timestamp": datetime.utcnow().isoformat(),
            "confidence": confidence,
            "metadata": metadata or {}
        }
        
        self.reasoning_traces.append(step)
    
    def analyze_reasoning_path(self) -> Dict[str, Any]:
        """Analyze the complete reasoning path."""
        if not self.reasoning_traces:
            return {
                "num_steps": 0,
                "complexity": 0.0,
                "confidence_trajectory": [],
                "bottlenecks": [],
                "decision_points": []
            }
        
        # Calculate complexity
        num_steps = len(self.reasoning_traces)
        
        # Confidence trajectory
        confidences = [s.get("confidence", 0.0) for s in self.reasoning_traces]
        
        # Identify bottlenecks (low confidence steps)
        bottlenecks = [
            s["step_id"] for s in self.reasoning_traces
            if s.get("confidence", 1.0) < 0.5
        ]
        
        # Identify decision points (high-impact steps)
        decision_points = [
            s["step_id"] for s in self.reasoning_traces
            if s.get("step_type") in ["decision", "branch", "critical"]
        ]
        
        # Calculate reasoning complexity
        # Based on number of steps, branches, and confidence variance
        complexity = num_steps * 0.3 + len(decision_points) * 0.5
        if confidences:
            complexity += np.var(confidences) * 0.2
        
        return {
            "num_steps": num_steps,
            "complexity": min(1.0, complexity / 10),  # Normalize
            "confidence_trajectory": confidences,
            "avg_confidence": np.mean(confidences) if confidences else 0.0,
            "min_confidence": min(confidences) if confidences else 0.0,
            "bottlenecks": bottlenecks,
            "decision_points": decision_points
        }
    
    def clear_trace(self):
        """Clear reasoning trace for new task."""
        self.reasoning_traces.clear()


class MetacognitiveMonitor:
    """
    Main metacognitive monitoring system.
    
    Integrates confidence estimation, attention monitoring, reasoning tracing,
    and self-reflection to provide comprehensive self-awareness.
    """
    
    def __init__(
        self,
        feature_dim: int = 128,
        confidence_threshold: float = 0.7,
        uncertainty_threshold: float = 0.4
    ):
        self.feature_dim = feature_dim
        self.confidence_threshold = confidence_threshold
        self.uncertainty_threshold = uncertainty_threshold
        
        # Core components
        self.confidence_estimator = ConfidenceEstimator(
            input_dim=feature_dim,
            hidden_dim=64
        )
        
        self.attention_monitor = AttentionMonitor()
        self.reasoning_tracer = ReasoningTracer()
        
        # State tracking
        self.current_state: Optional[MetacognitiveState] = None
        self.state_history: deque = deque(maxlen=1000)
        
        # Event tracking
        self.cognitive_events: List[CognitiveEvent] = []
        
        # Reflection insights
        self.insights: List[ReflectionInsight] = []
        
        # Performance tracking
        self.performance_history: deque = deque(maxlen=100)
        
        self.logger = logging.getLogger(__name__)
    
    def monitor_prediction(
        self,
        features: torch.Tensor,
        prediction: Any,
        attention_weights: Optional[np.ndarray] = None,
        ground_truth: Optional[Any] = None
    ) -> MetacognitiveState:
        """
        Monitor a prediction and update metacognitive state.
        
        Args:
            features: Model features/representations
            prediction: Model prediction
            attention_weights: Optional attention weights
            ground_truth: Optional ground truth for calibration
        
        Returns:
            MetacognitiveState
        """
        # Estimate confidence and uncertainty
        confidence_outputs = self.confidence_estimator(features)
        
        # Analyze attention if available
        attention_metrics = {}
        if attention_weights is not None:
            attention_metrics = self.attention_monitor.analyze_attention(
                attention_weights
            )
        
        # Analyze reasoning path
        reasoning_analysis = self.reasoning_tracer.analyze_reasoning_path()
        
        # Create metacognitive state
        state = MetacognitiveState(
            prediction_confidence=float(confidence_outputs["calibrated_confidence"]),
            epistemic_uncertainty=float(confidence_outputs["epistemic_uncertainty"]),
            aleatoric_uncertainty=float(confidence_outputs["aleatoric_uncertainty"]),
            total_uncertainty=float(confidence_outputs["total_uncertainty"]),
            attention_entropy=attention_metrics.get("normalized_entropy", 0.0),
            focus_score=attention_metrics.get("focus_score", 0.0),
            reasoning_steps=reasoning_analysis["num_steps"],
            reasoning_complexity=reasoning_analysis["complexity"],
            decision_path=[s["step_id"] for s in self.reasoning_tracer.reasoning_traces]
        )
        
        # Determine cognitive state
        state.cognitive_state = self._determine_cognitive_state(state)
        
        # Calculate metacognitive signals
        state.signals = self._calculate_metacognitive_signals(state)
        
        # Recommend intervention if needed
        state.recommended_intervention, state.intervention_confidence = \
            self._recommend_intervention(state)
        
        # Update state
        self.current_state = state
        self.state_history.append(state)
        
        # Check for significant events
        self._check_for_cognitive_events(state)
        
        # Emit telemetry
        OBSERVABILITY.emit_gauge(
            "metacognitive.confidence",
            state.prediction_confidence
        )
        OBSERVABILITY.emit_gauge(
            "metacognitive.uncertainty",
            state.total_uncertainty
        )
        
        return state
    
    def _determine_cognitive_state(
        self,
        state: MetacognitiveState
    ) -> CognitiveState:
        """Determine overall cognitive state."""
        # High confidence and low uncertainty = CONFIDENT
        if (state.prediction_confidence > self.confidence_threshold and
            state.total_uncertainty < self.uncertainty_threshold):
            return CognitiveState.CONFIDENT
        
        # Low confidence or high uncertainty = UNCERTAIN
        if (state.prediction_confidence < 0.5 or
            state.total_uncertainty > 0.6):
            return CognitiveState.UNCERTAIN
        
        # Check performance trend
        if len(self.performance_history) >= 10:
            recent_perf = list(self.performance_history)[-10:]
            trend = np.polyfit(range(len(recent_perf)), recent_perf, 1)[0]
            
            if trend < -0.05:
                return CognitiveState.DEGRADING
            elif trend > 0.05:
                return CognitiveState.RECOVERING
        
        return CognitiveState.STABLE
    
    def _calculate_metacognitive_signals(
        self,
        state: MetacognitiveState
    ) -> Dict[MetacognitiveSignal, float]:
        """Calculate various metacognitive signals."""
        signals = {}
        
        # Confidence signal
        signals[MetacognitiveSignal.CONFIDENCE] = state.prediction_confidence
        
        # Uncertainty signal
        signals[MetacognitiveSignal.UNCERTAINTY] = state.total_uncertainty
        
        # Attention signal
        signals[MetacognitiveSignal.ATTENTION] = state.focus_score
        
        # Complexity signal
        signals[MetacognitiveSignal.COMPLEXITY] = state.reasoning_complexity
        
        # Error likelihood (inverse of confidence + uncertainty)
        signals[MetacognitiveSignal.ERROR_LIKELIHOOD] = (
            (1 - state.prediction_confidence) * 0.5 +
            state.total_uncertainty * 0.5
        )
        
        # Surprise (deviation from expected)
        if len(self.state_history) > 0:
            prev_confidence = self.state_history[-1].prediction_confidence
            surprise = abs(state.prediction_confidence - prev_confidence)
            signals[MetacognitiveSignal.SURPRISE] = surprise
        else:
            signals[MetacognitiveSignal.SURPRISE] = 0.0
        
        return signals
    
    def _recommend_intervention(
        self,
        state: MetacognitiveState
    ) -> Tuple[InterventionType, float]:
        """Recommend intervention based on metacognitive state."""
        # High uncertainty or low confidence = defer to human
        if (state.total_uncertainty > 0.7 or
            state.prediction_confidence < 0.3):
            return InterventionType.DEFER_TO_HUMAN, 0.9
        
        # Moderate uncertainty = request more data
        if state.total_uncertainty > 0.5:
            return InterventionType.REQUEST_MORE_DATA, 0.7
        
        # High complexity = seek expert or simplify
        if state.reasoning_complexity > 0.8:
            if state.prediction_confidence < 0.6:
                return InterventionType.SEEK_EXPERT, 0.75
            else:
                return InterventionType.SIMPLIFY_TASK, 0.6
        
        # Degrading performance = trigger learning
        if state.cognitive_state == CognitiveState.DEGRADING:
            return InterventionType.TRIGGER_LEARNING, 0.8
        
        # Low focus = increase compute/attention
        if state.focus_score < 0.4:
            return InterventionType.INCREASE_COMPUTE, 0.65
        
        return InterventionType.NO_INTERVENTION, 1.0
    
    def _check_for_cognitive_events(self, state: MetacognitiveState) -> None:
        """Check for significant cognitive events."""
        # High uncertainty event
        if state.total_uncertainty > 0.8:
            event = CognitiveEvent(
                event_id=f"high_uncertainty_{datetime.utcnow().timestamp()}",
                event_type="high_uncertainty",
                severity=state.total_uncertainty,
                metacognitive_state=state,
                contributing_factors=["epistemic_uncertainty", "aleatoric_uncertainty"]
            )
            self.cognitive_events.append(event)
        
        # Performance degradation event
        if state.cognitive_state == CognitiveState.DEGRADING:
            event = CognitiveEvent(
                event_id=f"performance_degradation_{datetime.utcnow().timestamp()}",
                event_type="performance_degradation",
                severity=0.7,
                metacognitive_state=state,
                contributing_factors=["declining_performance_trend"]
            )
            self.cognitive_events.append(event)
        
        # Attention anomaly event
        anomalies = self.attention_monitor.detect_attention_anomalies()
        if anomalies:
            event = CognitiveEvent(
                event_id=f"attention_anomaly_{datetime.utcnow().timestamp()}",
                event_type="attention_anomaly",
                severity=0.6,
                metacognitive_state=state,
                contributing_factors=anomalies
            )
            self.cognitive_events.append(event)
    
    def reflect_on_performance(
        self,
        time_window: int = 100
    ) -> List[ReflectionInsight]:
        """
        Perform metacognitive reflection on recent performance.
        
        Args:
            time_window: Number of recent states to analyze
        
        Returns:
            List of insights
        """
        if len(self.state_history) < 10:
            return []
        
        insights = []
        recent_states = list(self.state_history)[-time_window:]
        
        # Analyze confidence calibration
        insights.append(self._analyze_confidence_calibration(recent_states))
        
        # Analyze uncertainty patterns
        insights.append(self._analyze_uncertainty_patterns(recent_states))
        
        # Analyze attention patterns
        insights.append(self._analyze_attention_patterns(recent_states))
        
        # Analyze reasoning complexity
        insights.append(self._analyze_reasoning_patterns(recent_states))
        
        # Store insights
        self.insights.extend([i for i in insights if i is not None])
        
        return insights
    
    def _analyze_confidence_calibration(
        self,
        states: List[MetacognitiveState]
    ) -> Optional[ReflectionInsight]:
        """Analyze how well-calibrated confidence is."""
        # Mock: In production, would compare confidence to actual accuracy
        avg_confidence = np.mean([s.prediction_confidence for s in states])
        
        if avg_confidence > 0.8:
            return ReflectionInsight(
                insight_id=f"calibration_{datetime.utcnow().timestamp()}",
                insight_type="strength",
                description="High confidence in predictions, suggesting well-calibrated model",
                confidence=0.8,
                recommendations=[
                    "Continue current confidence estimation approach",
                    "Monitor for overconfidence in edge cases"
                ],
                expected_impact=0.1
            )
        elif avg_confidence < 0.5:
            return ReflectionInsight(
                insight_id=f"calibration_{datetime.utcnow().timestamp()}",
                insight_type="limitation",
                description="Low confidence across predictions, suggesting miscalibration",
                confidence=0.75,
                recommendations=[
                    "Retrain confidence estimator",
                    "Collect more diverse training data",
                    "Review uncertainty estimation methodology"
                ],
                expected_impact=0.3
            )
        
        return None
    
    def _analyze_uncertainty_patterns(
        self,
        states: List[MetacognitiveState]
    ) -> Optional[ReflectionInsight]:
        """Analyze uncertainty patterns."""
        uncertainties = [s.total_uncertainty for s in states]
        avg_uncertainty = np.mean(uncertainties)
        std_uncertainty = np.std(uncertainties)
        
        if std_uncertainty > 0.3:
            return ReflectionInsight(
                insight_id=f"uncertainty_{datetime.utcnow().timestamp()}",
                insight_type="pattern",
                description=f"High variance in uncertainty (std={std_uncertainty:.3f}), suggesting inconsistent confidence",
                confidence=0.7,
                recommendations=[
                    "Investigate sources of uncertainty variance",
                    "Improve epistemic uncertainty estimation",
                    "Consider task-specific uncertainty calibration"
                ],
                expected_impact=0.25
            )
        
        return None
    
    def _analyze_attention_patterns(
        self,
        states: List[MetacognitiveState]
    ) -> Optional[ReflectionInsight]:
        """Analyze attention patterns."""
        focus_scores = [s.focus_score for s in states]
        avg_focus = np.mean(focus_scores)
        
        if avg_focus < 0.4:
            return ReflectionInsight(
                insight_id=f"attention_{datetime.utcnow().timestamp()}",
                insight_type="limitation",
                description=f"Low attention focus (avg={avg_focus:.3f}), may miss important information",
                confidence=0.8,
                recommendations=[
                    "Review attention mechanism design",
                    "Implement sparse attention or attention regularization",
                    "Investigate input characteristics causing scattered attention"
                ],
                expected_impact=0.35
            )
        
        return None
    
    def _analyze_reasoning_patterns(
        self,
        states: List[MetacognitiveState]
    ) -> Optional[ReflectionInsight]:
        """Analyze reasoning complexity patterns."""
        complexities = [s.reasoning_complexity for s in states]
        avg_complexity = np.mean(complexities)
        
        if avg_complexity > 0.8:
            return ReflectionInsight(
                insight_id=f"reasoning_{datetime.utcnow().timestamp()}",
                insight_type="improvement_opportunity",
                description=f"High reasoning complexity (avg={avg_complexity:.3f}), potential for optimization",
                confidence=0.75,
                recommendations=[
                    "Simplify reasoning chains where possible",
                    "Implement caching for common reasoning patterns",
                    "Consider more efficient reasoning strategies"
                ],
                expected_impact=0.3
            )
        
        return None
    
    def get_self_awareness_report(self) -> Dict[str, Any]:
        """Generate comprehensive self-awareness report."""
        if not self.current_state:
            return {"status": "no_data"}
        
        return {
            "current_cognitive_state": self.current_state.cognitive_state.value,
            "confidence": self.current_state.prediction_confidence,
            "uncertainty": self.current_state.total_uncertainty,
            "attention_focus": self.current_state.focus_score,
            "reasoning_complexity": self.current_state.reasoning_complexity,
            "recommended_intervention": self.current_state.recommended_intervention.value,
            "intervention_confidence": self.current_state.intervention_confidence,
            "recent_events": len([
                e for e in self.cognitive_events
                if (datetime.utcnow() - datetime.fromisoformat(e.timestamp)).seconds < 3600
            ]),
            "insights_discovered": len(self.insights),
            "performance_trend": self._calculate_performance_trend(),
            "knowledge_gaps": self.current_state.knowledge_gaps,
            "metacognitive_signals": {
                k.value: v for k, v in self.current_state.signals.items()
            }
        }
    
    def _calculate_performance_trend(self) -> str:
        """Calculate recent performance trend."""
        if len(self.performance_history) < 10:
            return "insufficient_data"
        
        recent = list(self.performance_history)[-10:]
        trend = np.polyfit(range(len(recent)), recent, 1)[0]
        
        if trend > 0.05:
            return "improving"
        elif trend < -0.05:
            return "declining"
        else:
            return "stable"
    
    def export_metacognitive_data(self, output_path: Path) -> None:
        """Export metacognitive data for analysis."""
        data = {
            "current_state": asdict(self.current_state) if self.current_state else None,
            "state_history": [asdict(s) for s in list(self.state_history)[-100:]],
            "cognitive_events": [asdict(e) for e in self.cognitive_events[-50:]],
            "insights": [asdict(i) for i in self.insights],
            "self_awareness_report": self.get_self_awareness_report(),
            "exported_at": datetime.utcnow().isoformat()
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        self.logger.info(f"Exported metacognitive data to {output_path}")


def create_metacognitive_monitor(
    feature_dim: int = 128,
    confidence_threshold: float = 0.7,
    uncertainty_threshold: float = 0.4
) -> MetacognitiveMonitor:
    """
    Factory function to create a configured Metacognitive Monitor.
    
    Args:
        feature_dim: Dimension of model features
        confidence_threshold: Threshold for high confidence
        uncertainty_threshold: Threshold for acceptable uncertainty
    
    Returns:
        Configured MetacognitiveMonitor
    """
    monitor = MetacognitiveMonitor(
        feature_dim=feature_dim,
        confidence_threshold=confidence_threshold,
        uncertainty_threshold=uncertainty_threshold
    )
    
    logging.info("Created Metacognitive Monitoring System")
    
    return monitor
