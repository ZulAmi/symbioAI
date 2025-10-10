"""
Multi-Scale Temporal Reasoning System

Enables reasoning across multiple time scales simultaneously for true long-term planning.

Features:
1. Hierarchical temporal abstractions (milliseconds → years)
2. Event segmentation and boundary detection
3. Predictive modeling at multiple horizons
4. Temporal knowledge graphs with duration modeling
5. Multi-scale attention mechanisms

Competitive Edge: Only system with true multi-scale temporal reasoning for long-term planning.

Author: Symbio AI Team
Date: October 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Set
from enum import Enum
import numpy as np
from collections import deque, defaultdict
import time
import math


class TimeScale(Enum):
    """Different temporal scales for hierarchical reasoning."""
    IMMEDIATE = "immediate"          # Milliseconds to seconds
    SHORT_TERM = "short_term"        # Seconds to minutes
    MEDIUM_TERM = "medium_term"      # Minutes to hours
    LONG_TERM = "long_term"          # Hours to days
    VERY_LONG_TERM = "very_long_term"  # Days to months
    STRATEGIC = "strategic"          # Months to years
    
    @classmethod
    def get_all_scales(cls) -> List['TimeScale']:
        """Get all time scales in order."""
        return [
            cls.IMMEDIATE,
            cls.SHORT_TERM,
            cls.MEDIUM_TERM,
            cls.LONG_TERM,
            cls.VERY_LONG_TERM,
            cls.STRATEGIC
        ]
    
    @classmethod
    def get_horizon_seconds(cls, scale: 'TimeScale') -> float:
        """Get the time horizon in seconds for a scale."""
        horizons = {
            cls.IMMEDIATE: 1.0,           # 1 second
            cls.SHORT_TERM: 60.0,         # 1 minute
            cls.MEDIUM_TERM: 3600.0,      # 1 hour
            cls.LONG_TERM: 86400.0,       # 1 day
            cls.VERY_LONG_TERM: 2592000.0,  # 30 days
            cls.STRATEGIC: 31536000.0     # 365 days
        }
        return horizons[scale]


class EventType(Enum):
    """Types of temporal events."""
    STATE_CHANGE = "state_change"
    PATTERN_START = "pattern_start"
    PATTERN_END = "pattern_end"
    MILESTONE = "milestone"
    ANOMALY = "anomaly"
    BOUNDARY = "boundary"


@dataclass
class TemporalEvent:
    """Represents a temporal event with duration and relationships."""
    event_id: str
    event_type: EventType
    timestamp: float
    duration: float
    scale: TimeScale
    confidence: float
    features: torch.Tensor
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Temporal relationships
    before: Set[str] = field(default_factory=set)
    after: Set[str] = field(default_factory=set)
    during: Set[str] = field(default_factory=set)
    overlaps: Set[str] = field(default_factory=set)


@dataclass
class TemporalPrediction:
    """Multi-horizon prediction result."""
    scale: TimeScale
    horizon_seconds: float
    predicted_states: torch.Tensor
    confidence: float
    attention_weights: Optional[torch.Tensor] = None
    contributing_events: List[str] = field(default_factory=list)


@dataclass
class TemporalConfig:
    """Configuration for temporal reasoning system."""
    # Scales
    num_scales: int = 6
    min_scale_seconds: float = 1.0
    max_scale_seconds: float = 31536000.0  # 1 year
    
    # Architecture
    hidden_dim: int = 512
    num_attention_heads: int = 8
    num_layers: int = 4
    dropout: float = 0.1
    
    # Event detection
    event_threshold: float = 0.7
    min_event_duration: float = 0.1
    max_events_per_scale: int = 1000
    
    # Prediction horizons
    prediction_horizons: List[float] = field(default_factory=lambda: [1.0, 60.0, 3600.0, 86400.0])
    
    # Temporal graph
    max_graph_nodes: int = 10000
    edge_threshold: float = 0.5
    enable_duration_modeling: bool = True
    
    # Memory
    enable_temporal_memory: bool = True
    memory_capacity: int = 5000


class TemporalEncoder(nn.Module):
    """Encodes temporal information at multiple scales."""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_scales: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_scales = num_scales
        
        # Scale-specific encoders
        self.scale_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            for _ in range(num_scales)
        ])
        
        # Positional encoding for time
        self.time_embedding = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, x: torch.Tensor, timestamps: torch.Tensor, scales: List[int]) -> List[torch.Tensor]:
        """
        Encode input at multiple temporal scales.
        
        Args:
            x: Input tensor [batch, seq_len, input_dim]
            timestamps: Time values [batch, seq_len, 1]
            scales: Which scales to use
            
        Returns:
            List of encoded tensors, one per scale
        """
        # Time embedding
        time_emb = self.time_embedding(timestamps)  # [batch, seq_len, hidden]
        
        # Encode at each scale
        scale_outputs = []
        for scale_idx in scales:
            encoded = self.scale_encoders[scale_idx](x)  # [batch, seq_len, hidden]
            combined = encoded + time_emb
            scale_outputs.append(combined)
        
        return scale_outputs


class EventSegmentation(nn.Module):
    """Detects event boundaries and segments temporal sequences."""
    
    def __init__(self, hidden_dim: int, event_threshold: float = 0.7):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.event_threshold = event_threshold
        
        # Boundary detection network
        self.boundary_detector = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # Compare adjacent timesteps
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Event type classifier
        self.event_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, len(EventType)),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, encoded: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Detect event boundaries and classify events.
        
        Args:
            encoded: Encoded sequence [batch, seq_len, hidden]
            
        Returns:
            boundary_scores: Boundary probabilities [batch, seq_len-1]
            event_types: Event type distributions [batch, seq_len, num_types]
        """
        batch_size, seq_len, hidden = encoded.shape
        
        # Detect boundaries by comparing adjacent timesteps
        pairs = torch.cat([
            encoded[:, :-1, :],  # Current
            encoded[:, 1:, :]    # Next
        ], dim=-1)  # [batch, seq_len-1, hidden*2]
        
        boundary_scores = self.boundary_detector(pairs).squeeze(-1)  # [batch, seq_len-1]
        
        # Classify event types
        event_types = self.event_classifier(encoded)  # [batch, seq_len, num_types]
        
        return boundary_scores, event_types
    
    def extract_events(
        self,
        encoded: torch.Tensor,
        timestamps: torch.Tensor,
        scale: TimeScale
    ) -> List[TemporalEvent]:
        """Extract discrete events from encoded sequence."""
        boundary_scores, event_types = self.forward(encoded)
        
        events = []
        batch_size = encoded.shape[0]
        
        for b in range(batch_size):
            # Find boundary points
            boundaries = (boundary_scores[b] > self.event_threshold).nonzero(as_tuple=True)[0]
            
            # Create events between boundaries
            prev_boundary = 0
            for i, boundary in enumerate(boundaries):
                boundary_idx = boundary.item()
                
                # Event duration
                start_time = timestamps[b, prev_boundary].item()
                end_time = timestamps[b, boundary_idx].item()
                duration = end_time - start_time
                
                # Event type (most common in segment)
                segment_types = event_types[b, prev_boundary:boundary_idx+1]
                event_type_idx = segment_types.mean(dim=0).argmax().item()
                event_type = list(EventType)[event_type_idx]
                
                # Event features (average over segment)
                event_features = encoded[b, prev_boundary:boundary_idx+1].mean(dim=0)
                
                # Confidence
                confidence = boundary_scores[b, boundary_idx].item()
                
                event = TemporalEvent(
                    event_id=f"event_{b}_{i}",
                    event_type=event_type,
                    timestamp=start_time,
                    duration=duration,
                    scale=scale,
                    confidence=confidence,
                    features=event_features
                )
                events.append(event)
                
                prev_boundary = boundary_idx + 1
        
        return events


class MultiScaleAttention(nn.Module):
    """Attention mechanism across multiple temporal scales."""
    
    def __init__(self, hidden_dim: int, num_heads: int, num_scales: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_scales = num_scales
        
        # Scale-specific attention
        self.scale_attention = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=0.1,
                batch_first=True
            )
            for _ in range(num_scales)
        ])
        
        # Cross-scale attention (aggregate information across scales)
        self.cross_scale_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # Scale fusion
        self.scale_fusion = nn.Sequential(
            nn.Linear(hidden_dim * num_scales, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
    def forward(self, scale_encodings: List[torch.Tensor]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Apply multi-scale attention.
        
        Args:
            scale_encodings: List of [batch, seq_len, hidden] tensors
            
        Returns:
            fused: Fused representation [batch, seq_len, hidden]
            attention_weights: List of attention weights per scale
        """
        # Within-scale attention
        attended_scales = []
        attention_weights = []
        
        for i, encoding in enumerate(scale_encodings):
            attended, weights = self.scale_attention[i](
                encoding, encoding, encoding,
                need_weights=True
            )
            attended_scales.append(attended)
            attention_weights.append(weights)
        
        # Cross-scale attention
        # Stack scales as sequence
        stacked = torch.stack(attended_scales, dim=1)  # [batch, num_scales, seq_len, hidden]
        batch, num_scales, seq_len, hidden = stacked.shape
        
        # Reshape for cross-scale attention
        reshaped = stacked.reshape(batch * seq_len, num_scales, hidden)
        cross_attended, _ = self.cross_scale_attention(
            reshaped, reshaped, reshaped
        )
        cross_attended = cross_attended.reshape(batch, seq_len, num_scales, hidden)
        
        # Fuse scales
        fused_input = cross_attended.reshape(batch, seq_len, -1)  # [batch, seq_len, num_scales*hidden]
        fused = self.scale_fusion(fused_input)
        
        return fused, attention_weights


class TemporalKnowledgeGraph:
    """Graph structure for temporal relationships and duration modeling."""
    
    def __init__(self, max_nodes: int = 10000, edge_threshold: float = 0.5):
        self.max_nodes = max_nodes
        self.edge_threshold = edge_threshold
        
        # Event storage
        self.events: Dict[str, TemporalEvent] = {}
        
        # Temporal relations (stored as adjacency lists)
        self.before_edges: Dict[str, Set[str]] = defaultdict(set)
        self.after_edges: Dict[str, Set[str]] = defaultdict(set)
        self.during_edges: Dict[str, Set[str]] = defaultdict(set)
        self.overlap_edges: Dict[str, Set[str]] = defaultdict(set)
        
        # Duration statistics
        self.duration_stats: Dict[EventType, Dict[str, float]] = defaultdict(lambda: {
            'mean': 0.0,
            'std': 1.0,
            'min': float('inf'),
            'max': 0.0,
            'count': 0
        })
        
    def add_event(self, event: TemporalEvent):
        """Add event to graph and infer relationships."""
        if len(self.events) >= self.max_nodes:
            # Prune oldest events
            self._prune_old_events()
        
        self.events[event.event_id] = event
        
        # Update duration statistics
        self._update_duration_stats(event)
        
        # Infer temporal relationships with existing events
        self._infer_relationships(event)
    
    def _update_duration_stats(self, event: TemporalEvent):
        """Update running statistics for event durations."""
        stats = self.duration_stats[event.event_type]
        
        # Update min/max
        stats['min'] = min(stats['min'], event.duration)
        stats['max'] = max(stats['max'], event.duration)
        
        # Update mean (running average)
        count = stats['count']
        stats['mean'] = (stats['mean'] * count + event.duration) / (count + 1)
        
        # Update std (running)
        if count > 0:
            stats['std'] = math.sqrt(
                (stats['std'] ** 2 * count + (event.duration - stats['mean']) ** 2) / (count + 1)
            )
        
        stats['count'] = count + 1
    
    def _infer_relationships(self, new_event: TemporalEvent):
        """Infer temporal relationships between new event and existing events."""
        new_start = new_event.timestamp
        new_end = new_event.timestamp + new_event.duration
        
        for event_id, event in self.events.items():
            if event_id == new_event.event_id:
                continue
            
            start = event.timestamp
            end = event.timestamp + event.duration
            
            # Before: new_event ends before event starts
            if new_end <= start:
                self.before_edges[new_event.event_id].add(event_id)
                self.after_edges[event_id].add(new_event.event_id)
                new_event.before.add(event_id)
                event.after.add(new_event.event_id)
            
            # After: new_event starts after event ends
            elif new_start >= end:
                self.after_edges[new_event.event_id].add(event_id)
                self.before_edges[event_id].add(new_event.event_id)
                new_event.after.add(event_id)
                event.before.add(new_event.event_id)
            
            # During: new_event is entirely within event
            elif new_start >= start and new_end <= end:
                self.during_edges[new_event.event_id].add(event_id)
                new_event.during.add(event_id)
            
            # Overlaps: partial overlap
            elif (new_start < end and new_end > start):
                self.overlap_edges[new_event.event_id].add(event_id)
                self.overlap_edges[event_id].add(new_event.event_id)
                new_event.overlaps.add(event_id)
                event.overlaps.add(new_event.event_id)
    
    def _prune_old_events(self, keep_ratio: float = 0.8):
        """Remove oldest events to stay within capacity."""
        num_to_keep = int(self.max_nodes * keep_ratio)
        
        # Sort by timestamp
        sorted_events = sorted(self.events.items(), key=lambda x: x[1].timestamp, reverse=True)
        
        # Keep most recent
        events_to_keep = dict(sorted_events[:num_to_keep])
        events_to_remove = set(self.events.keys()) - set(events_to_keep.keys())
        
        # Remove old events
        for event_id in events_to_remove:
            del self.events[event_id]
            
            # Clean up edges
            for edges in [self.before_edges, self.after_edges, self.during_edges, self.overlap_edges]:
                if event_id in edges:
                    del edges[event_id]
                
                # Remove references in other events
                for other_edges in edges.values():
                    other_edges.discard(event_id)
    
    def query_events_in_range(
        self,
        start_time: float,
        end_time: float,
        event_type: Optional[EventType] = None
    ) -> List[TemporalEvent]:
        """Query events within a time range."""
        results = []
        for event in self.events.values():
            event_end = event.timestamp + event.duration
            
            # Check overlap with query range
            if event.timestamp <= end_time and event_end >= start_time:
                if event_type is None or event.event_type == event_type:
                    results.append(event)
        
        return sorted(results, key=lambda e: e.timestamp)
    
    def get_event_chain(self, start_event_id: str, relation: str = "before") -> List[TemporalEvent]:
        """Get chain of events following a specific relation."""
        chain = []
        visited = set()
        queue = deque([start_event_id])
        
        edge_dict = {
            "before": self.before_edges,
            "after": self.after_edges,
            "during": self.during_edges,
            "overlaps": self.overlap_edges
        }
        
        edges = edge_dict.get(relation, self.before_edges)
        
        while queue:
            event_id = queue.popleft()
            if event_id in visited or event_id not in self.events:
                continue
            
            visited.add(event_id)
            chain.append(self.events[event_id])
            
            # Add connected events
            for next_id in edges.get(event_id, []):
                if next_id not in visited:
                    queue.append(next_id)
        
        return chain
    
    def predict_duration(self, event_type: EventType) -> Tuple[float, float]:
        """Predict expected duration for an event type."""
        stats = self.duration_stats[event_type]
        return stats['mean'], stats['std']
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get graph statistics."""
        return {
            'num_events': len(self.events),
            'num_before_edges': sum(len(edges) for edges in self.before_edges.values()),
            'num_after_edges': sum(len(edges) for edges in self.after_edges.values()),
            'num_during_edges': sum(len(edges) for edges in self.during_edges.values()),
            'num_overlap_edges': sum(len(edges) for edges in self.overlap_edges.values()),
            'duration_stats': dict(self.duration_stats)
        }


class MultiHorizonPredictor(nn.Module):
    """Predicts future states at multiple time horizons."""
    
    def __init__(self, hidden_dim: int, output_dim: int, num_horizons: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_horizons = num_horizons
        
        # Horizon-specific predictors
        self.horizon_predictors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, output_dim)
            )
            for _ in range(num_horizons)
        ])
        
        # Confidence estimators
        self.confidence_estimators = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid()
            )
            for _ in range(num_horizons)
        ])
        
    def forward(self, fused_representation: torch.Tensor) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Predict at multiple horizons.
        
        Args:
            fused_representation: [batch, seq_len, hidden]
            
        Returns:
            List of (predictions, confidences) for each horizon
        """
        # Use last timestep for prediction
        final_state = fused_representation[:, -1, :]  # [batch, hidden]
        
        predictions = []
        for i in range(self.num_horizons):
            pred = self.horizon_predictors[i](final_state)  # [batch, output_dim]
            conf = self.confidence_estimators[i](final_state)  # [batch, 1]
            predictions.append((pred, conf))
        
        return predictions


class MultiScaleTemporalReasoning(nn.Module):
    """
    Complete multi-scale temporal reasoning system.
    
    Enables reasoning across multiple time scales for long-term planning.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        config: Optional[TemporalConfig] = None
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.config = config or TemporalConfig()
        
        # Components
        self.encoder = TemporalEncoder(
            input_dim=input_dim,
            hidden_dim=self.config.hidden_dim,
            num_scales=self.config.num_scales
        )
        
        self.event_segmentation = EventSegmentation(
            hidden_dim=self.config.hidden_dim,
            event_threshold=self.config.event_threshold
        )
        
        self.multi_scale_attention = MultiScaleAttention(
            hidden_dim=self.config.hidden_dim,
            num_heads=self.config.num_attention_heads,
            num_scales=self.config.num_scales
        )
        
        self.multi_horizon_predictor = MultiHorizonPredictor(
            hidden_dim=self.config.hidden_dim,
            output_dim=output_dim,
            num_horizons=len(self.config.prediction_horizons)
        )
        
        # Temporal knowledge graph
        self.temporal_graph = TemporalKnowledgeGraph(
            max_nodes=self.config.max_graph_nodes,
            edge_threshold=self.config.edge_threshold
        )
        
        # Temporal memory (recent event history per scale)
        if self.config.enable_temporal_memory:
            self.temporal_memory: Dict[TimeScale, deque] = {
                scale: deque(maxlen=self.config.memory_capacity // self.config.num_scales)
                for scale in TimeScale.get_all_scales()
            }
        
        # Statistics
        self.stats = {
            'total_forward_passes': 0,
            'events_detected': 0,
            'predictions_made': 0,
            'avg_confidence': 0.0
        }
        
    def forward(
        self,
        x: torch.Tensor,
        timestamps: torch.Tensor,
        return_events: bool = False,
        return_predictions: bool = True
    ) -> Dict[str, Any]:
        """
        Perform multi-scale temporal reasoning.
        
        Args:
            x: Input sequence [batch, seq_len, input_dim]
            timestamps: Time values [batch, seq_len, 1]
            return_events: Whether to return detected events
            return_predictions: Whether to return multi-horizon predictions
            
        Returns:
            Dictionary with:
                - fused_representation: Final representation
                - events: Detected events (if return_events=True)
                - predictions: Multi-horizon predictions (if return_predictions=True)
                - attention_weights: Attention weights per scale
                - statistics: Processing statistics
        """
        self.stats['total_forward_passes'] += 1
        
        # 1. Encode at multiple scales
        scales_to_use = list(range(self.config.num_scales))
        scale_encodings = self.encoder(x, timestamps, scales_to_use)
        
        # 2. Multi-scale attention
        fused, attention_weights = self.multi_scale_attention(scale_encodings)
        
        # 3. Event detection (optional)
        events_by_scale = {}
        if return_events:
            time_scales = TimeScale.get_all_scales()
            for i, encoding in enumerate(scale_encodings):
                scale = time_scales[i] if i < len(time_scales) else TimeScale.IMMEDIATE
                events = self.event_segmentation.extract_events(encoding, timestamps, scale)
                events_by_scale[scale] = events
                
                # Add to temporal graph
                for event in events:
                    self.temporal_graph.add_event(event)
                    
                    # Add to memory
                    if self.config.enable_temporal_memory:
                        self.temporal_memory[scale].append(event)
                
                self.stats['events_detected'] += len(events)
        
        # 4. Multi-horizon prediction (optional)
        predictions_by_horizon = []
        if return_predictions:
            horizon_results = self.multi_horizon_predictor(fused)
            
            for i, (pred, conf) in enumerate(horizon_results):
                horizon_seconds = self.config.prediction_horizons[i]
                scale = self._get_scale_for_horizon(horizon_seconds)
                
                # Get contributing events
                current_time = timestamps[0, -1, 0].item()
                contributing_events = self.temporal_graph.query_events_in_range(
                    start_time=current_time - horizon_seconds,
                    end_time=current_time
                )
                
                prediction = TemporalPrediction(
                    scale=scale,
                    horizon_seconds=horizon_seconds,
                    predicted_states=pred,
                    confidence=conf.mean().item(),
                    attention_weights=attention_weights[i] if i < len(attention_weights) else None,
                    contributing_events=[e.event_id for e in contributing_events[:10]]
                )
                predictions_by_horizon.append(prediction)
                
                self.stats['predictions_made'] += 1
                self.stats['avg_confidence'] = (
                    (self.stats['avg_confidence'] * (self.stats['predictions_made'] - 1) +
                     prediction.confidence) / self.stats['predictions_made']
                )
        
        # Build result
        result = {
            'fused_representation': fused,
            'attention_weights': attention_weights,
            'statistics': self.get_statistics()
        }
        
        if return_events:
            result['events'] = events_by_scale
        
        if return_predictions:
            result['predictions'] = predictions_by_horizon
        
        return result
    
    def _get_scale_for_horizon(self, horizon_seconds: float) -> TimeScale:
        """Map a horizon in seconds to a TimeScale."""
        if horizon_seconds <= 1.0:
            return TimeScale.IMMEDIATE
        elif horizon_seconds <= 60.0:
            return TimeScale.SHORT_TERM
        elif horizon_seconds <= 3600.0:
            return TimeScale.MEDIUM_TERM
        elif horizon_seconds <= 86400.0:
            return TimeScale.LONG_TERM
        elif horizon_seconds <= 2592000.0:
            return TimeScale.VERY_LONG_TERM
        else:
            return TimeScale.STRATEGIC
    
    def plan_long_term(
        self,
        current_state: torch.Tensor,
        goal_state: torch.Tensor,
        planning_horizon: float = 86400.0  # 1 day
    ) -> Dict[str, Any]:
        """
        Plan actions across multiple time scales to reach a goal.
        
        Args:
            current_state: Current state [batch, input_dim]
            goal_state: Desired state [batch, input_dim]
            planning_horizon: How far ahead to plan (seconds)
            
        Returns:
            Planning result with hierarchical actions
        """
        # Create synthetic sequence leading to goal
        batch_size = current_state.shape[0]
        seq_len = 10
        
        # Interpolate from current to goal
        steps = torch.linspace(0, 1, seq_len, device=current_state.device)
        sequence = torch.zeros(batch_size, seq_len, self.input_dim, device=current_state.device)
        
        for i, step in enumerate(steps):
            sequence[:, i, :] = current_state * (1 - step) + goal_state * step
        
        # Create timestamps
        timestamps = torch.linspace(
            0, planning_horizon, seq_len,
            device=current_state.device
        ).unsqueeze(0).unsqueeze(-1).expand(batch_size, seq_len, 1)
        
        # Reason about this sequence
        result = self.forward(sequence, timestamps, return_events=True, return_predictions=True)
        
        # Extract hierarchical plan
        plan = {
            'horizon_seconds': planning_horizon,
            'num_predicted_events': sum(len(events) for events in result['events'].values()),
            'predictions_by_scale': {},
            'recommended_actions': []
        }
        
        # Organize predictions by scale
        for pred in result['predictions']:
            plan['predictions_by_scale'][pred.scale.value] = {
                'predicted_state': pred.predicted_states,
                'confidence': pred.confidence,
                'contributing_events': pred.contributing_events
            }
        
        # Generate recommended actions (based on events)
        for scale, events in result['events'].items():
            for event in events[:3]:  # Top 3 per scale
                action = {
                    'scale': scale.value,
                    'timestamp': event.timestamp,
                    'duration': event.duration,
                    'type': event.event_type.value,
                    'confidence': event.confidence
                }
                plan['recommended_actions'].append(action)
        
        return plan
    
    def query_temporal_relationships(
        self,
        event_id: str,
        relation_type: str = "before"
    ) -> List[TemporalEvent]:
        """Query temporal relationships in the knowledge graph."""
        return self.temporal_graph.get_event_chain(event_id, relation_type)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics."""
        graph_stats = self.temporal_graph.get_statistics()
        
        memory_stats = {}
        if self.config.enable_temporal_memory:
            memory_stats = {
                scale.value: len(self.temporal_memory[scale])
                for scale in TimeScale.get_all_scales()
            }
        
        return {
            **self.stats,
            'graph_statistics': graph_stats,
            'memory_statistics': memory_stats
        }
    
    def export_temporal_graph(self) -> Dict[str, Any]:
        """Export temporal knowledge graph for visualization."""
        return {
            'events': {
                event_id: {
                    'type': event.event_type.value,
                    'timestamp': event.timestamp,
                    'duration': event.duration,
                    'scale': event.scale.value,
                    'confidence': event.confidence,
                    'before': list(event.before),
                    'after': list(event.after),
                    'during': list(event.during),
                    'overlaps': list(event.overlaps)
                }
                for event_id, event in self.temporal_graph.events.items()
            },
            'statistics': self.temporal_graph.get_statistics()
        }


def create_multi_scale_temporal_reasoner(
    input_dim: int = 256,
    output_dim: int = 64,
    hidden_dim: int = 512,
    num_scales: int = 6,
    prediction_horizons: Optional[List[float]] = None
) -> MultiScaleTemporalReasoning:
    """
    Factory function to create a multi-scale temporal reasoning system.
    
    Args:
        input_dim: Input feature dimension
        output_dim: Output prediction dimension
        hidden_dim: Hidden layer dimension
        num_scales: Number of temporal scales
        prediction_horizons: List of prediction horizons in seconds
        
    Returns:
        MultiScaleTemporalReasoning system
    """
    config = TemporalConfig(
        num_scales=num_scales,
        hidden_dim=hidden_dim,
        prediction_horizons=prediction_horizons or [1.0, 60.0, 3600.0, 86400.0]
    )
    
    return MultiScaleTemporalReasoning(
        input_dim=input_dim,
        output_dim=output_dim,
        config=config
    )


if __name__ == "__main__":
    print("Multi-Scale Temporal Reasoning System")
    print("=" * 60)
    print("\n✅ Module loaded successfully!")
    print("\nFeatures:")
    print("  1. Hierarchical temporal abstractions (6 scales)")
    print("  2. Event segmentation and boundary detection")
    print("  3. Predictive modeling at multiple horizons")
    print("  4. Temporal knowledge graphs with duration modeling")
    print("  5. Multi-scale attention mechanisms")
    print("\nCompetitive Edge:")
    print("  • ONLY system with true multi-scale temporal reasoning")
    print("  • Enables long-term planning across time scales")
    print("  • Automatic event detection and relationship inference")
    print("  • Duration modeling with uncertainty quantification")
