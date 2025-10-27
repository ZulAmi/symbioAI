"""
training/derpp_causal.py - Causal Extension of Official Mammoth DER++
======================================================================

This module EXTENDS the official Mammoth DER++ implementation with causal
inference capabilities. It inherits from the official derpp.py and adds:

1. Causal graph learning between tasks
2. ATE-based importance sampling
3. Causal forgetting diagnostics

Base implementation: mammoth/models/derpp.py (official Mammoth)
Extension: Structural Causal Models (Pearl, 2009)

Author: [Your Name]
Date: October 23, 2025
"""

import sys
from pathlib import Path
import torch
import torch.nn.functional as F
from typing import Optional

# Add mammoth to path to import official DER++
mammoth_path = Path(__file__).parent.parent / 'mammoth'
if str(mammoth_path) not in sys.path:
    sys.path.insert(0, str(mammoth_path))

# Import official DER++ base class
from models.derpp import Derpp

# Import our causal inference modules
try:
    from training.causal_inference import StructuralCausalModel, CausalForgettingDetector
    from training.metrics_tracker import ContinualLearningMetrics
except ImportError:
    StructuralCausalModel = None
    CausalForgettingDetector = None
    ContinualLearningMetrics = None
    print("[WARNING] Causal inference modules not found. Running in base DER++ mode.")


class DerppCausal(Derpp):
    """
    Causal extension of official Mammoth DER++.
    
    Inherits all functionality from official DER++ and adds:
    - Phase 1: Official DER++ baseline (inherited)
    - Phase 2: Causal graph learning between tasks
    - Phase 3: ATE-based importance sampling
    - Phase 4: Causal forgetting diagnostics
    
    Usage:
        Same as official DER++, with optional causal flags:
        --enable_causal_graph_learning 1
        --use_causal_sampling 1
    """
    
    NAME = 'derpp-causal'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']
    
    @staticmethod
    def get_parser(parser):
        """Add causal-specific arguments to official DER++ parser."""
        # Get official DER++ arguments first (alpha, beta, buffer_size, etc.)
        parser = Derpp.get_parser(parser)
        
        # Add causal-specific arguments (matching your test scripts)
        parser.add_argument('--enable_causal_graph_learning', type=int, default=0,
                          help='Enable causal graph learning between tasks (0=off, 1=on)')
        parser.add_argument('--use_causal_sampling', type=int, default=0,
                          help='Use ATE-based importance sampling (0=off, 1=on)')
        parser.add_argument('--temperature', type=float, default=2.0,
                          help='Temperature for KL divergence (if used instead of MSE)')
        parser.add_argument('--num_tasks', type=int, default=10,
                          help='Total number of tasks (for causal graph)')
        parser.add_argument('--feature_dim', type=int, default=512,
                          help='Feature dimensionality for causal discovery')
        parser.add_argument('--causal_cache_size', type=int, default=200,
                          help='Max samples per task for causal graph learning')
        parser.add_argument('--importance_weight', type=float, default=0.5,
                          help='Weight for importance sampling (Phase 2)')
        parser.add_argument('--use_importance_sampling', type=int, default=0,
                          help='Enable importance sampling (0=off, 1=on)')
        
        # Causal-specific tuning parameters
        parser.add_argument('--causal_blend_ratio', type=float, default=0.7,
                          help='Causal vs recency blend (0.7 = 70%% causal, 30%% recency)')
        parser.add_argument('--sparsity_start', type=float, default=0.9,
                          help='Initial sparsification quantile (keep top X%%)')
        parser.add_argument('--sparsity_end', type=float, default=0.7,
                          help='Final sparsification quantile')
        parser.add_argument('--causal_warmup_tasks', type=int, default=3,
                          help='Number of tasks to warm up causal sampling (0=immediate)')
        
        return parser
    
    def __init__(self, backbone, loss, args, transform, dataset=None):
        """Initialize with official DER++ + causal extensions."""
        # Initialize official DER++ (inherits buffer, optimizer, etc.)
        super().__init__(backbone, loss, args, transform, dataset=dataset)
        
        # Causal extensions (with defaults matching test scripts)
        self.enable_causal_graph = getattr(args, 'enable_causal_graph_learning', 0)
        self.use_causal_sampling = getattr(args, 'use_causal_sampling', 0)
        self.use_importance_sampling = getattr(args, 'use_importance_sampling', 0)
        self.temperature = getattr(args, 'temperature', 2.0)
        self.num_tasks = getattr(args, 'num_tasks', 10)
        self.feature_dim = getattr(args, 'feature_dim', 512)
        self.causal_cache_size = getattr(args, 'causal_cache_size', 200)
        self.importance_weight = getattr(args, 'importance_weight', 0.5)
        
        # Causal-specific tuning parameters
        self.causal_blend_ratio = getattr(args, 'causal_blend_ratio', 0.7)
        self.sparsity_start = getattr(args, 'sparsity_start', 0.9)
        self.sparsity_end = getattr(args, 'sparsity_end', 0.7)
        self.causal_warmup_tasks = getattr(args, 'causal_warmup_tasks', 3)
        
        # Initialize Structural Causal Model if enabled
        self.scm = None
        self.causal_graph = None
        self.task_feature_cache = {}
        # Note: current_task is inherited from ContinualModel (read-only property)
        
        if self.enable_causal_graph and StructuralCausalModel is not None:
            self.scm = StructuralCausalModel(
                num_tasks=self.num_tasks,
                feature_dim=self.feature_dim
            )
            print(f"[DER++ Causal] ✅ Causal graph learning ENABLED")
            print(f"[DER++ Causal]    - Tasks: {self.num_tasks}")
            print(f"[DER++ Causal]    - Feature dim: {self.feature_dim}D")
            print(f"[DER++ Causal]    - Cache size: {self.causal_cache_size} samples/task")
            print(f"[DER++ Causal]    - Temperature: {self.temperature}")
        else:
            print(f"[DER++ Causal] Running in official DER++ mode (no causality)")
        
        # Metrics tracker
        self.metrics_tracker = ContinualLearningMetrics(num_tasks=self.num_tasks) if ContinualLearningMetrics else None
    
    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        """
        Training step with causal-weighted replay sampling.
        
        Phase 1 (causal OFF): Use official DER++ exactly
        Phase 2 (causal ON): Use causal graph to weight replay samples
        """
        # Cache features for causal graph learning
        if self.enable_causal_graph and self.scm is not None:
            self._cache_features_for_causal_learning(inputs, labels, not_aug_inputs)
        
        # If causal graph not available or disabled, use official DER++
        if not self.enable_causal_graph or self.causal_graph is None or not self.use_causal_sampling:
            return super().observe(inputs, labels, not_aug_inputs, epoch)
        
        # Otherwise, implement DER++ with causal-weighted replay
        self.opt.zero_grad()
        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)
        
        if not self.buffer.is_empty():
            # Get causal-weighted samples from buffer
            buf_inputs, buf_labels, buf_logits = self._get_causal_weighted_samples(
                self.args.minibatch_size
            )
            
            if buf_inputs is not None:
                # MSE loss on replayed logits (knowledge distillation)
                buf_outputs = self.net(buf_inputs)
                loss_mse = self.args.alpha * F.mse_loss(buf_outputs, buf_logits)
                loss += loss_mse
                
                # CE loss on replayed labels
                loss_ce = self.args.beta * self.loss(buf_outputs, buf_labels)
                loss += loss_ce
        
        loss.backward()
        self.opt.step()
        
        # Store in buffer (same as official DER++)
        self.buffer.add_data(
            examples=not_aug_inputs,
            labels=labels,
            logits=outputs.data
        )
        
        return loss.item()
    
    def _cache_features_for_causal_learning(self, inputs, labels, not_aug_inputs):
        """
        Cache features for causal graph learning (happens during training).
        
        This does NOT affect training - it's just bookkeeping for later analysis.
        """
        task_id = self.current_task  # Read from inherited property
        
        if task_id not in self.task_feature_cache:
            self.task_feature_cache[task_id] = {
                'features': [],
                'labels': [],
                'inputs': []
            }
        
        cache = self.task_feature_cache[task_id]
        
        # Only cache up to max size
        if len(cache['features']) < self.causal_cache_size:
            # Extract features from penultimate layer
            with torch.no_grad():
                try:
                    features = self.net(not_aug_inputs, returnt='features')
                except TypeError:
                    # Fallback if returnt not supported
                    features = self.net(not_aug_inputs)
                
                # Global average pooling if needed
                if features.dim() > 2:
                    features = features.mean(dim=[-1, -2])
                
                # Store one random sample from batch
                batch_size = inputs.size(0)
                idx = torch.randint(0, batch_size, (1,)).item()
                
                cache['features'].append(features[idx].cpu())
                cache['labels'].append(labels[idx].cpu())
                cache['inputs'].append(not_aug_inputs[idx].cpu())
    
    def _get_causal_weighted_samples(self, size):
        """
        Sample from buffer using causal graph weights with optimizations.
        
        QUICK WIN OPTIMIZATIONS:
        1. Warm start blending: Gradual transition from uniform to causal sampling
        2. Smoother importance scores: Blend causal strength with recency
        3. Adaptive sparsification: Consider more edges early, fewer later
        
        Samples are weighted by their causal importance based on the learned graph.
        Higher weight = more important for preventing forgetting.
        
        Args:
            size: Number of samples to retrieve
            
        Returns:
            (inputs, labels, logits) tuple
        """
        if self.buffer.is_empty():
            return None, None, None
        
        # Get all buffer data
        buf_inputs, buf_labels, buf_logits = self.buffer.get_data(
            self.buffer.buffer_size,
            transform=self.transform,
            device=self.device
        )
        
        if buf_inputs is None:
            return None, None, None
        
        # If no causal graph yet, use uniform sampling
        if self.causal_graph is None:
            indices = torch.randperm(buf_inputs.size(0))[:size]
            return buf_inputs[indices], buf_labels[indices], buf_logits[indices]
        
        # Warm start blending with tunable warmup schedule
        # Gradually transition from uniform to full causal based on warmup_tasks parameter
        current_task = self.current_task
        
        if self.causal_warmup_tasks == 0:
            # No warmup - use causal immediately
            causal_blend = 1.0
        elif current_task < self.causal_warmup_tasks:
            # Gradual warmup over specified number of tasks
            causal_blend = current_task / self.causal_warmup_tasks
        else:
            # Full causal after warmup period
            causal_blend = 1.0
        
        # Compute causal importance for each sample
        # Higher importance = more critical for preventing forgetting
        importance_scores = torch.zeros(buf_inputs.size(0), device=self.device)
        
        # Get task labels from buffer (assumes buffer stores task info)
        # Fallback: estimate from label ranges if task info not available
        if hasattr(self.buffer, 'task_labels'):
            buf_task_labels = self.buffer.task_labels[:buf_inputs.size(0)]
        else:
            # Estimate task from class label (10 classes per task for CIFAR-100)
            buf_task_labels = buf_labels // self.cpt
        
        # Weight samples by causal influence on current task
        for i, task_label in enumerate(buf_task_labels):
            task_id = int(task_label.item()) if torch.is_tensor(task_label) else int(task_label)
            
            if task_id < current_task and task_id < self.causal_graph.size(0):
                # Causal effect of past task on current task
                causal_strength = abs(self.causal_graph[task_id, current_task].item())
                
                # Blend causal strength with recency (newer tasks get slight boost)
                # Using tunable blend ratio instead of hardcoded 0.7
                recency_weight = 1.0 - (current_task - task_id) / max(current_task, 1)
                blended_importance = self.causal_blend_ratio * causal_strength + (1 - self.causal_blend_ratio) * recency_weight
                
                importance_scores[i] = blended_importance
            else:
                # No causal relationship or future task
                # Use recency as fallback instead of fixed 0.1
                if task_id < current_task:
                    recency_weight = 1.0 - (current_task - task_id) / max(current_task, 1)
                    importance_scores[i] = 0.3 * recency_weight  # Lower than causal-guided
                else:
                    importance_scores[i] = 0.1  # Small baseline importance
        
        # QUICK WIN #1 (continued): Blend causal importance with uniform
        uniform_probs = torch.ones_like(importance_scores) / importance_scores.size(0)
        
        # Normalize causal importance to probabilities
        if importance_scores.sum() > 0:
            causal_probs = importance_scores / importance_scores.sum()
        else:
            causal_probs = uniform_probs
        
        # Blend uniform and causal based on warm start factor
        sampling_probs = (1 - causal_blend) * uniform_probs + causal_blend * causal_probs
        
        # Sample according to blended importance
        try:
            indices = torch.multinomial(
                sampling_probs,
                num_samples=min(size, buf_inputs.size(0)),
                replacement=False
            )
        except RuntimeError:
            # If multinomial fails (e.g., all zeros), fall back to uniform
            indices = torch.randperm(buf_inputs.size(0))[:size]
        
        return buf_inputs[indices], buf_labels[indices], buf_logits[indices]
    
    def end_task(self, dataset):
        """
        Called at end of each task.
        
        Official DER++ part: inherited (no-op in base class)
        Causal extensions: Learn causal graph, update metrics
        """
        task_id = self.current_task  # Read from inherited property
        print(f"\n[DER++ Causal] End of Task {task_id}")
        print(f"  Buffer: {len(self.buffer)}/{self.buffer.buffer_size} samples")
        
        # Learn causal graph if enabled and we have enough tasks
        if self.enable_causal_graph and self.scm is not None and task_id >= 1:
            self._learn_causal_graph()
        
        # Update metrics tracker
        if self.metrics_tracker:
            self.metrics_tracker.update_current_task(task_id)
        
        # Call parent end_task (if it exists)
        if hasattr(super(), 'end_task'):
            super().end_task(dataset)
    
    def _learn_causal_graph(self):
        """
        Learn causal graph between tasks using cached features.
        
        This happens AFTER training, so it doesn't affect DER++ performance.
        
        QUICK WIN #3: Adaptive sparsification threshold
        - Early tasks: Keep more edges (0.9 quantile = top 10%)
        - Later tasks: Tighten to strongest edges (0.7 quantile = top 30%)
        """
        task_id = self.current_task  # Read from inherited property
        
        print(f"\n[Phase 2] Learning causal graph between tasks...")
        print(f"  Tasks completed: {task_id + 1}")
        print(f"  Cached task data: {list(self.task_feature_cache.keys())}")
        
        # Check if we have enough cached data
        valid_tasks = [
            tid for tid, cache in self.task_feature_cache.items()
            if len(cache['features']) >= 10
        ]
        
        if len(valid_tasks) < 2:
            print(f"  ⚠️ Not enough data yet (need ≥2 tasks with ≥10 samples)")
            return
        
        # Convert cached features to format expected by SCM
        task_data = {}
        task_labels = {}
        
        for tid in valid_tasks:
            cache = self.task_feature_cache[tid]
            if len(cache['features']) > 0:
                task_data[tid] = torch.stack(cache['features'])
                task_labels[tid] = torch.stack(cache['labels'])
        
        print(f"  Learning from {len(task_data)} tasks with sufficient data:")
        for tid, data in task_data.items():
            print(f"    Task {tid}: {data.shape[0]} samples, {data.shape[1]}D features")
        
        # Adaptive sparsification using tunable start/end thresholds
        # Gradually transition from sparsity_start to sparsity_end as tasks progress
        task_progress = task_id / max(self.num_tasks - 1, 1)
        adaptive_quantile = self.sparsity_start - (self.sparsity_start - self.sparsity_end) * task_progress
        adaptive_quantile = max(0.5, min(0.95, adaptive_quantile))  # Clamp to safe range [0.5, 0.95]
        print(f"  Using adaptive sparsification: {adaptive_quantile:.2f} quantile (start={self.sparsity_start:.2f}, end={self.sparsity_end:.2f})")
        
        try:
            # Learn causal structure using SCM with adaptive threshold
            self.causal_graph = self.scm.learn_causal_structure(
                task_data,
                task_labels,
                self.net,
                sparsification_quantile=adaptive_quantile
            )
            
            if self.causal_graph is not None:
                self._print_causal_graph_summary()
            else:
                print(f"  ⚠️ Causal graph learning returned None")
                
        except Exception as e:
            print(f"  ⚠️ Causal graph learning failed: {e}")
            import traceback
            traceback.print_exc()
    
    def _print_causal_graph_summary(self):
        """Print summary of learned causal graph."""
        print(f"  ✅ Causal graph learned successfully!")
        print(f"  Graph shape: {self.causal_graph.shape}")
        
        # Analyze strong causal dependencies (>0.5 strength)
        strong_edges = (self.causal_graph.abs() > 0.5).nonzero(as_tuple=False)
        print(f"  Strong causal dependencies ({len(strong_edges)} edges with |strength| > 0.5):")
        
        if len(strong_edges) > 0:
            for edge in strong_edges[:10]:  # Show first 10
                source, target = int(edge[0]), int(edge[1])
                strength = float(self.causal_graph[source, target])
                direction = "→" if strength > 0 else "↛"
                print(f"    Task {source} {direction} Task {target}: {strength:.3f}")
            
            if len(strong_edges) > 10:
                print(f"    ... and {len(strong_edges) - 10} more edges")
        else:
            print(f"    No strong dependencies found (all edges < 0.5)")
        
        # Compute graph statistics
        num_edges = (self.causal_graph.abs() > 0.1).sum().item()
        graph_density = num_edges / (self.num_tasks * (self.num_tasks - 1)) if self.num_tasks > 1 else 0
        mean_strength = self.causal_graph.abs().mean().item()
        max_strength = self.causal_graph.abs().max().item()
        
        print(f"  Graph statistics:")
        print(f"    Total edges (>0.1): {num_edges}")
        print(f"    Graph density: {graph_density:.2%}")
        print(f"    Mean |strength|: {mean_strength:.3f}")
        print(f"    Max |strength|: {max_strength:.3f}")


# For backward compatibility with existing code
DERPPCausal = DerppCausal
