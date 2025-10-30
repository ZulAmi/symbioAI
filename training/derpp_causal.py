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

Author: Zulhilmi Rahmat 
Date: October 23, 2025
"""

import sys
from pathlib import Path
import torch
import torch.nn.functional as F
from typing import Optional
import logging

# Setup logger
logger = logging.getLogger(__name__)

# Add repo root to path so we can import the Mammoth package normally
repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

# Import official DER++ base class
from mammoth.models.derpp import Derpp

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
        
        # Causal-specific tuning parameters (FIXED defaults for better performance)
        parser.add_argument('--causal_blend_ratio', type=float, default=0.3,
                          help='Weight for causal vs uniform sampling (0.3 = 30%% causal, 70%% uniform - safer default)')
        parser.add_argument('--sparsity_start', type=float, default=0.9,
                          help='Initial sparsification quantile (keep top X%%)')
        parser.add_argument('--sparsity_end', type=float, default=0.7,
                          help='Final sparsification quantile')
        parser.add_argument('--causal_warmup_tasks', type=int, default=5,
                          help='Number of tasks to warm up causal sampling (5 = safer, lets graph stabilize)')
        
        # TRUE INTERVENTIONAL CAUSALITY OPTIONS
        parser.add_argument('--use_true_causality', type=int, default=0,
                          help='Use TRUE interventional causality (0=off, 1=heuristic_only, 2=hybrid, 3=true_only)')
        parser.add_argument('--causal_num_interventions', type=int, default=50,
                          help='Number of intervention samples for TRUE causal measurement')
        parser.add_argument('--causal_effect_threshold', type=float, default=0.05,
                          help='Threshold for causal effect significance (harmful if >threshold)')
        parser.add_argument('--causal_hybrid_candidates', type=int, default=200,
                          help='For hybrid mode: number of candidates from heuristic to re-rank with TRUE causality')
        parser.add_argument('--causal_eval_interval', type=int, default=5,
                          help='Interval (in training steps) between expensive TRUE causal evaluations; reuse previous selection in between')
        parser.add_argument('--true_temp_lr', type=float, default=0.05,
                          help='Temporary learning rate for TRUE causal intervention micro-steps')
        parser.add_argument('--true_micro_steps', type=int, default=2,
                          help='Number of micro-steps to apply during TRUE causal interventions')
        
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
        
        # Causal-specific tuning parameters (FIXED defaults)
        self.causal_blend_ratio = getattr(args, 'causal_blend_ratio', 0.3)  # Reduced from 0.7
        self.sparsity_start = getattr(args, 'sparsity_start', 0.9)
        self.sparsity_end = getattr(args, 'sparsity_end', 0.7)
        self.causal_warmup_tasks = getattr(args, 'causal_warmup_tasks', 5)  # Increased from 3
        
        # TRUE INTERVENTIONAL CAUSALITY
        self.use_true_causality = getattr(args, 'use_true_causality', 0)
        self.causal_num_interventions = getattr(args, 'causal_num_interventions', 50)
        self.causal_effect_threshold = getattr(args, 'causal_effect_threshold', 0.05)
        self.causal_hybrid_candidates = getattr(args, 'causal_hybrid_candidates', 200)
        self.causal_eval_interval = getattr(args, 'causal_eval_interval', 5)
        self.true_temp_lr = getattr(args, 'true_temp_lr', 0.05)
        self.true_micro_steps = getattr(args, 'true_micro_steps', 2)
        # Internal cache for TRUE-only/hybrid reuse between intervals
        self._causal_step_counter = 0
        self._true_cached_batch = None  # (inputs, labels, logits)
        
        # Initialize Structural Causal Model if enabled
        self.scm = None
        self.causal_graph = None
        self.task_feature_cache = {}
        self.causal_forgetting_detector = None  # For TRUE causality
        # Note: current_task is inherited from ContinualModel (read-only property)
        
        if self.enable_causal_graph and StructuralCausalModel is not None:
            self.scm = StructuralCausalModel(
                num_tasks=self.num_tasks,
                feature_dim=self.feature_dim
            )
            print(f"[DER++ Causal] Causal graph learning ENABLED")
            print(f"[DER++ Causal]    - Tasks: {self.num_tasks}")
            print(f"[DER++ Causal]    - Feature dim: {self.feature_dim}D")
            print(f"[DER++ Causal]    - Cache size: {self.causal_cache_size} samples/task")
            print(f"[DER++ Causal]    - Temperature: {self.temperature}")
        else:
            print(f"[DER++ Causal] Running in official DER++ mode (no causality)")
        
        # Initialize TRUE Causal Forgetting Detector
        if self.use_true_causality and CausalForgettingDetector is not None:
            self.causal_forgetting_detector = CausalForgettingDetector(
                model=self.net,
                buffer_size=self.args.buffer_size,
                num_intervention_samples=self.causal_num_interventions,
                true_temp_lr=self.true_temp_lr,
                true_micro_steps=self.true_micro_steps
            )
            print(f"[DER++ Causal] TRUE INTERVENTIONAL CAUSALITY ENABLED")
            print(f"[DER++ Causal]    - Using CausalForgettingDetector")
            print(f"[DER++ Causal]    - Intervention samples: {self.causal_num_interventions}")
            print(f"[DER++ Causal]    - Effect threshold: {self.causal_effect_threshold}")
            print(f"[DER++ Causal]    - Method: Factual vs Counterfactual comparison")
        
        # Metrics tracker
        self.metrics_tracker = ContinualLearningMetrics(num_tasks=self.num_tasks) if ContinualLearningMetrics else None
    
    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        """
        Training step with causal-weighted replay sampling.
        
        Phase 1 (causal OFF): Use official DER++ exactly
        Phase 2 (causal ON): Use causal graph to weight replay samples
        """
        # Cache features for causal graph learning OR TRUE causality
        if (self.enable_causal_graph and self.scm is not None) or self.use_true_causality:
            self._cache_features_for_causal_learning(inputs, labels, not_aug_inputs)
        
        # If TRUE causality enabled, always use causal sampling (doesn't need PC graph)
        # Otherwise, fall back to official DER++ if PC graph not ready
        if not self.use_true_causality:
            # PC Algorithm mode: requires enable_causal_graph and built graph
            if not self.enable_causal_graph or self.causal_graph is None or not self.use_causal_sampling:
                return super().observe(inputs, labels, not_aug_inputs, epoch)
        
        # Otherwise, implement DER++ with causal-weighted replay
        self.opt.zero_grad()
        
        # IMPORTANT: Perform any causal sampling BEFORE building the main forward graph,
        # to avoid interfering with autograd state when TRUE interventions temporarily
        # modify model parameters during sampling.
        buf_inputs = buf_labels = buf_logits = None
        if not self.buffer.is_empty():
            buf_inputs, buf_labels, buf_logits = self._get_causal_weighted_samples(self.args.minibatch_size)
        
        # Now build the main forward graph after sampling
        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)
        
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
        Sample from buffer using causal graph weights OR TRUE causal attribution.
        
        TWO MODES:
        1. PC Algorithm (use_true_causality=0): Correlation-based graph discovery
        2. TRUE Causality (use_true_causality=1): Interventional causal effects
        
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
        
        # MODE 1: TRUE INTERVENTIONAL CAUSALITY
        if self.use_true_causality and self.causal_forgetting_detector is not None and self.current_task >= 1:
            return self._get_true_causal_samples(buf_inputs, buf_labels, buf_logits, size)
        
        # MODE 2: PC ALGORITHM (correlation-based)
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
                # CRITICAL: Graph indexing and importance calculation
                # ======================================================
                # The causal graph is stored as graph[source, target] where:
                #   graph[i, j] = strength of causal edge from Task i → Task j
                #   
                # PROBLEM: graph[old_task, current_task] doesn't exist yet!
                #   → The current task was just added, so no edges TO it exist in the graph
                #   → Example: Training Task 3, graph only has relationships between tasks 0,1,2
                #   → Lookup graph[0, 3] returns 0 because edge doesn't exist yet!
                #
                # SOLUTION: Use the SUM of outgoing edges from old_task as importance
                #   → If Task 0 strongly influenced Tasks 1 and 2, it's likely important for Task 3
                #   → Sum of graph[task_id, :] (all outgoing edges from this task)
                #   → This measures "how influential was this task historically?"
                #
                # Example: Training Task 3, buffer sample from Task 0
                #   → Sum: graph[0,1] + graph[0,2] = 0.678 + 0.594 = 1.272
                #   → High sum = Task 0 was influential = prioritize its samples
                #
                # Compute importance as sum of outgoing edges (row sum)
                causal_strength = abs(self.causal_graph[task_id, :].sum().item())
                
                # Normalize by number of ACTUAL edges (not potential targets)
                # This prevents importance from being diluted when few edges exist
                num_edges = (self.causal_graph[task_id, :].abs() > 0.01).sum().item()
                if num_edges > 0:
                    causal_strength = causal_strength / num_edges
                else:
                    causal_strength = 0.0
                
                # ASSERTION: Sanity check that we're not reading future→past edges
                assert task_id < current_task, \
                    f"Bug: Trying to read causal effect from future task {task_id} to past task {current_task}"
                
                # Use normalized causal influence as importance
                # Higher total influence = more important to replay
                blended_importance = causal_strength
                
                importance_scores[i] = blended_importance
            else:
                # No causal relationship or future task - use small baseline
                importance_scores[i] = 0.1
        
        # QUICK WIN #1 (continued): Blend causal importance with uniform
        uniform_probs = torch.ones_like(importance_scores) / importance_scores.size(0)
        
        # Normalize causal importance to probabilities
        if importance_scores.sum() > 0:
            causal_probs = importance_scores / importance_scores.sum()
        else:
            causal_probs = uniform_probs
        
        # Blend uniform and causal based on warm start factor
        sampling_probs = (1 - causal_blend) * uniform_probs + causal_blend * causal_probs
        
        # DEBUG: Log importance statistics once per task (first call after graph learned)
        if not hasattr(self, '_debug_logged_task') or self._debug_logged_task != current_task:
            if self.causal_graph is not None and current_task >= 1:
                self._debug_logged_task = current_task
                print(f"\n[DEBUG] Task {current_task} - Causal Sampling Diagnostics:")
                print(f"  Warmup: blend={causal_blend:.2f} (warmup_tasks={self.causal_warmup_tasks})")
                print(f"  Importance scores: mean={importance_scores.mean():.4f}, std={importance_scores.std():.4f}, min={importance_scores.min():.4f}, max={importance_scores.max():.4f}")
                print(f"  Samples with causal_strength>0: {(importance_scores > 0.1).sum().item()}/{importance_scores.size(0)}")
                print(f"  Causal probs: mean={causal_probs.mean():.6f}, std={causal_probs.std():.6f}, max={causal_probs.max():.6f}")
                print(f"  Final sampling probs: mean={sampling_probs.mean():.6f}, std={sampling_probs.std():.6f}, max={sampling_probs.max():.6f}")
                
                # Top-10 samples by sampling probability
                top_k = 10
                top_probs, top_indices = torch.topk(sampling_probs, min(top_k, sampling_probs.size(0)))
                print(f"  Top-{top_k} samples by sampling prob:")
                for rank, (idx, prob) in enumerate(zip(top_indices, top_probs)):
                    task_id = int(buf_task_labels[idx].item()) if torch.is_tensor(buf_task_labels[idx]) else int(buf_task_labels[idx])
                    imp = importance_scores[idx].item()
                    print(f"    #{rank+1}: idx={idx.item()}, task={task_id}, importance={imp:.4f}, prob={prob.item():.6f}")
        
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
    
    def _get_true_causal_samples(self, buf_inputs, buf_labels, buf_logits, size):
        """
        Sample using TRUE interventional causality via CausalForgettingDetector.
        
        Supports two modes controlled by use_true_causality argument:
        - Mode 1 (heuristic_only): Fast feature-based filtering
        - Mode 2 (hybrid): Two-stage approach
          1. Heuristic filtering: 500 → causal_hybrid_candidates (default 200)
          2. TRUE causal ranking: candidates → final size (default 128)
        
        TRUE causality uses FACTUAL vs COUNTERFACTUAL comparison:
        1. Factual: Performance WITH sample in buffer
        2. Counterfactual: Performance WITHOUT sample in buffer
        3. Causal effect = Factual - Counterfactual
        
        Samples that REDUCE forgetting (negative causal effect) are prioritized.
        
        Args:
            buf_inputs: Buffer inputs
            buf_labels: Buffer labels
            buf_logits: Buffer logits
            size: Number of samples to retrieve
        
        Returns:
            (inputs, labels, logits) tuple selected by causal effects
        """
        # Reuse previous TRUE selection between evaluations to save time
        if self.causal_eval_interval and self.causal_eval_interval > 1:
            if (self._true_cached_batch is not None and
                (self._causal_step_counter % self.causal_eval_interval) != 0):
                cached_inputs, cached_labels, cached_logits = self._true_cached_batch
                if cached_inputs.size(0) >= size:
                    # Advance counter and reuse cached selection (slice to requested size)
                    self._causal_step_counter += 1
                    return cached_inputs[:size], cached_labels[:size], cached_logits[:size]

        mode_name_map = {1: "HEURISTIC", 2: "HYBRID", 3: "TRUE-ONLY"}
        mode_name = mode_name_map.get(self.use_true_causality, "HEURISTIC")
        print(f"\n[{mode_name} CAUSALITY] Sampling from buffer...")
        
        # Prepare buffer samples for causal detector
        buffer_samples = [(buf_inputs[i], buf_labels[i]) for i in range(buf_inputs.size(0))]
        
        # Get old task data for measuring forgetting (use cached features if available)
        old_task_data = {}
        current_task = self.current_task
        
        if current_task >= 1 and hasattr(self, 'task_feature_cache'):
            # Use cached data from previous tasks
            for task_id in range(current_task):
                if task_id in self.task_feature_cache:
                    cache = self.task_feature_cache[task_id]
                    if len(cache['inputs']) > 0:
                        # Take up to 50 samples per task for forgetting measurement
                        n_samples = min(50, len(cache['inputs']))
                        task_inputs = torch.stack([cache['inputs'][i] for i in range(n_samples)])
                        task_labels = torch.stack([cache['labels'][i] for i in range(n_samples)])
                        old_task_data[task_id] = (task_inputs.to(self.device), task_labels.to(self.device))
        
        if len(old_task_data) == 0:
            # Fallback: Can't measure forgetting without old task data, use uniform sampling
            print(f"  WARNING: No old task data available for causal attribution. Using uniform sampling.")
            indices = torch.randperm(buf_inputs.size(0))[:size]
            # Cache selection and advance counter
            selected = (buf_inputs[indices], buf_labels[indices], buf_logits[indices])
            self._true_cached_batch = selected
            self._causal_step_counter += 1
            return selected
        
        # ============================================================
        # TRUE-ONLY MODE: Skip heuristic filtering entirely
        # ============================================================
        if self.use_true_causality == 3:
            # Prepare candidate pool (limit for runtime)
            num_candidates = min(self.causal_hybrid_candidates, buf_inputs.size(0))
            candidate_indices = torch.randperm(buf_inputs.size(0))[:num_candidates]
            print(f"  [TRUE-ONLY] Computing TRUE causal effects on {num_candidates} candidates...")
            print(f"  [TRUE-ONLY] Measuring forgetting across {len(old_task_data)} old tasks: {list(old_task_data.keys())}")

            true_causal_effects = []
            for idx in candidate_indices.tolist():
                candidate_sample = (buf_inputs[idx], buf_labels[idx])
                other_samples = [buffer_samples[i] for i in range(len(buffer_samples)) if i != idx]

                try:
                    effect = self.causal_forgetting_detector.attribute_forgetting(
                        candidate_sample=candidate_sample,
                        buffer_samples=other_samples,
                        old_task_data=old_task_data,
                        current_task_id=current_task,
                        use_true_intervention=True
                    )
                    true_causal_effects.append({
                        'index': idx,
                        'effect_size': effect.effect_size,
                        'confidence': effect.confidence,
                        'mechanism': effect.mechanism
                    })
                except Exception as e:
                    if len(true_causal_effects) == 0:
                        print(f"    [WARNING] TRUE causal attribution failed: {type(e).__name__}: {str(e)[:100]}")
                    true_causal_effects.append({
                        'index': idx,
                        'effect_size': 0.0,
                        'confidence': 0.0,
                        'mechanism': 'error'
                    })

            # Log stats
            print(f"  [TRUE-ONLY] TRUE causal effects computed:")
            if len(true_causal_effects) > 0:
                effect_sizes = [e['effect_size'] for e in true_causal_effects]
                print(f"    Effect size range: [{min(effect_sizes):.4f}, {max(effect_sizes):.4f}], mean={sum(effect_sizes)/len(effect_sizes):.4f}")
                print(f"    Beneficial samples: {sum(1 for e in true_causal_effects if e['effect_size'] < -self.causal_effect_threshold)}")
                print(f"    Harmful samples: {sum(1 for e in true_causal_effects if e['effect_size'] > self.causal_effect_threshold)}")
                print(f"    Neutral samples: {sum(1 for e in true_causal_effects if abs(e['effect_size']) <= self.causal_effect_threshold)}")

            sorted_true_effects = sorted(true_causal_effects, key=lambda x: x['effect_size'])
            selected_indices = [e['index'] for e in sorted_true_effects[:size]]
            indices = torch.tensor(selected_indices, dtype=torch.long)
            print(f"  [TRUE-ONLY] Selected {len(indices)} samples from {num_candidates} candidates")
            # Cache selection and advance counter
            selected = (buf_inputs[indices], buf_labels[indices], buf_logits[indices])
            self._true_cached_batch = selected
            self._causal_step_counter += 1
            return selected

        # ============================================================
        # STAGE 1: HEURISTIC FILTERING (Fast feature-based scoring)
        # ============================================================
        print(f"  [STAGE 1] Heuristic filtering on {buf_inputs.size(0)} samples...")
        
        heuristic_effects = []
        
        for idx in range(buf_inputs.size(0)):
            candidate_sample = (buf_inputs[idx], buf_labels[idx])
            
            # Get buffer WITHOUT this sample
            other_samples = [buffer_samples[i] for i in range(len(buffer_samples)) if i != idx]
            
            try:
                # FAST HEURISTIC scoring (feature interference)
                effect = self.causal_forgetting_detector.attribute_forgetting(
                    candidate_sample=candidate_sample,
                    buffer_samples=other_samples,
                    old_task_data=old_task_data,
                    current_task_id=current_task,
                    use_true_intervention=False  # Fast mode
                )
                
                heuristic_effects.append({
                    'index': idx,
                    'effect_size': effect.effect_size,
                    'confidence': effect.confidence,
                    'mechanism': effect.mechanism
                })
            except Exception as e:
                # Log error for debugging
                if len(heuristic_effects) == 0:  # Only log first error
                    print(f"    [WARNING] Heuristic scoring failed: {type(e).__name__}: {str(e)[:100]}")
                # If scoring fails, assign neutral effect
                heuristic_effects.append({
                    'index': idx,
                    'effect_size': 0.0,
                    'confidence': 0.0,
                    'mechanism': 'error'
                })
        
        # Log Stage 1 statistics
        print(f"  [STAGE 1] Heuristic scoring complete:")
        if len(heuristic_effects) > 0:
            effect_sizes = [e['effect_size'] for e in heuristic_effects]
            print(f"    Effect size range: [{min(effect_sizes):.4f}, {max(effect_sizes):.4f}], mean={sum(effect_sizes)/len(effect_sizes):.4f}")
            print(f"    Beneficial: {sum(1 for e in heuristic_effects if e['effect_size'] < -self.causal_effect_threshold)}")
            print(f"    Harmful: {sum(1 for e in heuristic_effects if e['effect_size'] > self.causal_effect_threshold)}")
            print(f"    Neutral: {sum(1 for e in heuristic_effects if abs(e['effect_size']) <= self.causal_effect_threshold)}")
        
        # MODE 1: Heuristic only - directly select from heuristic scores
        if self.use_true_causality == 1:
            # Sort by effect size (most beneficial = most negative)
            sorted_effects = sorted(heuristic_effects, key=lambda x: x['effect_size'])
            
            # Select top samples
            selected_indices = [e['index'] for e in sorted_effects[:size]]
            indices = torch.tensor(selected_indices, dtype=torch.long)
            
            print(f"  [HEURISTIC-ONLY] Selected {len(indices)} samples based on feature similarity.")
            return buf_inputs[indices], buf_labels[indices], buf_logits[indices]
        
        # ============================================================
        # MODE 2: HYBRID - Stage 2 with TRUE causal intervention
        # ============================================================
        
        # Select top candidates from heuristic filtering
        num_candidates = min(self.causal_hybrid_candidates, len(heuristic_effects))
        sorted_effects = sorted(heuristic_effects, key=lambda x: x['effect_size'])
        top_candidates = sorted_effects[:num_candidates]
        
        print(f"  [STAGE 2] TRUE causal re-ranking on top {num_candidates} candidates...")
        
        # Compute TRUE causal effects for top candidates
        true_causal_effects = []
        
        for candidate_dict in top_candidates:
            idx = candidate_dict['index']
            candidate_sample = (buf_inputs[idx], buf_labels[idx])
            
            # Get buffer WITHOUT this sample
            other_samples = [buffer_samples[i] for i in range(len(buffer_samples)) if i != idx]
            
            try:
                # TRUE CAUSAL INTERVENTION (gradient-based)
                effect = self.causal_forgetting_detector.attribute_forgetting(
                    candidate_sample=candidate_sample,
                    buffer_samples=other_samples,
                    old_task_data=old_task_data,
                    current_task_id=current_task,
                    use_true_intervention=True  # TRUE causality
                )
                
                true_causal_effects.append({
                    'index': idx,
                    'effect_size': effect.effect_size,
                    'confidence': effect.confidence,
                    'mechanism': effect.mechanism
                })
            except Exception as e:
                # Log error for debugging
                if len(true_causal_effects) == 0:  # Only log first error
                    print(f"    [WARNING] TRUE causal attribution failed: {type(e).__name__}: {str(e)[:100]}")
                # If causal attribution fails, keep heuristic score
                true_causal_effects.append({
                    'index': idx,
                    'effect_size': candidate_dict['effect_size'],  # Fallback to heuristic
                    'confidence': 0.0,
                    'mechanism': 'error_fallback'
                })
        
        # Log Stage 2 statistics
        print(f"  [STAGE 2] TRUE causal effects computed:")
        if len(true_causal_effects) > 0:
            effect_sizes = [e['effect_size'] for e in true_causal_effects]
            print(f"    Effect size range: [{min(effect_sizes):.4f}, {max(effect_sizes):.4f}], mean={sum(effect_sizes)/len(effect_sizes):.4f}")
            print(f"    Beneficial samples: {sum(1 for e in true_causal_effects if e['effect_size'] < -self.causal_effect_threshold)}")
            print(f"    Harmful samples: {sum(1 for e in true_causal_effects if e['effect_size'] > self.causal_effect_threshold)}")
            print(f"    Neutral samples: {sum(1 for e in true_causal_effects if abs(e['effect_size']) <= self.causal_effect_threshold)}")
        
        # Sort by TRUE causal effect (most beneficial = most negative)
        sorted_true_effects = sorted(true_causal_effects, key=lambda x: x['effect_size'])
        
        # Select final samples
        selected_indices = [e['index'] for e in sorted_true_effects[:size]]
        indices = torch.tensor(selected_indices, dtype=torch.long)
        
        print(f"  [HYBRID] Selected {len(indices)} samples: {num_candidates} candidates → {len(indices)} final")
        
        # Cache selection and advance counter
        selected = (buf_inputs[indices], buf_labels[indices], buf_logits[indices])
        self._true_cached_batch = selected
        self._causal_step_counter += 1
        return selected
    
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
        # Reset TRUE causal sampling cache at task boundary
        self._causal_step_counter = 0
        self._true_cached_batch = None
    
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
            print(f"  WARNING: Not enough data yet (need at least 2 tasks with at least 10 samples)")
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
                # VALIDATION: Ensure graph has expected shape and properties
                assert self.causal_graph.shape == (self.num_tasks, self.num_tasks), \
                    f"Bug: Causal graph has wrong shape {self.causal_graph.shape}, expected ({self.num_tasks}, {self.num_tasks})"
                
                # VALIDATION: Check that graph follows temporal ordering (no future→past edges)
                # For continual learning, Task i can only influence Task j if i < j (learned first)
                # Strong backward edges (j→i where j>i) indicate a bug
                for i in range(self.num_tasks):
                    for j in range(i):  # j < i means j is earlier than i
                        if abs(self.causal_graph[i, j]) > 0.3:  # Strong backward edge
                            logger.warning(
                                f"WARNING: Strong backward edge detected: "
                                f"Task {i} -> Task {j} (strength={self.causal_graph[i, j]:.3f}). "
                                f"This suggests task {i} causally affects earlier task {j}, which violates temporal ordering. "
                                f"This may indicate a bug in graph learning or feature entanglement."
                            )
                
                # VALIDATION: Diagonal should be zero (tasks don't cause themselves)
                diag_max = torch.diag(self.causal_graph).abs().max().item()
                if diag_max > 0.01:
                    logger.warning(
                        f"WARNING: Non-zero diagonal in causal graph (max={diag_max:.3f}). "
                        f"Tasks should not have causal edges to themselves. Setting diagonal to zero."
                    )
                    self.causal_graph.fill_diagonal_(0)
                
                self._print_causal_graph_summary()
            else:
                print(f"  WARNING: Causal graph learning returned None")
                
        except Exception as e:
            print(f"  WARNING: Causal graph learning failed: {e}")
            import traceback
            traceback.print_exc()
    
    def _print_causal_graph_summary(self):
        """Print summary of learned causal graph."""
        print(f"  Causal graph learned successfully")
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
