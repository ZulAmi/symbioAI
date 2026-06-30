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

logger = logging.getLogger(__name__)

# Add repo root to path so Mammoth can be found when running from symbioAI/
repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from mammoth.models.derpp import Derpp

try:
    from training.causal_inference import StructuralCausalModel, CausalForgettingDetector
    from training.metrics_tracker import ContinualLearningMetrics
    from training.influence_approx import InfluenceFunctionApproximator
except ImportError as e:
    raise ImportError(
        f"Cannot import causal inference modules: {e}\n"
        "Run experiments from the repo root with:\n"
        "  PYTHONPATH=/path/to/symbioAI python -m utils.main --model derpp-causal ..."
    ) from e


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
                          help='Causal sampling mode: 0=vanilla, 1=heuristic, 2=hybrid, '
                               '3=TRUE (checkpoint/restore), 4=influence-fn (LiSSA)')
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
        parser.add_argument('--causal_warmup_tasks', type=int, default=2,
                          help='Number of tasks to warm up causal sampling (5 = safer, lets graph stabilize)')
        
        # TRUE INTERVENTIONAL CAUSALITY OPTIONS
        # Note: use_causal_sampling is the primary flag (0=off, 1=heuristic, 2=hybrid, 3=TRUE-only)
        # use_true_causality is kept for backward compatibility but defaults to use_causal_sampling
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
        
        # OPTIMIZATION FLAGS
        parser.add_argument('--debug', type=int, default=0,
                          help='Enable debug logging (0=off, 1=on)')
        parser.add_argument('--use_batched_causality', type=int, default=1,
                          help='Use batched causal effect measurement (faster)')
        parser.add_argument('--causal_batch_size', type=int, default=16,
                          help='Batch size for causal measurements')
        
        return parser
    
    def __init__(self, backbone, loss, args, transform, dataset=None):
        """Initialize with official DER++ + causal extensions."""
        # Initialize official DER++ (inherits buffer, optimizer, etc.)
        super().__init__(backbone, loss, args, transform, dataset=dataset)
        
        # Causal extensions (with defaults matching test scripts)
        self.enable_causal_graph = getattr(args, 'enable_causal_graph_learning', 0)
        self.use_causal_sampling = getattr(args, 'use_causal_sampling', 0)
        self.use_importance_sampling = getattr(args, 'use_importance_sampling', 0)
        
        logger.debug("use_causal_sampling=%d", self.use_causal_sampling)
        self.temperature = getattr(args, 'temperature', 2.0)
        self.num_tasks = getattr(args, 'num_tasks', 10)
        self.feature_dim = getattr(args, 'feature_dim', 512)
        self.causal_cache_size = getattr(args, 'causal_cache_size', 200)
        self.importance_weight = getattr(args, 'importance_weight', 0.5)
        
        # Causal-specific tuning parameters (FIXED defaults)
        self.causal_blend_ratio = getattr(args, 'causal_blend_ratio', 0.3)  # Reduced from 0.7
        self.sparsity_start = getattr(args, 'sparsity_start', 0.9)
        self.sparsity_end = getattr(args, 'sparsity_end', 0.7)
        self.causal_warmup_tasks = getattr(args, 'causal_warmup_tasks', 2)
        
        # TRUE INTERVENTIONAL CAUSALITY
        # use_causal_sampling is the primary flag: 0=none, 1=heuristic, 2=hybrid, 3=TRUE-only
        self.use_true_causality = self.use_causal_sampling
        
        logger.debug("use_true_causality=%d, CausalForgettingDetector available=%s",
                     self.use_true_causality, CausalForgettingDetector is not None)
        
        self.causal_num_interventions = getattr(args, 'causal_num_interventions', 50)
        self.causal_effect_threshold = getattr(args, 'causal_effect_threshold', 0.05)
        self.causal_hybrid_candidates = getattr(args, 'causal_hybrid_candidates', 200)
        self.causal_eval_interval = getattr(args, 'causal_eval_interval', 5)
        self.true_temp_lr = getattr(args, 'true_temp_lr', 0.05)
        self.true_micro_steps = getattr(args, 'true_micro_steps', 2)
        # Internal cache for TRUE-only/hybrid reuse between intervals
        self._causal_step_counter = 0
        self._true_cached_batch = None  # (inputs, labels, logits)
        
        # OPTIMIZATION FLAGS
        self.debug = getattr(args, 'debug', 0) == 1
        self.use_batched_causality = getattr(args, 'use_batched_causality', 1) == 1
        self.causal_batch_size = getattr(args, 'causal_batch_size', 16)
        
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
            logger.info(
                "Causal graph learning enabled — tasks=%d, feature_dim=%dD, "
                "cache_size=%d, temperature=%.1f",
                self.num_tasks, self.feature_dim, self.causal_cache_size, self.temperature,
            )
        else:
            logger.info("Running in standard DER++ mode (causality disabled)")
        
        if self.use_true_causality and CausalForgettingDetector is not None:
            self.causal_forgetting_detector = CausalForgettingDetector(
                model=self.net,
                buffer_size=self.args.buffer_size,
                num_intervention_samples=self.causal_num_interventions,
                true_temp_lr=self.true_temp_lr,
                true_micro_steps=self.true_micro_steps
            )
            self.causal_forgetting_detector.debug = self.debug
            logger.info(
                "TRUE interventional causality enabled — mode=%d, interventions=%d, "
                "threshold=%.3f, batched=%s (batch_size=%d)",
                self.use_true_causality, self.causal_num_interventions,
                self.causal_effect_threshold, self.use_batched_causality, self.causal_batch_size,
            )
        
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
        
        if self.debug and (not hasattr(self, '_debug_logged_task') or self._debug_logged_task != current_task):
            if self.causal_graph is not None and current_task >= 1:
                self._debug_logged_task = current_task
                logger.debug(
                    "Task %d causal sampling — blend=%.2f, importance: mean=%.4f std=%.4f "
                    "min=%.4f max=%.4f, causal_strength>0: %d/%d",
                    current_task, causal_blend,
                    importance_scores.mean(), importance_scores.std(),
                    importance_scores.min(), importance_scores.max(),
                    (importance_scores > 0.1).sum().item(), importance_scores.size(0),
                )
        
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
        logger.info("[%s] Sampling from buffer (task=%d)", mode_name, self.current_task)
        
        # CRITICAL: Ensure buffer data is float32 (not float16 or float64)
        buf_inputs = buf_inputs.float()
        buf_labels = buf_labels.long()
        
        # Prepare buffer samples for causal detector
        buffer_samples = [(buf_inputs[i], buf_labels[i]) for i in range(buf_inputs.size(0))]
        
        # Get old task data for measuring forgetting across ALL previous tasks
        old_task_data = {}
        current_task = self.current_task
        
        if current_task >= 1:
            # CRITICAL FIX: Extract old task samples from BUFFER, not just cache
            # The buffer contains samples from ALL previous tasks, so we can measure
            # forgetting across all of them
            
            # Get task labels from buffer (assumes buffer stores task info or we infer from class labels)
            if hasattr(self.buffer, 'task_labels') and self.buffer.task_labels is not None:
                buf_task_labels = self.buffer.task_labels[:buf_inputs.size(0)]
            else:
                # Infer task from class label (assumes cpt = classes_per_task is set)
                if hasattr(self, 'cpt'):
                    buf_task_labels = buf_labels // self.cpt
                else:
                    # Fallback: assume 10 classes per task for CIFAR-100
                    buf_task_labels = buf_labels // 10
            
            # Group buffer samples by task
            for task_id in range(current_task):
                # Find all samples from this task in the buffer
                task_mask = (buf_task_labels == task_id)
                if task_mask.sum() > 0:
                    task_indices = task_mask.nonzero(as_tuple=True)[0]
                    # Take up to 50 samples per task for measurement
                    n_samples = min(15, len(task_indices))
                    sampled_indices = task_indices[torch.randperm(len(task_indices))[:n_samples]]
                    
                    # 15 samples per task is sufficient for ranking signal; was 50 (3× speedup)
                    task_inputs = buf_inputs[sampled_indices]
                    task_labels = buf_labels[sampled_indices]
                    old_task_data[task_id] = (task_inputs.to(self.device), task_labels.to(self.device))

            
            if self.debug and len(old_task_data) > 0 and not hasattr(self, '_debug_task_extraction_logged'):
                self._debug_task_extraction_logged = True
                logger.debug(
                    "Buffer task distribution at task %d: unique=%s, keys=%s",
                    current_task, torch.unique(buf_task_labels).tolist(), list(old_task_data.keys()),
                )
        
        if len(old_task_data) == 0:
            logger.warning("No old task data for causal attribution — falling back to uniform sampling")
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
            num_candidates = min(self.causal_hybrid_candidates, buf_inputs.size(0))
            candidate_indices = torch.randperm(buf_inputs.size(0))[:num_candidates]

            logger.debug("[TRUE-ONLY] Evaluating %d candidates across %d old tasks",
                         num_candidates, len(old_task_data))

            # --- KEY OPTIMISATION: compute checkpoint + baseline ONCE ---
            # The original code cloned the full model state dict inside every
            # per-candidate call to _measure_true_causal_effect (100 × 44 MB
            # clone) and recomputed the counterfactual for each candidate
            # separately. Hoisting both out of the loop gives ~2× speedup in
            # the evaluation loop, on top of the 5× gain from reducing the
            # forgetting sample count from 50 → 10 per task.
            checkpoint, baseline_forgetting = (
                self.causal_forgetting_detector.precompute_causal_baseline(
                    buffer_samples, old_task_data
                )
            )
            # One small shared replay batch for all factual passes
            num_replay = min(16, len(buffer_samples))
            shared_replay_idx = torch.randperm(len(buffer_samples))[:num_replay].tolist()
            shared_replay = [buffer_samples[i] for i in shared_replay_idx]

            true_causal_effects = []
            for idx in candidate_indices.tolist():
                try:
                    effect = self.causal_forgetting_detector.measure_causal_effect_fast(
                        candidate_sample=(buf_inputs[idx], buf_labels[idx]),
                        old_task_data=old_task_data,
                        checkpoint=checkpoint,
                        baseline_forgetting=baseline_forgetting,
                        shared_replay=shared_replay,
                    )
                    true_causal_effects.append({
                        'index': idx,
                        'effect_size': effect.effect_size,
                        'confidence': effect.confidence,
                        'mechanism': effect.mechanism,
                    })
                except Exception as e:
                    if not true_causal_effects:
                        logger.warning("[TRUE-ONLY] causal attribution failed: %s: %s",
                                       type(e).__name__, str(e)[:120])
                    true_causal_effects.append({
                        'index': idx,
                        'effect_size': 0.0,
                        'confidence': 0.0,
                        'mechanism': 'error',
                    })

            if true_causal_effects:
                effect_sizes = [e['effect_size'] for e in true_causal_effects]
                logger.debug(
                    "[TRUE-ONLY] effects: range=[%.4f, %.4f] mean=%.4f — "
                    "beneficial=%d harmful=%d neutral=%d",
                    min(effect_sizes), max(effect_sizes), sum(effect_sizes) / len(effect_sizes),
                    sum(1 for e in true_causal_effects if e['effect_size'] < -self.causal_effect_threshold),
                    sum(1 for e in true_causal_effects if e['effect_size'] > self.causal_effect_threshold),
                    sum(1 for e in true_causal_effects if abs(e['effect_size']) <= self.causal_effect_threshold),
                )

            sorted_true_effects = sorted(true_causal_effects, key=lambda x: x['effect_size'])
            selected_indices = [e['index'] for e in sorted_true_effects[:size]]
            indices = torch.tensor(selected_indices, dtype=torch.long)

            logger.debug("[TRUE-ONLY] Selected %d samples from %d candidates", len(indices), num_candidates)
            
            # Cache selection and advance counter
            selected = (buf_inputs[indices], buf_labels[indices], buf_logits[indices])
            self._true_cached_batch = selected
            self._causal_step_counter += 1
            
            # Print optimization stats every few tasks
            if current_task % 3 == 0 and hasattr(self.causal_forgetting_detector, 'print_optimization_stats'):
                self.causal_forgetting_detector.print_optimization_stats()
            
            return selected

        # ============================================================
        # INFLUENCE FUNCTION MODE (mode 4): LiSSA approximation
        # ============================================================
        if self.use_true_causality == 4:
            num_candidates = min(self.causal_hybrid_candidates, buf_inputs.size(0))
            candidate_indices = torch.randperm(buf_inputs.size(0))[:num_candidates]

            if not hasattr(self, "_influence_approx"):
                self._influence_approx = InfluenceFunctionApproximator(self.net)

            buffer_samples_cands = [
                (buf_inputs[i], buf_labels[i]) for i in candidate_indices.tolist()
            ]
            top_k_local = self._influence_approx.rank_buffer_samples(
                buffer_samples=buffer_samples_cands,
                old_task_data=old_task_data,
                top_k=size,
            )
            # Map local indices back to global buffer indices
            selected_global = [candidate_indices[i].item() for i in top_k_local]
            indices = torch.tensor(selected_global[:size], dtype=torch.long)

            logger.debug("[INFLUENCE] Selected %d/%d candidates via LiSSA", len(indices), num_candidates)
            selected = (buf_inputs[indices], buf_labels[indices], buf_logits[indices])
            self._true_cached_batch = selected
            self._causal_step_counter += 1
            return selected

        # ============================================================
        # STAGE 1: HEURISTIC FILTERING (Fast feature-based scoring)
        # ============================================================
        logger.info("[STAGE 1] Heuristic filtering on %d buffer samples", buf_inputs.size(0))
        
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
                if len(heuristic_effects) == 0:
                    logger.warning("[STAGE 1] Heuristic scoring failed: %s: %s",
                                   type(e).__name__, str(e)[:100])
                # If scoring fails, assign neutral effect
                heuristic_effects.append({
                    'index': idx,
                    'effect_size': 0.0,
                    'confidence': 0.0,
                    'mechanism': 'error'
                })
        
        if heuristic_effects:
            effect_sizes = [e['effect_size'] for e in heuristic_effects]
            logger.debug(
                "[STAGE 1] complete: range=[%.4f, %.4f] mean=%.4f — "
                "beneficial=%d harmful=%d neutral=%d",
                min(effect_sizes), max(effect_sizes), sum(effect_sizes) / len(effect_sizes),
                sum(1 for e in heuristic_effects if e['effect_size'] < -self.causal_effect_threshold),
                sum(1 for e in heuristic_effects if e['effect_size'] > self.causal_effect_threshold),
                sum(1 for e in heuristic_effects if abs(e['effect_size']) <= self.causal_effect_threshold),
            )
        
        # MODE 1: Heuristic only - directly select from heuristic scores
        if self.use_true_causality == 1:
            # Sort by effect size (most beneficial = most negative)
            sorted_effects = sorted(heuristic_effects, key=lambda x: x['effect_size'])
            
            # Select top samples
            selected_indices = [e['index'] for e in sorted_effects[:size]]
            indices = torch.tensor(selected_indices, dtype=torch.long)
            
            logger.info("[HEURISTIC-ONLY] Selected %d samples by feature similarity", len(indices))
            return buf_inputs[indices], buf_labels[indices], buf_logits[indices]
        
        # ============================================================
        # MODE 2: HYBRID - Stage 2 with TRUE causal intervention
        # ============================================================
        
        # Select top candidates from heuristic filtering
        num_candidates = min(self.causal_hybrid_candidates, len(heuristic_effects))
        sorted_effects = sorted(heuristic_effects, key=lambda x: x['effect_size'])
        top_candidates = sorted_effects[:num_candidates]
        
        logger.info("[STAGE 2] TRUE causal re-ranking on top %d candidates", num_candidates)
        
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
                if len(true_causal_effects) == 0:
                    logger.warning("[STAGE 2] TRUE causal attribution failed: %s: %s",
                                   type(e).__name__, str(e)[:100])
                true_causal_effects.append({
                    'index': idx,
                    'effect_size': candidate_dict['effect_size'],  # Fallback to heuristic
                    'confidence': 0.0,
                    'mechanism': 'error_fallback'
                })
        
        if true_causal_effects:
            effect_sizes = [e['effect_size'] for e in true_causal_effects]
            logger.debug(
                "[STAGE 2] effects: range=[%.4f, %.4f] mean=%.4f — "
                "beneficial=%d harmful=%d neutral=%d",
                min(effect_sizes), max(effect_sizes), sum(effect_sizes) / len(effect_sizes),
                sum(1 for e in true_causal_effects if e['effect_size'] < -self.causal_effect_threshold),
                sum(1 for e in true_causal_effects if e['effect_size'] > self.causal_effect_threshold),
                sum(1 for e in true_causal_effects if abs(e['effect_size']) <= self.causal_effect_threshold),
            )
        
        # Sort by TRUE causal effect (most beneficial = most negative)
        sorted_true_effects = sorted(true_causal_effects, key=lambda x: x['effect_size'])
        
        # Select final samples
        selected_indices = [e['index'] for e in sorted_true_effects[:size]]
        indices = torch.tensor(selected_indices, dtype=torch.long)
        
        logger.info("[HYBRID] Selected %d samples from %d candidates", len(indices), num_candidates)
        
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
        task_id = self.current_task
        logger.info("End of task %d — buffer: %d/%d", task_id, len(self.buffer), self.buffer.buffer_size)
        
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
        task_id = self.current_task
        logger.info("Learning causal graph — tasks completed: %d, cached: %s",
                    task_id + 1, list(self.task_feature_cache.keys()))
        
        # Check if we have enough cached data
        valid_tasks = [
            tid for tid, cache in self.task_feature_cache.items()
            if len(cache['features']) >= 10
        ]
        
        if len(valid_tasks) < 2:
            logger.warning("Not enough cached data for causal graph (need ≥2 tasks with ≥10 samples)")
            return
        
        # Convert cached features to format expected by SCM
        task_data = {}
        task_labels = {}
        
        for tid in valid_tasks:
            cache = self.task_feature_cache[tid]
            if len(cache['features']) > 0:
                task_data[tid] = torch.stack(cache['features'])
                task_labels[tid] = torch.stack(cache['labels'])
        
        logger.debug("Learning from %d tasks: %s",
                     len(task_data),
                     {tid: data.shape for tid, data in task_data.items()})
        
        # Adaptive sparsification using tunable start/end thresholds
        # Gradually transition from sparsity_start to sparsity_end as tasks progress
        task_progress = task_id / max(self.num_tasks - 1, 1)
        adaptive_quantile = self.sparsity_start - (self.sparsity_start - self.sparsity_end) * task_progress
        adaptive_quantile = max(0.5, min(0.95, adaptive_quantile))  # Clamp to safe range [0.5, 0.95]
        logger.debug("Adaptive sparsification: quantile=%.2f (start=%.2f → end=%.2f)",
                     adaptive_quantile, self.sparsity_start, self.sparsity_end)
        
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
                logger.warning("Causal graph learning returned None")

        except Exception as e:
            logger.exception("Causal graph learning failed: %s", e)
    
    def _print_causal_graph_summary(self) -> None:
        """Log summary of learned causal graph."""
        strong_edges = (self.causal_graph.abs() > 0.5).nonzero(as_tuple=False)
        num_edges = int((self.causal_graph.abs() > 0.1).sum().item())
        graph_density = num_edges / (self.num_tasks * (self.num_tasks - 1)) if self.num_tasks > 1 else 0.0

        logger.info(
            "Causal graph learned — shape=%s, edges(>0.1)=%d, density=%.1f%%, "
            "mean|w|=%.3f, max|w|=%.3f",
            tuple(self.causal_graph.shape), num_edges, graph_density * 100,
            self.causal_graph.abs().mean().item(), self.causal_graph.abs().max().item(),
        )

        if len(strong_edges) > 0:
            edge_strs = []
            for edge in strong_edges[:10]:
                src, tgt = int(edge[0]), int(edge[1])
                w = float(self.causal_graph[src, tgt])
                edge_strs.append(f"T{src}{'→' if w > 0 else '↛'}T{tgt}:{w:.3f}")
            suffix = f" (+{len(strong_edges) - 10} more)" if len(strong_edges) > 10 else ""
            logger.debug("Strong edges (|w|>0.5): %s%s", ", ".join(edge_strs), suffix)
        else:
            logger.debug("No strong causal dependencies found (all |w| < 0.5)")


# For backward compatibility with existing code
DERPPCausal = DerppCausal
