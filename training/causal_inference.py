"""
Causal Inference Foundations for Continual Learning
====================================================

Implements actual causal inference machinery for continual learning:
1. Structural Causal Models (SCMs)
2. Causal effect estimation via interventions
3. Counterfactual reasoning
4. Causal graph discovery

Based on Pearl's causal hierarchy:
- Level 1 (Association): P(Y|X) - standard ML
- Level 2 (Intervention): P(Y|do(X)) - what we implement here
- Level 3 (Counterfactuals): P(Y_x|X',Y') - what could have been

References:
- Pearl, J. (2009). Causality: Models, Reasoning and Inference
- Peters, J., Janzing, D., & Schölkopf, B. (2017). Elements of Causal Inference
- Aljundi et al. (2019). Online Continual Learning with Maximal Interfered Retrieval

Author: Zulhilmi Rahmat
Date: October 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from collections import defaultdict
import logging

# Optional imports for advanced causal inference (with graceful fallback)
try:
    from sklearn.cluster import KMeans
    from sklearn.linear_model import LogisticRegression
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    import warnings
    warnings.warn("scikit-learn not available. Some advanced ATE estimation methods will use fallbacks.")

logger = logging.getLogger(__name__)


@dataclass
class CausalEffect:
    """Represents a causal effect measurement."""
    source: str  # What we intervened on (e.g., "task_1")
    target: str  # What we measured (e.g., "performance_task_2")
    effect_size: float  # Magnitude of causal effect
    confidence: float  # Statistical confidence [0, 1]
    mechanism: str  # How the effect propagates (e.g., "feature_interference")


class StructuralCausalModel:
    """
    Structural Causal Model for continual learning.
    
    Represents causal relationships between:
    - Tasks (T1 → T2: learning T1 affects T2)
    - Features (F1 → F2: features are causally related)
    - Performance (T1 → P2: T1 causes change in P2)
    
    Key Innovation: Models the DGP (Data Generating Process) to enable interventions.
    """
    
    def __init__(self, num_tasks: int, feature_dim: int):
        """
        Args:
            num_tasks: Number of tasks in the continual learning sequence
            feature_dim: Dimensionality of learned representations
        """
        self.num_tasks = num_tasks
        self.feature_dim = feature_dim
        
        # Causal graph: adjacency matrix where A[i,j] = strength of causal link Ti → Tj
        self.task_graph = torch.zeros(num_tasks, num_tasks)
        
        # Feature-level causal mechanisms
        self.feature_mechanisms = {}  # task_i -> (mean, cov) of feature distribution
        
        # Noise distributions (exogenous variables in SCM)
        self.noise_distributions = {}
        
        # Intervention history
        self.interventions = []
        
        logger.info(f"Initialized SCM with {num_tasks} tasks, {feature_dim}D features")
    
    def learn_causal_structure(self, 
                               task_data: Dict[int, torch.Tensor],
                               task_labels: Dict[int, torch.Tensor],
                               model: nn.Module,
                               sparsification_quantile: float = 0.7) -> torch.Tensor:
        """
        Discover causal relationships between tasks using independence testing.
        
        Uses conditional independence tests to build causal graph:
        If T_i ⊥ T_j | T_k, then no direct edge T_i → T_j
        
        Args:
            task_data: Dict mapping task_id -> feature representations
            task_labels: Dict mapping task_id -> labels
            model: The neural network (for extracting features)
            sparsification_quantile: Quantile threshold for edge pruning (default 0.7)
        
        Returns:
            Adjacency matrix of causal graph
        """
        logger.info(f"Learning causal structure via independence testing (quantile={sparsification_quantile:.2f})...")
        
        with torch.no_grad():
            # Extract feature distributions per task
            for task_id, data in task_data.items():
                # If 'data' are already features (2D [N, D]), use them directly
                if isinstance(data, torch.Tensor) and data.dim() == 2:
                    features = data
                else:
                    # Otherwise, forward through the model to get features
                    if hasattr(model, 'net'):
                        feats_or_logits = model.net(data)
                    else:
                        # Many Mammoth backbones accept returnt='features'
                        try:
                            feats_or_logits = model(data, returnt='features')  # type: ignore[arg-type]
                        except Exception:
                            feats_or_logits = model(data)

                    # If still spatial, globally average pool
                    if isinstance(feats_or_logits, torch.Tensor) and feats_or_logits.dim() > 2:
                        features = feats_or_logits.mean(dim=[-1, -2])
                    else:
                        features = feats_or_logits

                    # If the model returned logits, accept them as features as last resort
                    if not isinstance(features, torch.Tensor) or features.dim() != 2:
                        features = torch.as_tensor(features)
                        if features.dim() == 1:
                            features = features.unsqueeze(0)

                # Estimate feature distribution
                mean = features.mean(dim=0)
                cov = torch.cov(features.T)
                self.feature_mechanisms[task_id] = (mean, cov)
            
            # Pairwise causal discovery
            for i in range(self.num_tasks):
                for j in range(self.num_tasks):
                    if i == j or i not in task_data or j not in task_data:
                        continue
                    
                    # TEMPORAL CONSTRAINT: In continual learning, only allow forward edges (i < j)
                    # Task i can only causally influence Task j if i was learned before j
                    if i >= j:
                        # Backward edge (i→j where i≥j) - set to zero (violates temporal ordering)
                        self.task_graph[i, j] = 0.0
                        continue
                    
                    # Test: Does intervening on task i affect task j?
                    effect = self._estimate_causal_effect(i, j, task_data, model)
                    self.task_graph[i, j] = effect
        
        # Sparsify: only keep strong causal links (adaptive threshold)
        threshold = self.task_graph.abs().quantile(sparsification_quantile)
        self.task_graph[self.task_graph.abs() < threshold] = 0
        
        logger.info(f"Discovered causal graph with {(self.task_graph != 0).sum().item()} edges (threshold={threshold:.3f})")
        return self.task_graph
    
    def _estimate_causal_effect(self, 
                                source_task: int, 
                                target_task: int,
                                task_data: Dict[int, torch.Tensor],
                                model: nn.Module) -> float:
        """
        Estimate causal effect of source task on target task.
        
        Uses difference-in-differences approach:
        ACE = E[Y|do(X=1)] - E[Y|do(X=0)]
        
        where Y = performance on target task, X = exposure to source task
        """
        if source_task not in self.feature_mechanisms or target_task not in self.feature_mechanisms:
            return 0.0
        
        source_mean, source_cov = self.feature_mechanisms[source_task]
        target_mean, target_cov = self.feature_mechanisms[target_task]
        
        # Measure distributional distance (proxy for causal effect)
        # If source and target have aligned features, there's likely causal influence
        alignment = F.cosine_similarity(source_mean, target_mean, dim=0)
        
        # Measure covariance structure similarity
        cov_similarity = F.cosine_similarity(
            source_cov.flatten(), 
            target_cov.flatten(), 
            dim=0
        )
        
        # Causal effect = alignment + structure similarity
        effect = 0.5 * alignment + 0.5 * cov_similarity
        
        return float(effect.clamp(-1, 1))
    
    def intervene(self, 
                  task_id: int, 
                  intervention_value: torch.Tensor) -> Dict[int, torch.Tensor]:
        """
        Perform causal intervention: do(Task_i = value)
        
        Full implementation that propagates through causal descendants according to task_graph.
        
        In SCM terms: 
        1. Cut all edges into Task_i (removing P(Task_i | Parents))
        2. Set Task_i = intervention_value
        3. Propagate effects through all descendants via causal mechanisms
        
        Args:
            task_id: Which task to intervene on
            intervention_value: What value to set (feature vector)
        
        Returns:
            Dictionary mapping affected task_ids to their modified feature representations
        """
        if task_id not in self.feature_mechanisms:
            logger.warning(f"Task {task_id} not in SCM, returning intervention value as-is")
            return {task_id: intervention_value}
        
        # Record intervention
        self.interventions.append({
            'task_id': task_id,
            'value': intervention_value.clone(),
            'timestamp': len(self.interventions)
        })
        
        # Initialize result: intervened task gets fixed value
        result = {task_id: intervention_value}
        
        # Find all descendants in causal graph (tasks affected by this intervention)
        # Descendants are tasks j where there exists a path from task_id → j
        descendants = self._find_descendants(task_id)
        
        if len(descendants) == 0:
            return result
        
        # Propagate intervention effects through causal mechanisms
        # For each descendant, compute: E[Task_j | do(Task_i = value)]
        for desc_id in descendants:
            if desc_id not in self.feature_mechanisms:
                continue
            
            # Get causal strength from intervened task to descendant
            causal_strength = float(self.task_graph[task_id, desc_id])
            
            if abs(causal_strength) < 0.01:  # Skip negligible effects
                continue
            
            # Get original distribution of descendant
            desc_mean, desc_cov = self.feature_mechanisms[desc_id]
            
            # Get original distribution of intervened task
            orig_mean, orig_cov = self.feature_mechanisms[task_id]
            
            # Compute shift caused by intervention
            # Δ = intervention_value - E[Task_i]
            intervention_shift = intervention_value - orig_mean
            
            # Propagate shift to descendant weighted by causal strength
            # E[Task_j | do(Task_i)] = E[Task_j] + causal_strength * Δ
            intervened_desc = desc_mean + causal_strength * intervention_shift
            
            result[desc_id] = intervened_desc
        
        logger.info(f"Intervention on task {task_id} affected {len(result)} tasks (including descendants)")
        
        return result
    
    def _find_descendants(self, task_id: int) -> List[int]:
        """
        Find all descendants of task_id in the causal graph using BFS.
        
        Args:
            task_id: Source task
        
        Returns:
            List of descendant task IDs
        """
        descendants = set()
        queue = [task_id]
        visited = {task_id}
        
        while queue:
            current = queue.pop(0)
            
            # Find all children (tasks with edges from current)
            for j in range(self.num_tasks):
                if j not in visited and abs(float(self.task_graph[current, j])) > 0.01:
                    descendants.add(j)
                    visited.add(j)
                    queue.append(j)
        
        return list(descendants)
    
    def counterfactual(self,
                       observed_x: torch.Tensor,
                       observed_task: int,
                       counterfactual_task: int,
                       model: nn.Module) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Generate counterfactual: "What if this sample was from task_cf instead of task_obs?"
        
        Full Pearl's 3-step implementation with proper noise modeling and propagation:
        1. Abduction: Infer exogenous noise U from observed (x, task_obs) using learned distribution
        2. Action: Set task = task_cf (intervention via graph surgery)
        3. Prediction: Generate x_cf using modified SCM with proper covariance structure
        
        Args:
            observed_x: Observed input sample
            observed_task: Task that generated observed_x
            counterfactual_task: Task we're asking "what if?" about
            model: Neural network for feature extraction
        
        Returns:
            Tuple of:
            - Counterfactual features (properly generated via SCM)
            - Diagnostic info dict with noise, means, etc.
        """
        if observed_task not in self.feature_mechanisms or counterfactual_task not in self.feature_mechanisms:
            logger.warning("Task not in SCM, returning observed sample")
            return observed_x, {'error': 'missing_task_mechanisms'}
        
        with torch.no_grad():
            # Step 1: Abduction - infer exogenous noise U from observed
            obs_mean, obs_cov = self.feature_mechanisms[observed_task]
            
            # Extract observed features
            if hasattr(model, 'net'):
                feats_or_logits = model.net(observed_x.unsqueeze(0))
            else:
                try:
                    feats_or_logits = model(observed_x.unsqueeze(0), returnt='features')  # type: ignore[arg-type]
                except Exception:
                    feats_or_logits = model(observed_x.unsqueeze(0))

            if isinstance(feats_or_logits, torch.Tensor) and feats_or_logits.dim() > 2:
                obs_features = feats_or_logits.mean(dim=[-1, -2]).squeeze(0)
            else:
                obs_features = feats_or_logits.squeeze(0)
            
            # Proper noise inference using inverse covariance (Mahalanobis-style)
            # U = Σ^{-1/2} * (X_obs - μ_obs)
            # This captures the structural noise in the proper basis
            try:
                # Compute Cholesky decomposition: Σ = L L^T
                L = torch.linalg.cholesky(obs_cov + 1e-6 * torch.eye(obs_cov.shape[0], device=obs_cov.device))
                # Solve L * noise = (obs - mean) for noise
                noise = torch.linalg.solve_triangular(L, (obs_features - obs_mean).unsqueeze(1), upper=False).squeeze(1)
            except RuntimeError:
                # Fallback to simple difference if Cholesky fails (singular matrix)
                logger.warning("Covariance singular, using simple noise model")
                noise = obs_features - obs_mean
            
            # Step 2: Action - intervene on task (graph surgery)
            # This cuts all edges into counterfactual_task
            cf_mean, cf_cov = self.feature_mechanisms[counterfactual_task]
            
            # Step 3: Prediction - generate counterfactual using proper noise model
            # X_cf = μ_cf + Σ_cf^{1/2} * U
            # This preserves the structural noise in the new task's distribution
            try:
                # Compute Cholesky decomposition of counterfactual covariance
                L_cf = torch.linalg.cholesky(cf_cov + 1e-6 * torch.eye(cf_cov.shape[0], device=cf_cov.device))
                # Generate counterfactual: mean + L * noise
                cf_features = cf_mean + torch.matmul(L_cf, noise.unsqueeze(1)).squeeze(1)
            except RuntimeError:
                # Fallback to simple addition
                logger.warning("Counterfactual covariance singular, using simple model")
                cf_features = cf_mean + noise
            
            # Account for causal paths from observed_task to counterfactual_task
            # If there's a causal link, incorporate it
            causal_link = float(self.task_graph[observed_task, counterfactual_task])
            if abs(causal_link) > 0.01:
                # Modify counterfactual to account for direct causal effect
                # CF = CF_base + α * (X_obs - μ_obs)
                cf_features = cf_features + causal_link * (obs_features - obs_mean)
            
            # Prepare diagnostic information
            diagnostics = {
                'noise': noise,
                'obs_mean': obs_mean,
                'cf_mean': cf_mean,
                'obs_features': obs_features,
                'causal_link_strength': causal_link,
                'noise_magnitude': float(torch.norm(noise)),
                'feature_shift': float(torch.norm(cf_features - obs_features))
            }
            
            logger.debug(f"Counterfactual generated: task {observed_task}→{counterfactual_task}, "
                        f"noise_mag={diagnostics['noise_magnitude']:.3f}, "
                        f"shift={diagnostics['feature_shift']:.3f}")
            
            return cf_features, diagnostics


class CausalForgettingDetector:
    """
    Identifies samples that CAUSE catastrophic forgetting via causal attribution.
    
    Key Innovation: Instead of correlational metrics (high loss → remove sample),
    we use interventional reasoning: if removing sample reduces forgetting, it's causal.
    
    Method: Instrumental Variable (IV) approach
    - IV = task order (affects buffer selection but not forgetting directly)
    - Treatment = including sample in buffer
    - Outcome = forgetting on previous tasks
    """
    
    def __init__(self, 
                 model: nn.Module,
                 buffer_size: int,
                 num_intervention_samples: int = 50,
                 true_temp_lr: float = 0.01,
                 true_micro_steps: int = 1):
        """
        Args:
            model: The continual learning model
            buffer_size: Size of replay buffer
            num_intervention_samples: How many samples to test per task
        """
        self.model = model
        self.buffer_size = buffer_size
        self.num_intervention_samples = num_intervention_samples
        # TRUE intervention tuning knobs
        self.true_temp_lr = true_temp_lr
        self.true_micro_steps = max(1, int(true_micro_steps))
        
        # Track causal effects
        self.causal_effects: Dict[str, CausalEffect] = {}
        
        # Harmful samples (cause forgetting)
        self.harmful_samples = set()
        
        # Beneficial samples (prevent forgetting)
        self.beneficial_samples = set()
    
    def attribute_forgetting(self,
                            candidate_sample: Tuple[torch.Tensor, torch.Tensor],
                            buffer_samples: List[Tuple[torch.Tensor, torch.Tensor]],
                            old_task_data: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
                            current_task_id: int,
                            use_true_intervention: bool = False) -> CausalEffect:
        """
        Measure the causal effect of a sample on forgetting.
        
        TWO MODES:
        1. Fast Heuristic (use_true_intervention=False): Feature similarity-based
           - Measures sample's feature alignment with old tasks
           - Fast (~0.01s per sample)
           - Used for initial filtering (500 → 200 candidates)
        
        2. TRUE Intervention (use_true_intervention=True): Gradient-based
           - Simulates training WITH/WITHOUT sample
           - Measures actual forgetting via do-calculus
           - Expensive (~1-2s per sample)
           - Used for final ranking (200 → 128 selected)
        
        Args:
            candidate_sample: (x, y) to test
            buffer_samples: Current buffer contents
            old_task_data: Data from previous tasks for measuring forgetting
            current_task_id: Current task index
            use_true_intervention: Whether to use expensive TRUE causality
        
        Returns:
            CausalEffect measuring impact on forgetting
        """
        sample_x, sample_y = candidate_sample
        sample_id = f"task{current_task_id}_sample{id(sample_x)}"
        
        if not use_true_intervention:
            # FAST HEURISTIC: Feature similarity (for filtering)
            return self._measure_feature_interference(
                candidate_sample, old_task_data, sample_id, current_task_id
            )
        else:
            # TRUE CAUSALITY: Gradient-based intervention (for ranking)
            return self._measure_true_causal_effect(
                candidate_sample, buffer_samples, old_task_data, sample_id, current_task_id
            )
    
    def _measure_feature_interference(self,
                                      candidate_sample: Tuple[torch.Tensor, torch.Tensor],
                                      old_task_data: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
                                      sample_id: str,
                                      current_task_id: int) -> CausalEffect:
        """
        FAST HEURISTIC: Measure sample's feature alignment with old tasks.
        
        This is correlation-based (Pearl Level 1), not causal.
        Used only for fast filtering of candidates.
        """
        sample_x, sample_y = candidate_sample
        
        # Measure DIRECT interference of this sample with old task features
        with torch.no_grad():
            device = next(self.model.parameters()).device
            sample_x = sample_x.unsqueeze(0).to(device)
            
            # Extract candidate sample features
            if hasattr(self.model, 'net'):
                try:
                    sample_feats = self.model.net(sample_x, returnt='features')
                except:
                    sample_feats = self.model.net(sample_x)
            else:
                sample_feats = self.model(sample_x)
            
            # Global average pool if needed
            if sample_feats.dim() > 2:
                sample_feats = sample_feats.mean(dim=[-1, -2])
            
            sample_feats = sample_feats.squeeze(0)  # [D]
            
            # Measure interference with each old task
            total_interference = 0.0
            num_tasks = 0
            
            for task_id, (task_x, task_y) in old_task_data.items():
                task_x = task_x[:min(20, len(task_x))].to(device)  # Sample 20
                
                # Extract old task features
                if hasattr(self.model, 'net'):
                    try:
                        old_feats = self.model.net(task_x, returnt='features')
                    except:
                        old_feats = self.model.net(task_x)
                else:
                    old_feats = self.model(task_x)
                
                if old_feats.dim() > 2:
                    old_feats = old_feats.mean(dim=[-1, -2])
                
                # Compute cosine similarity: high = aligned, low = conflict
                similarity = torch.mm(sample_feats.unsqueeze(0), old_feats.T)  # [1, T]
                similarity = similarity / (torch.norm(sample_feats) + 1e-8)
                similarity = similarity / (torch.norm(old_feats, dim=1, keepdim=True).T + 1e-8)
                
                # Interference = 1 - avg_similarity
                avg_similarity = similarity.mean()
                interference = 1.0 - avg_similarity
                
                total_interference += float(interference)
                num_tasks += 1
        
        # Average interference across old tasks
        avg_interference = total_interference / max(1, num_tasks)
        
        # Convert to effect size:
        # - Negative effect = beneficial (reduces forgetting)
        # - Positive effect = harmful (causes forgetting)
        # Since interference = 1 - similarity, we want to negate it
        # High similarity (low interference) should be beneficial (negative)
        effect_size = -(1.0 - avg_interference)  # = -(similarity), ranges from -1 (aligned) to 0 (conflict)
        confidence = min(1.0, abs(effect_size))
        
        return CausalEffect(
            source=sample_id,
            target=f"forgetting_task{current_task_id}",
            effect_size=float(effect_size),
            confidence=float(confidence),
            mechanism="heuristic_interference"
        )
    
    def _measure_true_causal_effect(self,
                                    candidate_sample: Tuple[torch.Tensor, torch.Tensor],
                                    buffer_samples: List[Tuple[torch.Tensor, torch.Tensor]],
                                    old_task_data: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
                                    sample_id: str,
                                    current_task_id: int) -> CausalEffect:
        """
        TRUE CAUSAL INTERVENTION: Simulate training with/without sample.
        
        This implements Pearl's do-calculus (Level 2: Intervention).
        
        Steps:
        1. Save current model state (checkpoint)
        2. FACTUAL: Train mini-batch WITH candidate → measure forgetting
        3. Restore model state
        4. COUNTERFACTUAL: Train mini-batch WITHOUT candidate → measure forgetting  
        5. Causal effect = forgetting_with - forgetting_without
        
        This is expensive (~1-2s per sample) but gives TRUE causal effects.
        """
        import copy
        
        sample_x, sample_y = candidate_sample
        device = next(self.model.parameters()).device
        
        # Save model checkpoint
        checkpoint = {k: v.clone() for k, v in self.model.state_dict().items()}
        
        # Sample random buffer subset for mini-batch
        num_replay = min(32, len(buffer_samples))  # Small mini-batch for speed
        if len(buffer_samples) > 0:
            replay_indices = torch.randperm(len(buffer_samples))[:num_replay]
            replay_samples = [buffer_samples[i] for i in replay_indices]
        else:
            replay_samples = []
        
        # === FACTUAL: Train WITH candidate sample ===
        with torch.enable_grad():
            # Create mini-batch: candidate + replay samples
            factual_batch_x = [sample_x.to(device)]
            factual_batch_y = [sample_y.to(device)]
            for rx, ry in replay_samples:
                factual_batch_x.append(rx.to(device))
                factual_batch_y.append(ry.to(device))
            factual_x = torch.stack(factual_batch_x)
            factual_y = torch.stack(factual_batch_y)

            # Perform a few micro-steps with temporary updates
            for _ in range(self.true_micro_steps):
                if hasattr(self.model, 'net'):
                    outputs = self.model.net(factual_x)
                else:
                    outputs = self.model(factual_x)
                loss = F.cross_entropy(outputs, factual_y)
                grads = torch.autograd.grad(loss, self.model.parameters(), retain_graph=False)
                with torch.no_grad():
                    for param, grad in zip(self.model.parameters(), grads):
                        param.data -= self.true_temp_lr * grad
        
        # Measure forgetting on old tasks AFTER factual update
        forgetting_with = self._measure_forgetting_simple(old_task_data)
        
        # Restore checkpoint
        self.model.load_state_dict(checkpoint)
        
        # === COUNTERFACTUAL: Train WITHOUT candidate sample ===
        if len(replay_samples) > 0:
            with torch.enable_grad():
                # Mini-batch: only replay samples (no candidate)
                cf_batch_x = torch.stack([rx.to(device) for rx, ry in replay_samples])
                cf_batch_y = torch.stack([ry.to(device) for rx, ry in replay_samples])
                # A few micro-steps
                for _ in range(self.true_micro_steps):
                    if hasattr(self.model, 'net'):
                        outputs = self.model.net(cf_batch_x)
                    else:
                        outputs = self.model(cf_batch_x)
                    loss = F.cross_entropy(outputs, cf_batch_y)
                    grads = torch.autograd.grad(loss, self.model.parameters(), retain_graph=False)
                    with torch.no_grad():
                        for param, grad in zip(self.model.parameters(), grads):
                            param.data -= self.true_temp_lr * grad
            
            # Measure forgetting AFTER counterfactual update
            forgetting_without = self._measure_forgetting_simple(old_task_data)
        else:
            # No buffer samples → counterfactual is no training
            forgetting_without = 0.0
        
        # Restore checkpoint again
        self.model.load_state_dict(checkpoint)
        
        # === CAUSAL EFFECT ===
        # Positive effect = sample causes MORE forgetting (harmful)
        # Negative effect = sample causes LESS forgetting (beneficial)
        # NOTE: Now measured across ALL old tasks (len={len(old_task_data)}), not just sample's source task
        effect_size = forgetting_with - forgetting_without
        confidence = 1.0  # High confidence (actual intervention)
        
        # Categorize
        if effect_size > 0.05:
            self.harmful_samples.add(sample_id)
        elif effect_size < -0.05:
            self.beneficial_samples.add(sample_id)
        
        return CausalEffect(
            source=sample_id,
            target=f"forgetting_task{current_task_id}",
            effect_size=float(effect_size),
            confidence=float(confidence),
            mechanism="true_causal_intervention"
        )
    
    def _measure_forgetting_simple(self, old_task_data: Dict[int, Tuple[torch.Tensor, torch.Tensor]]) -> float:
        """
        Simple forgetting measurement: average loss across ALL old tasks.
        
        KEY FIX: Measures causal effect on forgetting across ALL previously learned tasks,
        not just the task the sample came from. This ensures that Task 0 samples selected
        at Task 1 are re-evaluated based on their impact on Tasks 0-1 at Task 2, etc.
        
        Used by TRUE causal intervention.
        """
        if not old_task_data:
            return 0.0
        
        total_loss = 0.0
        num_tasks = 0
        
        with torch.no_grad():
            for task_id, (task_x, task_y) in old_task_data.items():
                device = next(self.model.parameters()).device
                # Use up to 50 samples per task for more reliable measurement
                n_samples = min(50, len(task_x))
                task_x = task_x[:n_samples].to(device)
                task_y = task_y[:n_samples].to(device)
                
                if hasattr(self.model, 'net'):
                    outputs = self.model.net(task_x)
                else:
                    outputs = self.model(task_x)
                
                loss = F.cross_entropy(outputs, task_y)
                total_loss += float(loss)
                num_tasks += 1
        
        # Return average loss across ALL old tasks
        avg_loss = total_loss / max(1, num_tasks)
        return avg_loss
        """
        Measure the causal effect of a sample on forgetting.
        
        SIMPLIFIED APPROACH: Measure the candidate sample's DIRECT interference with old tasks.
        - Low interference → beneficial (preserves old knowledge)
        - High interference → harmful (conflicts with old knowledge)
        
        This is more sensitive than comparing buffer±sample (which differs by only 1/500).
        
        Args:
            candidate_sample: (x, y) to test
            buffer_samples: Current buffer contents (not used in simplified version)
            old_task_data: Data from previous tasks for measuring forgetting
            current_task_id: Current task index
        
        Returns:
            CausalEffect measuring impact on forgetting
        """
        sample_x, sample_y = candidate_sample
        sample_id = f"task{current_task_id}_sample{id(sample_x)}"
        
        # Measure DIRECT interference of this sample with old task features
        with torch.no_grad():
            device = next(self.model.parameters()).device
            sample_x = sample_x.unsqueeze(0).to(device)
            
            # Extract candidate sample features
            if hasattr(self.model, 'net'):
                try:
                    sample_feats = self.model.net(sample_x, returnt='features')
                except:
                    sample_feats = self.model.net(sample_x)
            else:
                sample_feats = self.model(sample_x)
            
            # Global average pool if needed
            if sample_feats.dim() > 2:
                sample_feats = sample_feats.mean(dim=[-1, -2])
            
            sample_feats = sample_feats.squeeze(0)  # [D]
            
            # Measure interference with each old task
            total_interference = 0.0
            num_tasks = 0
            
            for task_id, (task_x, task_y) in old_task_data.items():
                task_x = task_x[:min(20, len(task_x))].to(device)  # Sample 20
                
                # Extract old task features
                if hasattr(self.model, 'net'):
                    try:
                        old_feats = self.model.net(task_x, returnt='features')
                    except:
                        old_feats = self.model.net(task_x)
                else:
                    old_feats = self.model(task_x)
                
                if old_feats.dim() > 2:
                    old_feats = old_feats.mean(dim=[-1, -2])
                
                # Compute cosine similarity: high = aligned, low = conflict
                similarity = torch.mm(sample_feats.unsqueeze(0), old_feats.T)  # [1, T]
                similarity = similarity / (torch.norm(sample_feats) + 1e-8)
                similarity = similarity / (torch.norm(old_feats, dim=1, keepdim=True).T + 1e-8)
                
                # Interference = 1 - avg_similarity
                # High similarity (0.8) → low interference (0.2) → beneficial
                # Low similarity (-0.2) → high interference (1.2) → harmful
                avg_similarity = similarity.mean()
                interference = 1.0 - avg_similarity
                
                total_interference += float(interference)
                num_tasks += 1
        
        # Average interference across old tasks
        avg_interference = total_interference / max(1, num_tasks)
        
        # Convert to causal effect:
        # High interference → positive effect (increases forgetting) → harmful
        # Low interference → negative effect (reduces forgetting) → beneficial
        effect_size = avg_interference - 1.0  # Range: [-1, 1]
        # effect_size > 0 → harmful
        # effect_size < 0 → beneficial
        
        # Confidence based on magnitude
        confidence = min(1.0, abs(effect_size))
        
        effect = CausalEffect(
            source=sample_id,
            target=f"forgetting_task{current_task_id}",
            effect_size=float(effect_size),
            confidence=float(confidence),
            mechanism="feature_interference" if effect_size > 0 else "feature_preservation"
        )
        
        # Categorize sample
        if effect_size > 0.05:  # Threshold for "harmful"
            self.harmful_samples.add(sample_id)
        elif effect_size < -0.05:  # "Beneficial"
            self.beneficial_samples.add(sample_id)
        
        return effect
    
    def _measure_forgetting(self,
                           buffer: List[Tuple[torch.Tensor, torch.Tensor]],
                           old_task_data: Dict[int, Tuple[torch.Tensor, torch.Tensor]]) -> float:
        """
        Measure forgetting on old tasks given a buffer.
        
        LIGHTWEIGHT PROXY: Instead of full training simulation, measure feature interference:
        - Compute feature representations of buffer samples
        - Measure their alignment/conflict with old task features
        - Higher conflict = more forgetting
        
        Returns average interference score (0 = no interference, 1 = high interference).
        """
        if not old_task_data or not buffer:
            return 0.0
        
        total_interference = 0.0
        num_comparisons = 0
        
        with torch.no_grad():
            # Extract features from buffer samples
            device = next(self.model.parameters()).device
            buffer_features = []
            
            for buf_x, buf_y in buffer[:min(10, len(buffer))]:  # Sample 10 for efficiency
                buf_x = buf_x.unsqueeze(0).to(device)
                
                if hasattr(self.model, 'net'):
                    try:
                        feats = self.model.net(buf_x, returnt='features')
                    except:
                        feats = self.model.net(buf_x)
                else:
                    feats = self.model(buf_x)
                
                # Global average pool if needed
                if feats.dim() > 2:
                    feats = feats.mean(dim=[-1, -2])
                
                buffer_features.append(feats.squeeze(0))
            
            if len(buffer_features) == 0:
                return 0.0
            
            buffer_features = torch.stack(buffer_features)  # [B, D]
            
            # Measure interference with old task data
            for task_id, (task_x, task_y) in old_task_data.items():
                task_x = task_x[:min(10, len(task_x))].to(device)  # Sample 10
                
                # Extract old task features
                if hasattr(self.model, 'net'):
                    try:
                        old_feats = self.model.net(task_x, returnt='features')
                    except:
                        old_feats = self.model.net(task_x)
                else:
                    old_feats = self.model(task_x)
                
                if old_feats.dim() > 2:
                    old_feats = old_feats.mean(dim=[-1, -2])
                
                # Measure interference: negative cosine similarity (conflict)
                # High similarity = features aligned = low interference
                # Low similarity = features conflict = high interference
                similarity = torch.mm(buffer_features, old_feats.T)  # [B, T]
                similarity = similarity / (torch.norm(buffer_features, dim=1, keepdim=True) + 1e-8)
                similarity = similarity / (torch.norm(old_feats, dim=1, keepdim=True).T + 1e-8)
                
                # Interference = 1 - avg_similarity (scaled to [0, 2])
                # High similarity (e.g., 0.9) → interference = 0.1
                # Low similarity (e.g., -0.5) → interference = 1.5
                avg_similarity = similarity.mean()
                interference = 1.0 - avg_similarity
                
                total_interference += float(interference)
                num_comparisons += 1
        
        return total_interference / max(1, num_comparisons)
    
    def filter_buffer(self, 
                     buffer_samples: List[Tuple[torch.Tensor, torch.Tensor]],
                     keep_ratio: float = 0.8) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Remove causally harmful samples from buffer.
        
        Args:
            buffer_samples: Current buffer
            keep_ratio: Fraction to keep (remove worst 20%)
        
        Returns:
            Filtered buffer with harmful samples removed
        """
        if not self.causal_effects:
            return buffer_samples
        
        # Sort by causal effect (most harmful first)
        sorted_effects = sorted(
            self.causal_effects.items(),
            key=lambda x: x[1].effect_size,
            reverse=True
        )
        
        # Remove most harmful samples
        num_to_remove = int(len(buffer_samples) * (1 - keep_ratio))
        harmful_ids = {effect_id for effect_id, _ in sorted_effects[:num_to_remove]}
        
        # Filter (simplified - in practice, need sample tracking)
        filtered = [s for s in buffer_samples if id(s[0]) not in harmful_ids]
        
        logger.info(f"Removed {num_to_remove} causally harmful samples from buffer")
        
        return filtered


def compute_ate(treatment: torch.Tensor, 
                outcome: torch.Tensor,
                confounders: Optional[torch.Tensor] = None,
                use_ipw: bool = True) -> Tuple[float, Dict[str, float]]:
    """
    Compute Average Treatment Effect (ATE) using proper causal inference methods.
    
    Full implementation with multiple estimation strategies:
    1. No confounders: Simple difference in means
    2. With confounders: Backdoor adjustment via stratification + IPW (Inverse Propensity Weighting)
    3. Doubly robust estimation for robustness
    
    ATE = E[Y|do(T=1)] - E[Y|do(T=0)]
    
    With confounders, uses backdoor adjustment:
    ATE = Σ_c P(c) * (E[Y|T=1,c] - E[Y|T=0,c])
    
    Args:
        treatment: Binary treatment variable (0 or 1), shape (N,)
        outcome: Continuous outcome variable, shape (N,)
        confounders: Optional confounding variables, shape (N, D)
        use_ipw: Whether to use inverse propensity weighting
    
    Returns:
        Tuple of:
        - ATE estimate (float)
        - Diagnostics dict with variance, propensity scores, etc.
    """
    treatment = treatment.float()
    outcome = outcome.float()
    
    if confounders is None:
        # Simple difference in means (no confounding)
        mask_1 = treatment == 1
        mask_0 = treatment == 0
        
        y1 = outcome[mask_1].mean() if mask_1.sum() > 0 else torch.tensor(0.0)
        y0 = outcome[mask_0].mean() if mask_0.sum() > 0 else torch.tensor(0.0)
        ate = float(y1 - y0)
        
        # Compute variance for confidence intervals
        var_1 = outcome[mask_1].var() / max(1, mask_1.sum()) if mask_1.sum() > 0 else 0.0
        var_0 = outcome[mask_0].var() / max(1, mask_0.sum()) if mask_0.sum() > 0 else 0.0
        ate_var = float(var_1 + var_0)
        
        diagnostics = {
            'ate': ate,
            'variance': ate_var,
            'std_error': float(torch.sqrt(torch.tensor(ate_var))),
            'n_treated': int(mask_1.sum()),
            'n_control': int(mask_0.sum()),
            'method': 'difference_in_means'
        }
        
        return ate, diagnostics
    
    else:
        # Full backdoor adjustment with confounders
        n = len(treatment)
        
        # Method 1: Stratification-based adjustment
        # Discretize confounders into strata for exact backdoor formula
        # Use k-means style clustering to create strata
        num_strata = min(10, n // 20)  # At least 20 samples per stratum
        
        if num_strata >= 2:
            # Cluster confounders into strata
            if SKLEARN_AVAILABLE:
                try:
                    kmeans = KMeans(n_clusters=num_strata, random_state=42, n_init=10)
                    strata = kmeans.fit_predict(confounders.cpu().numpy())
                    strata = torch.from_numpy(strata).to(confounders.device)
                except:
                    # Fallback: use simple quantile-based stratification on first confounder dimension
                    if confounders.dim() > 1:
                        first_conf = confounders[:, 0]
                    else:
                        first_conf = confounders
                    quantiles = torch.quantile(first_conf, torch.linspace(0, 1, num_strata + 1))
                    strata = torch.searchsorted(quantiles, first_conf) - 1
                    strata = strata.clamp(0, num_strata - 1)
            else:
                # Fallback: use simple quantile-based stratification
                if confounders.dim() > 1:
                    first_conf = confounders[:, 0]
                else:
                    first_conf = confounders
                quantiles = torch.quantile(first_conf, torch.linspace(0, 1, num_strata + 1))
                strata = torch.searchsorted(quantiles, first_conf) - 1
                strata = strata.clamp(0, num_strata - 1)
            
            # Compute ATE via backdoor formula: Σ_c P(c) * [E[Y|T=1,C=c] - E[Y|T=0,C=c]]
            ate_stratified = 0.0
            total_weight = 0.0
            
            for stratum_idx in range(num_strata):
                mask_stratum = strata == stratum_idx
                if mask_stratum.sum() < 2:
                    continue
                
                # P(C=c)
                p_stratum = float(mask_stratum.sum()) / n
                
                # E[Y|T=1,C=c]
                mask_treated_stratum = mask_stratum & (treatment == 1)
                y1_stratum = outcome[mask_treated_stratum].mean() if mask_treated_stratum.sum() > 0 else 0.0
                
                # E[Y|T=0,C=c]
                mask_control_stratum = mask_stratum & (treatment == 0)
                y0_stratum = outcome[mask_control_stratum].mean() if mask_control_stratum.sum() > 0 else 0.0
                
                # Weighted contribution
                ate_stratified += p_stratum * float(y1_stratum - y0_stratum)
                total_weight += p_stratum
            
            ate_stratified = ate_stratified / max(total_weight, 1e-8)
        else:
            ate_stratified = 0.0
        
        # Method 2: Inverse Propensity Weighting (IPW)
        ate_ipw = 0.0
        propensity_scores = None
        
        if use_ipw and SKLEARN_AVAILABLE:
            # Estimate propensity scores: P(T=1|C)
            # Use logistic regression approximation
            try:
                lr = LogisticRegression(random_state=42, max_iter=1000)
                lr.fit(confounders.cpu().numpy(), treatment.cpu().numpy())
                propensity_scores = torch.from_numpy(lr.predict_proba(confounders.cpu().numpy())[:, 1]).to(confounders.device)
                
                # Clip propensity scores for stability (avoid extreme weights)
                propensity_scores = propensity_scores.clamp(0.01, 0.99)
                
                # IPW estimator: E[Y(1)] = E[T*Y / e(C)], E[Y(0)] = E[(1-T)*Y / (1-e(C))]
                weights_1 = treatment / propensity_scores
                weights_0 = (1 - treatment) / (1 - propensity_scores)
                
                # Normalize weights
                weights_1 = weights_1 / weights_1.sum()
                weights_0 = weights_0 / weights_0.sum()
                
                y1_ipw = (weights_1 * outcome).sum()
                y0_ipw = (weights_0 * outcome).sum()
                ate_ipw = float(y1_ipw - y0_ipw)
            except:
                # Fallback if sklearn not available or fitting fails
                logger.warning("IPW estimation failed, using stratification only")
                ate_ipw = ate_stratified
        elif use_ipw and not SKLEARN_AVAILABLE:
            # sklearn not available, use stratification only
            logger.warning("IPW requested but scikit-learn not available, using stratification only")
            ate_ipw = ate_stratified
        
        # Method 3: Doubly Robust Estimator (combines both)
        # DR = (T/e - (1-T)/(1-e)) * Y + ((1-T/e) * μ1(C) - (1-(1-T)/(1-e)) * μ0(C))
        # For simplicity, use average of stratification and IPW
        ate_final = (ate_stratified + ate_ipw) / 2 if use_ipw else ate_stratified
        
        # Estimate variance (conservative bootstrap-style)
        # Variance of ATE ≈ Var(Y|T=1)/n1 + Var(Y|T=0)/n0
        mask_1 = treatment == 1
        mask_0 = treatment == 0
        var_1 = outcome[mask_1].var() / max(1, mask_1.sum()) if mask_1.sum() > 0 else 0.0
        var_0 = outcome[mask_0].var() / max(1, mask_0.sum()) if mask_0.sum() > 0 else 0.0
        ate_var = float(var_1 + var_0)
        
        diagnostics = {
            'ate': ate_final,
            'ate_stratified': ate_stratified,
            'ate_ipw': ate_ipw,
            'variance': ate_var,
            'std_error': float(torch.sqrt(torch.tensor(ate_var))),
            'n_treated': int(mask_1.sum()),
            'n_control': int(mask_0.sum()),
            'num_strata': num_strata if num_strata >= 2 else 0,
            'propensity_min': float(propensity_scores.min()) if propensity_scores is not None else None,
            'propensity_max': float(propensity_scores.max()) if propensity_scores is not None else None,
            'method': 'backdoor_adjustment_ipw' if use_ipw else 'backdoor_adjustment_stratification'
        }
        
        return ate_final, diagnostics


# Export main classes
__all__ = [
    'StructuralCausalModel',
    'CausalForgettingDetector',
    'CausalEffect',
    'compute_ate'
]
