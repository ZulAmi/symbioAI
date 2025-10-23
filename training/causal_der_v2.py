"""
training/causal_der_v2.py - Clean DER++ Baseline for Phase 1 Validation
========================================================================

FIXED Oct 22, 2025:
- Uses minibatch_size parameter correctly
- Passes transform for data augmentation on replay samples
- Simple list-based buffer (matches official DER++)
- Exact copy of official DER++ loss computation

This is the BASELINE implementation for Phase 1 validation.
Target: Match official DER++ performance (73.81% Task-IL on CIFAR-100)

"""

import sys
from pathlib import Path

# Add mammoth utils to path for apply_transform
mammoth_path = Path(__file__).parent.parent / 'mammoth'
if str(mammoth_path) not in sys.path:
    sys.path.insert(0, str(mammoth_path))

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict

# Import Mammoth's apply_transform for correct batch transform handling
from utils.augmentations import apply_transform

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional

# Import comprehensive metrics tracker
try:
    from training.metrics_tracker import ContinualLearningMetrics, estimate_ate_for_sample
except ImportError:
    # Fallback if module not found
    ContinualLearningMetrics = None
    estimate_ate_for_sample = None

# Import causal inference modules for Phase 3: Graph Learning
try:
    from training.causal_inference import StructuralCausalModel, CausalForgettingDetector
except ImportError:
    try:
        # Try backup location
        sys.path.insert(0, str(Path(__file__).parent.parent / 'cleanup_backup_20251020_113904'))
        from causal_inference import StructuralCausalModel, CausalForgettingDetector
    except ImportError:
        StructuralCausalModel = None
        CausalForgettingDetector = None


class CausalDEREngine:
    """
    DER++ Engine with Phase 2: Causal Importance Scoring
    
    Phase 1 (VALIDATED ✅): Clean DER++ baseline - 70.19% Task-IL
    Phase 2 (CURRENT): Add importance-weighted sampling - Target: +1-2% (71-72%)
    
    Loss = CE(current) + α·MSE(replay_logits) + β·CE(replay_labels)
    
    Key Points:
    - Samples buffer TWICE (once for MSE, once for CE)
    - Uses minibatch_size parameter (not hardcoded!)
    - Stores logits for dark knowledge replay
    - **NEW: Importance-weighted sampling (70% important, 30% random)**
    - Reservoir sampling for buffer management
    """
    
    def __init__(
        self,
        alpha: float,
        beta: float,
        buffer_size: int,
        minibatch_size: int = 32,
        use_importance_sampling: bool = False,  # Phase 2 feature
        importance_weight: float = 0.7,  # 70% importance, 30% random
        enable_causal_graph_learning: bool = False,  # Phase 3 feature
        num_tasks: int = 10,  # Phase 3: Total number of tasks
        feature_dim: int = 512,  # Phase 3: Feature dimensionality
        temperature: float = 2.0,  # FIX 1: Temperature for KL divergence
        replay_warmup_tasks: int = 1,  # FIX 4: Skip replay for first N tasks
        initial_importance_weight: float = 0.3,  # FIX 3: Annealing start
        final_importance_weight: float = 0.6,  # FIX 3: Annealing end
        **kwargs  # Accept extra args for future compatibility
    ):
        """
        Initialize DER++ engine with Phase 2 & Phase 3 enhancements.
        
        Args:
            alpha: MSE distillation weight (actually KL divergence weight)
            beta: CE replay weight  
            buffer_size: Buffer capacity
            minibatch_size: Replay batch size (CRITICAL: must match batch_size!)
            use_importance_sampling: Enable importance-weighted sampling (Phase 2)
            importance_weight: Weight for importance vs random (default: 0.7)
            enable_causal_graph_learning: Learn causal graph between tasks (Phase 3)
            num_tasks: Total number of tasks in sequence (Phase 3)
            feature_dim: Dimensionality of learned features (Phase 3)
            temperature: Temperature scaling for KL divergence (FIX 1)
            replay_warmup_tasks: Number of tasks before replay starts (FIX 4)
            initial_importance_weight: Starting importance ratio (FIX 3)
            final_importance_weight: Final importance ratio (FIX 3)
        """
        self.alpha = alpha
        self.beta = beta
        self.buffer_size = buffer_size
        self.minibatch_size = minibatch_size
        self.use_importance_sampling = use_importance_sampling
        self.importance_weight = importance_weight
        
        # FIX 1: Temperature for stable KL divergence
        self.temperature = temperature
        
        # FIX 3: Importance annealing schedule
        self.initial_importance_weight = initial_importance_weight
        self.final_importance_weight = final_importance_weight
        self.current_task = 0
        
        # FIX 4: Replay warmup
        self.replay_warmup_tasks = replay_warmup_tasks
        
        # Simple list-based buffer (matches Mammoth's Buffer class behavior)
        self.buffer_data = []
        self.buffer_labels = []
        self.buffer_log_probs = []  # FIX 1: Store log-probs instead of raw logits
        self.buffer_importances = []  # Phase 2: Store importance scores
        self.buffer_task_ids = []  # Phase 3: Track which task each sample belongs to
        self.num_seen = 0  # For reservoir sampling
        
        # Statistics tracking
        self.importance_stats = {
            'mean_importance': 0.0,
            'min_importance': 0.0,
            'max_importance': 0.0,
            'num_high_importance': 0,
        }
        
        # Comprehensive metrics tracker (for publication)
        self.metrics_tracker = ContinualLearningMetrics(num_tasks=num_tasks) if ContinualLearningMetrics else None
        self.compute_ate_metrics = False  # Enable when needed (expensive)
        
        # Phase 3: Causal Graph Learning
        self.enable_causal_graph_learning = enable_causal_graph_learning
        self.num_tasks = num_tasks
        self.feature_dim = feature_dim
        self.causal_graph = None
        self.tasks_seen = 0
        
        # Initialize Structural Causal Model for graph learning
        if enable_causal_graph_learning and StructuralCausalModel is not None:
            self.scm = StructuralCausalModel(num_tasks=num_tasks, feature_dim=feature_dim)
            self.task_feature_cache = {}  # Cache features per task for graph learning
            print(f"[CausalDER-v2] ✅ Structural Causal Model initialized for graph learning")
        else:
            self.scm = None
            self.task_feature_cache = {}
        
        # Causal Forgetting Detector (Phase 3)
        self.causal_forgetting_detector = None
        self.enable_causal_forgetting_detector = enable_causal_graph_learning  # Enable with graph learning
        
        # Print configuration
        if enable_causal_graph_learning:
            phase = "Phase 3 (Causal Graph Learning)"
        elif use_importance_sampling:
            phase = "Phase 2 (Importance Sampling)"
        else:
            phase = "Phase 1 (Baseline)"
        
        print(f"[CausalDER-v2] {phase}")
        print(f"[CausalDER-v2] α={alpha}, β={beta}, buffer={buffer_size}, minibatch={minibatch_size}")
        print(f"[CausalDER-v2] FIX 1: KL divergence with T={temperature}")
        print(f"[CausalDER-v2] FIX 2: Importance noise 2% (was 10%)")
        print(f"[CausalDER-v2] FIX 3: Importance annealing {initial_importance_weight:.1%}→{final_importance_weight:.1%}")
        print(f"[CausalDER-v2] FIX 4: Replay warmup - skip first {replay_warmup_tasks} task(s)")
        if use_importance_sampling:
            print(f"[CausalDER-v2] Importance sampling: {importance_weight:.1%} by importance, {1-importance_weight:.1%} random")
        if enable_causal_graph_learning:
            print(f"[CausalDER-v2] Causal graph learning: ENABLED ({num_tasks} tasks, {feature_dim}D features)")
        if self.metrics_tracker:
            print(f"[CausalDER-v2] ✅ Comprehensive metrics tracking enabled")
    
    def is_empty(self) -> bool:
        """Check if buffer is empty."""
        return len(self.buffer_data) == 0
    
    def __len__(self) -> int:
        """Get current buffer size."""
        return len(self.buffer_data)
    
    def compute_importance(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute importance score for each sample (Phase 2).
        
        Importance = loss_normalized * (1 - confidence)^2 + noise
        
        High importance means:
        - High loss (model struggles)
        - Low confidence (model uncertain)
        - Should be replayed more often
        
        Args:
            logits: Model predictions (B, num_classes)
            labels: Ground truth labels (B,)
        
        Returns:
            Importance scores (B,) - higher = more important
        """
        # Compute loss per sample (no reduction)
        loss_per_sample = F.cross_entropy(logits, labels, reduction='none')
        
        # Normalize loss to [0, 1]
        loss_min = loss_per_sample.min()
        loss_max = loss_per_sample.max()
        loss_normalized = (loss_per_sample - loss_min) / (loss_max - loss_min + 1e-8)
        
        # Compute confidence (max probability)
        probs = F.softmax(logits, dim=1)
        confidence = probs.max(dim=1)[0]
        
        # Importance = loss * uncertainty^2 (multiplicative, emphasize both)
        # Square the uncertainty to emphasize low-confidence samples more
        uncertainty = 1 - confidence
        importance = loss_normalized * (uncertainty ** 2)
        
        # FIX 2: Reduce noise from 10% to 2% for better signal preservation
        importance += 0.02 * torch.rand_like(importance)
        
        return importance
    
    def add_data(
        self,
        examples: torch.Tensor,
        labels: torch.Tensor,
        logits: torch.Tensor,
        importances: Optional[torch.Tensor] = None,
        task_id: int = 0
    ):
        """
        Add data to buffer using reservoir sampling with importance scores (Phase 2).
        
        Args:
            examples: Input data (B, C, H, W)
            labels: Ground truth labels (B,)
            logits: Model predictions/logits (B, num_classes)
            importances: Importance scores (B,) - optional, computed if not provided
            task_id: Task ID for this data (Phase 3)
        """
        # Move to CPU for storage
        examples = examples.cpu()
        labels = labels.cpu()
        logits = logits.cpu()
        
        # FIX 1: Convert logits to log-probs (fp32) for stable KL divergence
        with torch.no_grad():
            log_probs = F.log_softmax(logits / self.temperature, dim=1).float()
        
        # Compute importance if not provided (Phase 2)
        if importances is None and self.use_importance_sampling:
            importances = self.compute_importance(logits, labels).cpu()
        elif importances is not None:
            importances = importances.cpu()
        
        for i in range(examples.shape[0]):
            importance_score = importances[i].item() if importances is not None else 1.0
            
            if len(self.buffer_data) < self.buffer_size:
                # Buffer not full, just append
                self.buffer_data.append(examples[i])
                self.buffer_labels.append(labels[i])
                self.buffer_log_probs.append(log_probs[i])
                self.buffer_importances.append(importance_score)
                self.buffer_task_ids.append(task_id)
            else:
                # Buffer full, reservoir sampling
                idx = torch.randint(0, self.num_seen + i + 1, (1,)).item()
                if idx < self.buffer_size:
                    self.buffer_data[idx] = examples[i]
                    self.buffer_labels[idx] = labels[i]
                    self.buffer_log_probs[idx] = log_probs[i]
                    self.buffer_importances[idx] = importance_score
                    self.buffer_task_ids[idx] = task_id
            
        self.num_seen += examples.shape[0]
        
        # Update importance statistics (Phase 2)
        if self.use_importance_sampling and len(self.buffer_importances) > 0:
            imp_tensor = torch.tensor(self.buffer_importances)
            self.importance_stats['mean_importance'] = float(imp_tensor.mean())
            self.importance_stats['min_importance'] = float(imp_tensor.min())
            self.importance_stats['max_importance'] = float(imp_tensor.max())
            self.importance_stats['num_high_importance'] = int((imp_tensor > imp_tensor.median()).sum())
    
    def get_data(
        self,
        size: int,
        device: torch.device,
        transform=None
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Sample data from buffer with importance-weighted sampling (Phase 2).
        
        Phase 1: Uniform random sampling
        Phase 2: Annealed importance-weighted + random (FIX 3)
        
        Args:
            size: Number of samples to retrieve
            device: Device to move data to
            transform: Optional data augmentation
        
        Returns:
            (data, labels, log_probs) tuple or (None, None, None) if empty
        """
        if self.is_empty():
            return None, None, None
        
        size = min(size, len(self.buffer_data))
        
        # FIX 3: Annealed importance sampling (0.3 → 0.6 over tasks)
        if self.use_importance_sampling and len(self.buffer_importances) > 0:
            # Compute current importance weight using linear annealing
            if self.num_tasks > 1:
                progress = self.current_task / (self.num_tasks - 1)
                current_weight = (
                    self.initial_importance_weight + 
                    progress * (self.final_importance_weight - self.initial_importance_weight)
                )
            else:
                current_weight = self.importance_weight
            
            # Split sampling: annealed% by importance, remaining% random
            n_importance = int(size * current_weight)
            n_random = size - n_importance
            
            # Sample by importance (weighted by importance scores)
            importances = torch.tensor(self.buffer_importances, dtype=torch.float32)
            probs = importances / importances.sum()
            
            importance_indices = torch.multinomial(
                probs, 
                num_samples=n_importance,
                replacement=False
            )
            
            # Sample randomly (uniform)
            remaining = list(set(range(len(self.buffer_data))) - set(importance_indices.tolist()))
            if n_random > 0 and len(remaining) > 0:
                random_indices = torch.tensor(
                    torch.randperm(len(remaining))[:n_random].tolist()
                )
                random_indices = torch.tensor([remaining[i] for i in random_indices])
                indices = torch.cat([importance_indices, random_indices])
            else:
                indices = importance_indices
        else:
            # Phase 1: Uniform random sampling
            indices = torch.randperm(len(self.buffer_data))[:size]
        
        # Gather samples
        data = torch.stack([self.buffer_data[i] for i in indices])
        labels = torch.stack([self.buffer_labels[i] for i in indices])
        log_probs = torch.stack([self.buffer_log_probs[i] for i in indices])
        
        # Apply transform if provided (before moving to device, like official buffer)
        if transform is not None:
            data = apply_transform(data, transform)
        
        # Move to device
        data = data.to(device)
        labels = labels.to(device)
        log_probs = log_probs.to(device)
        
        return data, labels, log_probs
    
    def compute_loss(
        self,
        model: nn.Module,
        data: torch.Tensor,
        target: torch.Tensor,
        output: torch.Tensor,
        task_id: int,
        transform=None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute DER++ loss with FIXED numerics.
        
        FIXES APPLIED:
        - FIX 1: Use KL divergence instead of MSE (with temperature)
        - FIX 4: Skip replay for first N tasks (warmup period)
        
        Loss = CE(current) + α·KL(replay_log_probs || current_log_probs) + β·CE(replay_labels)
        
        Args:
            model: The neural network
            data: Current batch inputs
            target: Current batch labels
            output: Model predictions on current batch (already computed)
            task_id: Current task ID
            transform: Data augmentation for replay
        
        Returns:
            (loss, info_dict) tuple
        """
        device = data.device
        
        # Current task CE loss (using F.cross_entropy like official)
        loss = F.cross_entropy(output, target)
        
        info = {
            'current_loss': float(loss.detach().cpu()),
            'replay_kl': 0.0,
            'replay_ce': 0.0,
        }
        
        # FIX 4: Skip replay during warmup period
        if task_id < self.replay_warmup_tasks:
            return loss, info
        
        # If buffer is empty, return current loss only
        if self.is_empty():
            return loss, info
        
        # ========== KL Divergence Term (First Sampling) - FIX 1 ==========
        buf_inputs, _, buf_log_probs = self.get_data(
            size=self.minibatch_size,
            device=device,
            transform=transform
        )
        
        if buf_inputs is not None:
            # Current model's predictions on buffer samples
            buf_outputs = model(buf_inputs)
            buf_log_probs_current = F.log_softmax(buf_outputs / self.temperature, dim=1)
            
            # KL divergence: KL(old || new) using stored log-probs as target
            # KL(P||Q) = sum(P * (log(P) - log(Q)))
            # We have log(P) stored, so: exp(log P) * (log P - log Q)
            loss_kl = self.alpha * F.kl_div(
                buf_log_probs_current,  # log Q (current model)
                buf_log_probs,          # log P (stored)
                log_target=True,
                reduction='batchmean'
            )
            loss += loss_kl
            info['replay_kl'] = float(loss_kl.detach().cpu())
        
        # ========== CE Term (Second Sampling) ==========
        buf_inputs, buf_labels, _ = self.get_data(
            size=self.minibatch_size,
            device=device,
            transform=transform
        )
        
        if buf_inputs is not None:
            buf_outputs = model(buf_inputs)
            loss_ce = self.beta * F.cross_entropy(buf_outputs, buf_labels)
            loss += loss_ce
            info['replay_ce'] = float(loss_ce.detach().cpu())
        
        info['total_loss'] = float(loss.detach().cpu())
        
        return loss, info
    
    def extract_features(self, model: nn.Module, data: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Extract 512D feature representations from penultimate layer (Phase 3).
        
        FIXED: Now extracts features from penultimate layer instead of logits.
        
        Args:
            model: Neural network
            data: Input data (B, C, H, W)
        
        Returns:
            Feature representations (B, 512) or None if extraction fails
        """
        if model is None:
            return None
        
        with torch.no_grad():
            try:
                # CRITICAL FIX: Extract features from penultimate layer
                if hasattr(model, 'net'):
                    # Mammoth models have .net attribute
                    # Use returnt='features' to get penultimate layer (512D)
                    features = model.net(data, returnt='features')
                elif hasattr(model, 'feature_extractor'):
                    features = model.feature_extractor(data)
                else:
                    # Fallback: forward pass with returnt parameter if supported
                    try:
                        features = model(data, returnt='features')
                    except TypeError:
                        # Model doesn't support returnt, use regular forward
                        features = model(data)
                
                # If features are spatial (e.g., conv features), global average pool
                if isinstance(features, torch.Tensor) and features.dim() > 2:
                    features = features.mean(dim=[-1, -2])  # Global average pooling
                
                # Flatten to (B, D)
                if isinstance(features, torch.Tensor):
                    if features.dim() > 2:
                        features = features.view(features.size(0), -1)
                    return features
                
            except Exception as e:
                print(f"[CausalDER-v2] Warning: Feature extraction failed: {e}")
                return None
        
        return None
    
    def store(
        self,
        data: torch.Tensor,
        target: torch.Tensor,
        logits: torch.Tensor,
        task_id: int,
        model: nn.Module = None
    ):
        """
        Store samples in buffer with importance scoring (Phase 2) and feature caching (Phase 3).
        
        Args:
            data: Input batch
            target: Labels
            logits: Model predictions
            task_id: Current task
            model: Model for feature extraction (Phase 3)
        """
        # Phase 2: Compute importance scores before storing
        if self.use_importance_sampling:
            importances = self.compute_importance(logits, target)
            self.add_data(data, target, logits, importances, task_id)
        else:
            self.add_data(data, target, logits, None, task_id)
        
        # Phase 3: Cache features for causal graph learning
        if self.enable_causal_graph_learning and model is not None and self.scm is not None:
            features = self.extract_features(model, data)
            if features is not None:
                # Initialize task cache if needed
                if task_id not in self.task_feature_cache:
                    self.task_feature_cache[task_id] = {
                        'features': [],
                        'targets': [],
                        'logits': []
                    }
                
                # Store up to 200 samples per task for graph learning (memory efficient)
                cache = self.task_feature_cache[task_id]
                if len(cache['features']) < 200:
                    # Store one sample at a time (sample randomly from batch)
                    batch_size = data.size(0)
                    idx = torch.randint(0, batch_size, (1,)).item()
                    
                    cache['features'].append(features[idx].cpu())
                    cache['targets'].append(target[idx].cpu())
                    cache['logits'].append(logits[idx].cpu())
    
    def end_task(self, model: nn.Module, task_id: int):
        """
        Called at end of each task.
        
        Phase 1: Print buffer statistics
        Phase 2: Print importance statistics with distribution details
        Phase 2.5: Compute comprehensive CL metrics (for publication)
        FIX 3: Update current task for importance annealing
        
        Args:
            model: The neural network
            task_id: Completed task ID
        """
        # FIX 3: Update task counter for annealing
        self.current_task = task_id + 1
        
        print(f"\n[Causal-DER] End of Task {task_id}")
        print(f"  Buffer: {len(self)}/{self.buffer_size} samples")
        print(f"  Total seen: {self.num_seen} samples")
        
        # Phase 2: Print detailed importance statistics
        if self.use_importance_sampling and len(self.buffer_importances) > 0:
            imp_tensor = torch.tensor(self.buffer_importances)
            stats = self.importance_stats
            
            # Compute percentiles
            p25 = float(torch.quantile(imp_tensor, 0.25))
            p50 = float(torch.quantile(imp_tensor, 0.50))
            p75 = float(torch.quantile(imp_tensor, 0.75))
            p90 = float(torch.quantile(imp_tensor, 0.90))
            std = float(imp_tensor.std())
            
            print(f"  Importance stats:")
            print(f"    Mean: {stats['mean_importance']:.3f} (±{std:.3f})")
            print(f"    Range: [{stats['min_importance']:.3f}, {stats['max_importance']:.3f}]")
            print(f"    Percentiles: p25={p25:.3f}, p50={p50:.3f}, p75={p75:.3f}, p90={p90:.3f}")
            print(f"    High-importance samples (>median): {stats['num_high_importance']}/{len(self.buffer_importances)}")
        
        # NEW: Compute causal metrics if enabled
        if self.compute_ate_metrics and estimate_ate_for_sample and len(self) >= 20:
            print(f"  Computing ATE metrics (may take a few seconds)...")
            
            # Sample subset of buffer for ATE estimation
            num_samples = min(50, len(self))
            indices = torch.randperm(len(self))[:num_samples].tolist()
            
            ate_scores = []
            for idx in indices:
                data = self.buffer_data[idx]
                label = self.buffer_labels[idx]
                
                # Get other samples for comparison
                other_samples = [(self.buffer_data[i], self.buffer_labels[i]) 
                                for i in range(len(self)) if i != idx]
                
                device = next(model.parameters()).device
                ate = estimate_ate_for_sample(
                    model, data, label, other_samples, device, num_reference=20
                )
                ate_scores.append(ate)
                
                # Record in metrics tracker
                if self.metrics_tracker:
                    self.metrics_tracker.record_sample_attribution(task_id, ate)
            
            print(f"    ATE (mean): {sum(ate_scores)/len(ate_scores):.4f}")
            print(f"    ATE (std): {torch.tensor(ate_scores).std().item():.4f}")
        
        # Update metrics tracker
        if self.metrics_tracker:
            self.metrics_tracker.update_current_task(task_id)
        
        # Phase 3: Learn causal graph between tasks
        self.tasks_seen += 1
        if self.enable_causal_graph_learning and self.scm is not None and self.tasks_seen >= 2:
            print(f"\n[Phase 3] Learning causal graph between tasks...")
            print(f"  Tasks seen so far: {self.tasks_seen}")
            print(f"  Cached task data: {list(self.task_feature_cache.keys())}")
            
            # Check if we have enough cached data
            valid_tasks = [tid for tid, cache in self.task_feature_cache.items() 
                          if len(cache['features']) >= 10]
            
            if len(valid_tasks) >= 2:
                # Convert cached features to format expected by SCM
                task_data = {}
                task_labels = {}
                
                for tid in valid_tasks:
                    cache = self.task_feature_cache[tid]
                    if len(cache['features']) > 0:
                        task_data[tid] = torch.stack(cache['features'])
                        task_labels[tid] = torch.stack(cache['targets'])
                
                print(f"  Learning from {len(task_data)} tasks with sufficient data")
                for tid, data in task_data.items():
                    print(f"    Task {tid}: {data.shape[0]} samples, feature dim: {data.shape[1]}")
                
                try:
                    # Learn causal structure using SCM
                    self.causal_graph = self.scm.learn_causal_structure(
                        task_data, 
                        task_labels, 
                        model
                    )
                    
                    if self.causal_graph is not None:
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
                        graph_density = num_edges / (self.num_tasks * (self.num_tasks - 1))
                        mean_strength = self.causal_graph.abs().mean().item()
                        max_strength = self.causal_graph.abs().max().item()
                        
                        print(f"  Graph statistics:")
                        print(f"    Total edges (>0.1): {num_edges}")
                        print(f"    Graph density: {graph_density:.2%}")
                        print(f"    Mean |strength|: {mean_strength:.3f}")
                        print(f"    Max |strength|: {max_strength:.3f}")
                        
                except Exception as e:
                    print(f"  ⚠️ Causal graph learning failed: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"  Not enough data yet (need ≥2 tasks with ≥10 samples each)")
                print(f"  Valid tasks: {valid_tasks}")
        
        # Initialize causal forgetting detector if enabled
        if self.enable_causal_forgetting_detector and self.causal_forgetting_detector is None and CausalForgettingDetector is not None:
            if len(self) >= 50:
                print(f"  Initializing Causal Forgetting Detector...")
                try:
                    self.causal_forgetting_detector = CausalForgettingDetector(
                        model=model,
                        buffer_size=len(self),
                        num_intervention_samples=50
                    )
                    print(f"  ✅ Causal Forgetting Detector ready!")
                except Exception as e:
                    print(f"  ⚠️ Forgetting detector initialization failed: {e}")
    
    def get_statistics(self) -> Dict:
        """
        Get training statistics.
        
        Returns:
            Dictionary with buffer and importance stats
        """
        stats = {
            'buffer': {
                'size': len(self),
                'capacity': self.buffer_size,
                'num_seen': self.num_seen,
            }
        }
        
        # Phase 2: Add importance statistics
        if self.use_importance_sampling:
            stats['importance'] = self.importance_stats
        
        return stats
