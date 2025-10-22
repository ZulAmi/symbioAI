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
        **kwargs  # Accept extra args for future compatibility
    ):
        """
        Initialize DER++ engine with Phase 2 enhancements.
        
        Args:
            alpha: MSE distillation weight
            beta: CE replay weight  
            buffer_size: Buffer capacity
            minibatch_size: Replay batch size (CRITICAL: must match batch_size!)
            use_importance_sampling: Enable importance-weighted sampling (Phase 2)
            importance_weight: Weight for importance vs random (default: 0.7)
        """
        self.alpha = alpha
        self.beta = beta
        self.buffer_size = buffer_size
        self.minibatch_size = minibatch_size
        self.use_importance_sampling = use_importance_sampling
        self.importance_weight = importance_weight
        
        # Simple list-based buffer (matches Mammoth's Buffer class behavior)
        self.buffer_data = []
        self.buffer_labels = []
        self.buffer_logits = []
        self.buffer_importances = []  # Phase 2: Store importance scores
        self.num_seen = 0  # For reservoir sampling
        
        # Statistics tracking
        self.importance_stats = {
            'mean_importance': 0.0,
            'min_importance': 0.0,
            'max_importance': 0.0,
            'num_high_importance': 0,
        }
        
        phase = "Phase 2 (Importance Sampling)" if use_importance_sampling else "Phase 1 (Baseline)"
        print(f"[CausalDER-v2] {phase}")
        print(f"[CausalDER-v2] α={alpha}, β={beta}, buffer={buffer_size}, minibatch={minibatch_size}")
        if use_importance_sampling:
            print(f"[CausalDER-v2] Importance sampling: {importance_weight:.1%} by importance, {1-importance_weight:.1%} random")
    
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
        
        Importance = loss + (1 - confidence) + small_noise
        
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
        
        # Compute confidence (max probability)
        probs = F.softmax(logits, dim=1)
        confidence = probs.max(dim=1)[0]
        
        # Importance = high loss + low confidence
        # Normalize to [0, 1] range
        importance = (loss_per_sample / (loss_per_sample.max() + 1e-6)) + (1 - confidence)
        
        # Add small random noise to break ties
        importance += 0.01 * torch.rand_like(importance)
        
        return importance
    
    def add_data(
        self,
        examples: torch.Tensor,
        labels: torch.Tensor,
        logits: torch.Tensor,
        importances: Optional[torch.Tensor] = None
    ):
        """
        Add data to buffer using reservoir sampling with importance scores (Phase 2).
        
        Args:
            examples: Input data (B, C, H, W)
            labels: Ground truth labels (B,)
            logits: Model predictions/logits (B, num_classes)
            importances: Importance scores (B,) - optional, computed if not provided
        """
        # Move to CPU for storage
        examples = examples.cpu()
        labels = labels.cpu()
        logits = logits.cpu()
        
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
                self.buffer_logits.append(logits[i])
                self.buffer_importances.append(importance_score)
            else:
                # Buffer full, reservoir sampling
                idx = torch.randint(0, self.num_seen + i + 1, (1,)).item()
                if idx < self.buffer_size:
                    self.buffer_data[idx] = examples[i]
                    self.buffer_labels[idx] = labels[i]
                    self.buffer_logits[idx] = logits[i]
                    self.buffer_importances[idx] = importance_score
            
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
        Phase 2: 70% importance-weighted + 30% random
        
        Args:
            size: Number of samples to retrieve
            device: Device to move data to
            transform: Optional data augmentation
        
        Returns:
            (data, labels, logits) tuple or (None, None, None) if empty
        """
        if self.is_empty():
            return None, None, None
        
        size = min(size, len(self.buffer_data))
        
        # Phase 2: Importance-weighted sampling
        if self.use_importance_sampling and len(self.buffer_importances) > 0:
            # Split sampling: 70% by importance, 30% random
            n_importance = int(size * self.importance_weight)
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
        logits = torch.stack([self.buffer_logits[i] for i in indices])
        
        # Apply transform if provided (before moving to device, like official buffer)
        if transform is not None:
            data = apply_transform(data, transform)
        
        # Move to device
        data = data.to(device)
        labels = labels.to(device)
        logits = logits.to(device)
        
        return data, labels, logits
    
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
        Compute DER++ loss - EXACT copy of official derpp.py observe() logic.
        
        Loss = CE(current) + α·MSE(replay_logits) + β·CE(replay_labels)
        
        Args:
            model: The neural network
            data: Current batch inputs
            target: Current batch labels
            output: Model predictions on current batch (already computed)
            task_id: Current task ID (unused in Phase 1)
            transform: Data augmentation for replay
        
        Returns:
            (loss, info_dict) tuple
        """
        device = data.device
        
        # Current task CE loss (using F.cross_entropy like official)
        loss = F.cross_entropy(output, target)
        
        info = {
            'current_loss': float(loss.detach().cpu()),
            'replay_mse': 0.0,
            'replay_ce': 0.0,
        }
        
        # If buffer is empty, return current loss only
        if self.is_empty():
            return loss, info
        
        # ========== MSE Term (First Sampling) ==========
        buf_inputs, _, buf_logits = self.get_data(
            size=self.minibatch_size,
            device=device,
            transform=transform
        )
        
        if buf_inputs is not None:
            buf_outputs = model(buf_inputs)
            loss_mse = self.alpha * F.mse_loss(buf_outputs, buf_logits)
            loss += loss_mse
            info['replay_mse'] = float(loss_mse.detach().cpu())
        
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
    
    def store(
        self,
        data: torch.Tensor,
        target: torch.Tensor,
        logits: torch.Tensor,
        task_id: int,
        model: nn.Module = None
    ):
        """
        Store samples in buffer with importance scoring (Phase 2).
        
        Args:
            data: Input batch
            target: Labels
            logits: Model predictions
            task_id: Current task (unused in Phase 1-2)
            model: Model (unused in Phase 1-2)
        """
        # Phase 2: Compute importance scores before storing
        if self.use_importance_sampling:
            importances = self.compute_importance(logits, target)
            self.add_data(data, target, logits, importances)
        else:
            self.add_data(data, target, logits, None)
    
    def end_task(self, model: nn.Module, task_id: int):
        """
        Called at end of each task.
        
        Phase 1: Print buffer statistics
        Phase 2: Print importance statistics
        
        Args:
            model: The neural network
            task_id: Completed task ID
        """
        print(f"\n[Causal-DER] End of Task {task_id}")
        print(f"  Buffer: {len(self)}/{self.buffer_size} samples")
        print(f"  Total seen: {self.num_seen} samples")
        
        # Phase 2: Print importance statistics
        if self.use_importance_sampling and len(self.buffer_importances) > 0:
            stats = self.importance_stats
            print(f"  Importance stats:")
            print(f"    Mean: {stats['mean_importance']:.3f}")
            print(f"    Range: [{stats['min_importance']:.3f}, {stats['max_importance']:.3f}]")
            print(f"    High-importance samples: {stats['num_high_importance']}/{len(self.buffer_importances)}")
    
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
