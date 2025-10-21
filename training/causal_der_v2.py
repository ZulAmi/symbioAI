"""
Causal Dark Experience Replay Engine v2 - Clean Implementation
================================================================

Built incrementally on proven DER++ baseline (56% Task-IL on seq-cifar100).

Phase 1: Clean DER++ baseline (THIS FILE)
- Exact copy of Mammoth's derpp.py implementation
- Simple, readable, no sanitization
- Target: Match 56% Task-IL performance

Future Phases (add incrementally after baseline verified):
- Phase 2: Add causal importance scoring
- Phase 3: Add causal sampling
- Phase 4: Add task graph learning

Author: Symbio AI
Date: October 20, 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass


@dataclass
class DERSample:
    """Sample stored in replay buffer."""
    data: torch.Tensor      # Input data
    target: torch.Tensor    # Ground truth label
    logits: torch.Tensor    # Teacher logits (dark knowledge)
    task_id: int            # Which task this sample came from
    
    # Future: Add causal importance field
    # causal_importance: float = 1.0


class ReplayBuffer:
    """
    Simple replay buffer with reservoir sampling.
    
    Exact implementation matching Mammoth's Buffer class.
    """
    
    def __init__(self, capacity: int):
        """
        Args:
            capacity: Maximum number of samples to store
        """
        self.capacity = capacity
        self.buffer: List[DERSample] = []
        self.num_seen = 0  # Total samples seen (for reservoir sampling)
    
    def is_empty(self) -> bool:
        """Check if buffer is empty."""
        return len(self.buffer) == 0
    
    def __len__(self) -> int:
        """Return current buffer size."""
        return len(self.buffer)
    
    def add(self, sample: DERSample):
        """
        Add sample to buffer using reservoir sampling.
        
        Reservoir sampling ensures uniform distribution over all seen samples.
        """
        if len(self.buffer) < self.capacity:
            # Buffer not full: simply append
            self.buffer.append(sample)
        else:
            # Buffer full: reservoir sampling
            # Replace random sample with probability capacity/num_seen
            idx = torch.randint(0, self.num_seen + 1, (1,)).item()
            if idx < self.capacity:
                self.buffer[idx] = sample
        
        self.num_seen += 1
    
    def sample(self, batch_size: int, device: torch.device) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Sample batch from buffer (uniform random sampling).
        
        Args:
            batch_size: Number of samples to retrieve
            device: Device to move samples to
        
        Returns:
            (data, targets, logits) tuple, or None if buffer is empty
        """
        if self.is_empty():
            return None
        
        # Sample indices uniformly at random
        batch_size = min(batch_size, len(self.buffer))
        indices = torch.randperm(len(self.buffer))[:batch_size]
        
        # Gather samples
        samples = [self.buffer[i] for i in indices]
        
        # Stack into batches and move to device
        data = torch.stack([s.data for s in samples]).to(device)
        targets = torch.stack([s.target for s in samples]).to(device)
        logits = torch.stack([s.logits for s in samples]).to(device)
        
        return data, targets, logits


class CausalDEREngine:
    """
    Causal-DER Engine v2 - Clean Baseline
    
    Phase 1: Exact DER++ implementation
    - Simple 3-line loss: CE(current) + α·MSE(replay) + β·CE(replay)
    - Uniform buffer sampling
    - Reservoir sampling for storage
    
    Target: Match Mammoth DER++ performance (56% Task-IL on seq-cifar100)
    """
    
    def __init__(
        self,
        alpha: float = 0.1,
        beta: float = 0.5,
        buffer_size: int = 500,
        **kwargs  # Accept extra args for future compatibility, ignore for now
    ):
        """
        Initialize DER++ engine.
        
        Args:
            alpha: MSE distillation weight (default: 0.1 from DER++ paper)
            beta: CE replay weight (default: 0.5 from DER++ paper)
            buffer_size: Buffer capacity (default: 500)
        """
        # Core DER++ hyperparameters
        self.alpha = alpha
        self.beta = beta
        
        # Replay buffer
        self.buffer = ReplayBuffer(capacity=buffer_size)
        
        # Statistics
        self.stats = {
            'total_samples_stored': 0,
            'total_batches_processed': 0,
        }
        
        print(f"[CausalDER-v2] Initialized with α={alpha}, β={beta}, buffer={buffer_size}")
        print(f"[CausalDER-v2] Phase 1: Clean DER++ baseline (no causal features)")
    
    def compute_loss(
        self,
        model: nn.Module,
        data: torch.Tensor,
        target: torch.Tensor,
        output: torch.Tensor,
        task_id: int
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute DER++ loss (EXACT implementation from Mammoth).
        
        Loss = CE(current_data) + α·MSE(replay_logits) + β·CE(replay_labels)
        
        Args:
            model: The neural network
            data: Current batch inputs
            target: Current batch labels
            output: Model predictions on current batch
            task_id: Current task ID (for future use)
        
        Returns:
            (loss, info_dict) tuple
        """
        # Current task loss
        current_loss = F.cross_entropy(output, target)
        
        info = {
            'current_loss': float(current_loss.detach().cpu()),
            'replay_mse': 0.0,
            'replay_ce': 0.0,
            'total_loss': float(current_loss.detach().cpu())
        }
        
        # If buffer is empty, return current loss only
        if self.buffer.is_empty():
            self.stats['total_batches_processed'] += 1
            return current_loss, info
        
        # Get device from data
        device = data.device
        
        # Sample for MSE distillation (first sampling)
        replay_batch_size = min(32, len(self.buffer))  # Default minibatch size
        replay_data = self.buffer.sample(replay_batch_size, device)
        
        if replay_data is None:
            self.stats['total_batches_processed'] += 1
            return current_loss, info
        
        buf_inputs_mse, _, buf_logits_mse = replay_data
        
        # Forward pass on replay data
        buf_outputs_mse = model(buf_inputs_mse)
        
        # MSE loss between current predictions and stored logits
        replay_mse = self.alpha * F.mse_loss(buf_outputs_mse, buf_logits_mse)
        
        # Sample for CE loss (second sampling - DER++ samples twice)
        replay_data_ce = self.buffer.sample(replay_batch_size, device)
        
        if replay_data_ce is None:
            total_loss = current_loss + replay_mse
            info['replay_mse'] = float(replay_mse.detach().cpu())
            info['total_loss'] = float(total_loss.detach().cpu())
            self.stats['total_batches_processed'] += 1
            return total_loss, info
        
        buf_inputs_ce, buf_targets_ce, _ = replay_data_ce
        
        # Forward pass on second replay batch
        buf_outputs_ce = model(buf_inputs_ce)
        
        # CE loss on replay labels
        replay_ce = self.beta * F.cross_entropy(buf_outputs_ce, buf_targets_ce)
        
        # Total DER++ loss
        total_loss = current_loss + replay_mse + replay_ce
        
        # Update info
        info['replay_mse'] = float(replay_mse.detach().cpu())
        info['replay_ce'] = float(replay_ce.detach().cpu())
        info['total_loss'] = float(total_loss.detach().cpu())
        
        self.stats['total_batches_processed'] += 1
        
        return total_loss, info
    
    def store(
        self,
        data: torch.Tensor,
        target: torch.Tensor,
        logits: torch.Tensor,
        task_id: int,
        model: nn.Module = None
    ):
        """
        Store samples in replay buffer.
        
        Uses reservoir sampling to maintain uniform distribution.
        
        Args:
            data: Input batch (B, C, H, W)
            target: Labels (B,)
            logits: Model predictions (B, num_classes)
            task_id: Current task ID
            model: Model (unused in Phase 1, for future causal importance)
        """
        batch_size = data.size(0)
        
        for idx in range(batch_size):
            # Create sample
            sample = DERSample(
                data=data[idx].detach().cpu(),
                target=target[idx].detach().cpu(),
                logits=logits[idx].detach().cpu(),
                task_id=task_id
            )
            
            # Add to buffer (reservoir sampling happens inside)
            self.buffer.add(sample)
            self.stats['total_samples_stored'] += 1
    
    def end_task(self, model: nn.Module, task_id: int):
        """
        Called at end of each task.
        
        Phase 1: Just print statistics
        Future: Add causal graph learning
        
        Args:
            model: The neural network
            task_id: Completed task ID
        """
        print(f"\n{'='*60}")
        print(f"END OF TASK {task_id}")
        print(f"{'='*60}")
        print(f"Buffer size: {len(self.buffer)}/{self.buffer.capacity}")
        print(f"Total samples stored: {self.stats['total_samples_stored']}")
        print(f"Total batches processed: {self.stats['total_batches_processed']}")
        print(f"{'='*60}\n")
        
        # Future: Add causal graph learning here
        # if self.enable_causal_graph_learning:
        #     self.learn_causal_graph(model)
    
    def get_statistics(self) -> Dict:
        """
        Get training statistics.
        
        Returns:
            Dictionary with buffer and training stats
        """
        # Count samples per task
        samples_per_task = {}
        for sample in self.buffer.buffer:
            tid = sample.task_id
            samples_per_task[tid] = samples_per_task.get(tid, 0) + 1
        
        return {
            'buffer': {
                'size': len(self.buffer),
                'capacity': self.buffer.capacity,
                'samples_per_task': samples_per_task,
                'avg_causal_importance': 1.0,  # Uniform in Phase 1
            },
            'training': {
                'total_samples_stored': self.stats['total_samples_stored'],
                'total_batches_processed': self.stats['total_batches_processed'],
            }
        }


# ============================================================================
# FUTURE PHASES (Commented out - add incrementally after Phase 1 verified)
# ============================================================================

"""
PHASE 2: Causal Importance Scoring
-----------------------------------

def compute_causal_importance(
    data: torch.Tensor,
    target: torch.Tensor,
    logits: torch.Tensor,
    task_id: int,
    model: nn.Module
) -> float:
    '''
    Compute importance score based on:
    1. Prediction confidence (low confidence = more important)
    2. Loss value (high loss = more important)
    3. Task novelty (new patterns = more important)
    
    Returns:
        importance: float in [0, 1]
    '''
    with torch.no_grad():
        # Prediction confidence (entropy)
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum()
        
        # Loss value
        loss = F.cross_entropy(logits.unsqueeze(0), target.unsqueeze(0))
        
        # Normalize and combine
        importance = 0.5 * entropy / math.log(probs.size(-1)) + 0.5 * loss.item()
        
    return float(importance)


PHASE 3: Causal Sampling
-------------------------

def sample_with_importance(
    buffer: List[DERSample],
    batch_size: int,
    temperature: float = 1.0
) -> List[int]:
    '''
    Sample buffer based on importance weights.
    
    Args:
        buffer: List of samples with causal_importance scores
        batch_size: Number of samples to retrieve
        temperature: Temperature for softmax (higher = more uniform)
    
    Returns:
        indices: List of sampled indices
    '''
    importances = torch.tensor([s.causal_importance for s in buffer])
    
    # Temperature-scaled softmax
    probs = F.softmax(importances / temperature, dim=0)
    
    # Sample with replacement based on probabilities
    indices = torch.multinomial(probs, batch_size, replacement=True)
    
    return indices.tolist()


PHASE 4: Task Graph Learning
-----------------------------

class TaskGraph:
    '''
    Learn causal dependencies between tasks.
    
    Tracks: Does replaying Task i help with Task j?
    '''
    
    def __init__(self, num_tasks: int):
        self.num_tasks = num_tasks
        self.adjacency = torch.zeros(num_tasks, num_tasks)
    
    def learn_dependencies(self, model, buffer, current_task):
        '''
        Measure impact of each task's samples on current task performance.
        
        High impact = causal edge from source task to current task
        '''
        # For each previous task
        for task_id in range(current_task):
            # Sample from this task
            task_samples = [s for s in buffer if s.task_id == task_id]
            if not task_samples:
                continue
            
            # Measure gradient alignment (proxy for helpfulness)
            # Samples that align with current gradient are helpful
            alignment = measure_gradient_alignment(model, task_samples)
            
            # Store in adjacency matrix
            self.adjacency[task_id, current_task] = alignment
    
    def get_task_weights(self, current_task: int) -> torch.Tensor:
        '''
        Get importance weights for each task relative to current task.
        
        Returns:
            weights: (num_tasks,) tensor
        '''
        return self.adjacency[:, current_task]

"""
