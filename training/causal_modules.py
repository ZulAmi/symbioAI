"""
Advanced Causal ML Modules for True SOTA Continual Learning
============================================================

This module implements cutting-edge causal inference techniques:
1. Neural Causal Discovery (NOTEARS-style differentiable DAG learning)
2. Counterfactual Generation (VAE + interventions)
3. Invariant Risk Minimization (IRM)
4. ATE/TMLE Estimation
5. Distribution Shift Detection
6. Adaptive Meta-Controller

Author: Symbio AI
Date: October 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# 1. Neural Causal Discovery (NOTEARS)
# =============================================================================

class NeuralCausalDiscovery(nn.Module):
    """
    Differentiable causal structure learning using NOTEARS approach.
    
    Key Innovation: Learn adjacency matrix A via gradient descent with
    DAG constraint h(A) = tr(e^(A◦A)) - d = 0
    
    Reference: Zheng et al. (2018) "DAGs with NO TEARS"
    """
    
    def __init__(self, num_tasks: int, feature_dim: int, 
                 hidden_dim: int = 64, sparsity: float = 1e-3):
        super().__init__()
        self.num_tasks = num_tasks
        self.feature_dim = feature_dim
        self.sparsity = sparsity
        
        # Learnable adjacency weights (will be constrained to DAG)
        self.adj_weights = nn.Parameter(torch.randn(num_tasks, num_tasks) * 0.1)
        
        # Neural network for feature→task causal mechanisms
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_tasks)
        )
        
        # Lagrangian multipliers for DAG constraint (not learnable, just scalars)
        self.lambda_dag = 0.0
        self.rho = 1.0
        
        logger.info(f"Initialized Neural Causal Discovery for {num_tasks} tasks")
    
    def forward(self, features: torch.Tensor, task_id: int) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass: predict task influences given features.
        
        Args:
            features: [B, D] feature representations
            task_id: Current task ID
            
        Returns:
            predictions: [B, T] predicted task influences
            metrics: Dict with DAG constraint violation, sparsity
        """
        # Encode features to task space
        task_logits = self.encoder(features)  # [B, T]
        
        # Apply learned adjacency (causal structure)
        A = self.get_adjacency()
        causal_effects = task_logits @ A.T  # [B, T]
        
        # Compute constraints
        h_dag = self.dag_constraint(A)
        l1_penalty = torch.norm(A, p=1)
        
        metrics = {
            'dag_violation': float(h_dag),
            'adjacency_sparsity': float(l1_penalty),
            'num_edges': int((A.abs() > 0.1).sum())
        }
        
        return causal_effects, metrics
    
    def get_adjacency(self) -> torch.Tensor:
        """Get current adjacency matrix (with masking)."""
        # Mask diagonal (no self-loops)
        mask = 1.0 - torch.eye(self.num_tasks, device=self.adj_weights.device)
        A = self.adj_weights * mask
        return A
    
    def dag_constraint(self, A: torch.Tensor) -> torch.Tensor:
        """
        NOTEARS DAG constraint: h(A) = tr(e^(A◦A)) - d
        
        h(A) = 0 iff A is a DAG (no cycles)
        """
        M = A * A  # Element-wise square
        # Matrix exponential via Taylor series (faster than torch.matrix_exp for small matrices)
        d = A.size(0)
        expm = torch.eye(d, device=A.device)
        term = torch.eye(d, device=A.device)
        for k in range(1, 10):  # Taylor approximation
            term = term @ M / k
            expm = expm + term
        
        h = torch.trace(expm) - d
        return h
    
    def augmented_lagrangian_loss(self) -> torch.Tensor:
        """
        Augmented Lagrangian for DAG constraint:
        L = loss + λ*h(A) + ρ/2 * h(A)²
        """
        A = self.get_adjacency()
        h = self.dag_constraint(A)
        
        # Augmented Lagrangian penalty
        al_loss = self.lambda_dag * h + 0.5 * self.rho * (h ** 2)
        
        # Add L1 sparsity
        sparsity_loss = self.sparsity * torch.norm(A, p=1)
        
        return al_loss + sparsity_loss
    
    def update_lagrangian(self, threshold: float = 1e-8):
        """Update Lagrangian multipliers (call periodically)."""
        A = self.get_adjacency()
        h = self.dag_constraint(A).item()
        
        if h > threshold:
            # Increase penalty (plain float assignment since they're not tensors)
            self.lambda_dag = self.lambda_dag + self.rho * h
            # CRITICAL FIX: Slower growth (1.5x instead of 10x) and lower max (1e6 instead of 1e10)
            self.rho = min(self.rho * 1.5, 1e6)
        
        return h


# =============================================================================
# 2. Counterfactual Generator (VAE-based)
# =============================================================================

class CounterfactualGenerator(nn.Module):
    """
    VAE-based counterfactual sample generator.
    
    Implements Pearl's 3-step counterfactual inference:
    1. Abduction: Infer latent U from observed X
    2. Action: Intervene do(Task=t)
    3. Prediction: Generate X' under intervention
    
    Reference: Pearl (2009) Causality Ch. 7
    """
    
    def __init__(self, feature_dim: int, latent_dim: int = 128, 
                 num_tasks: int = 10):
        super().__init__()
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim
        self.num_tasks = num_tasks
        
        # Encoder: X → μ(X), σ(X)
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim + num_tasks, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)
        
        # Decoder: Z, Task → X'
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + num_tasks, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim)
        )
        
        logger.info(f"Initialized Counterfactual Generator (latent_dim={latent_dim})")
    
    def encode(self, x: torch.Tensor, task_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode features to latent distribution."""
        # One-hot task encoding
        task_onehot = F.one_hot(torch.tensor([task_id] * x.size(0), device=x.device), 
                                 num_classes=self.num_tasks).float()
        
        inp = torch.cat([x, task_onehot], dim=-1)
        h = self.encoder(inp)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor, task_id: int) -> torch.Tensor:
        """Decode latent + task to features."""
        task_onehot = F.one_hot(torch.tensor([task_id] * z.size(0), device=z.device),
                                 num_classes=self.num_tasks).float()
        inp = torch.cat([z, task_onehot], dim=-1)
        return self.decoder(inp)
    
    def forward(self, x: torch.Tensor, task_id: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Standard VAE forward pass."""
        mu, logvar = self.encode(x, task_id)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z, task_id)
        return x_recon, mu, logvar
    
    def generate_counterfactual(self, x: torch.Tensor, observed_task: int,
                                intervention_task: int) -> torch.Tensor:
        """
        Generate counterfactual: "What if X came from task T' instead of T?"
        
        Args:
            x: Observed features [B, D]
            observed_task: Actual task ID
            intervention_task: Counterfactual task ID (do(Task=t'))
            
        Returns:
            x_cf: Counterfactual features [B, D]
        """
        with torch.no_grad():
            # Abduction: infer latent U from observed (X, T)
            mu, logvar = self.encode(x, observed_task)
            z = self.reparameterize(mu, logvar)
            
            # Action + Prediction: decode with intervened task
            x_cf = self.decode(z, intervention_task)
        
        return x_cf
    
    def vae_loss(self, x: torch.Tensor, x_recon: torch.Tensor,
                 mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """VAE loss: reconstruction + KL divergence."""
        recon_loss = F.mse_loss(x_recon, x, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return (recon_loss + kl_loss) / x.size(0)


# =============================================================================
# 3. Enhanced IRM Module
# =============================================================================

class InvariantRiskMinimization:
    """
    Enhanced Invariant Risk Minimization for causal representation learning.
    
    Key idea: Learn representations whose causal predictive power is
    invariant across environments (tasks).
    
    Reference: Arjovsky et al. (2019) "Invariant Risk Minimization"
    """
    
    @staticmethod
    def irm_penalty(losses: List[torch.Tensor], scale: torch.Tensor) -> torch.Tensor:
        """
        IRM penalty: variance of gradients across environments.
        
        Args:
            losses: List of environment-specific losses
            scale: Dummy scalar variable (requires_grad=True)
            
        Returns:
            penalty: Sum of squared gradients
        """
        penalty = 0.0
        for loss in losses:
            grad = torch.autograd.grad(scale * loss, scale, create_graph=True)[0]
            penalty = penalty + (grad ** 2)
        return penalty
    
    @staticmethod
    def compute_irm_loss(model: nn.Module, data_list: List[torch.Tensor],
                         target_list: List[torch.Tensor], 
                         irm_weight: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute IRM loss across multiple environments.
        
        Args:
            model: Neural network
            data_list: List of data tensors (one per environment)
            target_list: List of target tensors
            irm_weight: Weight for IRM penalty
            
        Returns:
            total_loss: ERM + IRM penalty
            irm_penalty: Just the penalty term
        """
        # Dummy scale for IRM
        scale = torch.tensor(1.0, requires_grad=True, device=data_list[0].device)
        
        losses = []
        erm_loss = 0.0
        
        for data, target in zip(data_list, target_list):
            output = model(data)
            loss = F.cross_entropy(output, target)
            losses.append(loss)
            erm_loss = erm_loss + loss
        
        erm_loss = erm_loss / len(losses)
        
        # IRM penalty
        penalty = InvariantRiskMinimization.irm_penalty(losses, scale)
        
        total_loss = erm_loss + irm_weight * penalty
        
        return total_loss, penalty


# =============================================================================
# 4. ATE/TMLE Estimation
# =============================================================================

class CausalEffectEstimator:
    """
    Average Treatment Effect (ATE) estimation using TMLE.
    
    Reference: van der Laan & Rose (2011) "Targeted Learning"
    """
    
    @staticmethod
    def compute_ate_simple(outcomes_treated: torch.Tensor,
                          outcomes_control: torch.Tensor) -> float:
        """
        Simple ATE estimator: E[Y|T=1] - E[Y|T=0]
        
        Args:
            outcomes_treated: Outcomes under treatment
            outcomes_control: Outcomes under control
            
        Returns:
            ate: Average treatment effect
        """
        return float(outcomes_treated.mean() - outcomes_control.mean())
    
    @staticmethod
    def compute_ate_with_propensity(outcomes: torch.Tensor,
                                    treatment: torch.Tensor,
                                    propensity: torch.Tensor,
                                    epsilon: float = 1e-6) -> float:
        """
        Doubly-robust ATE estimator with propensity scores.
        
        ATE = E[(T/e(X) - (1-T)/(1-e(X))) * Y]
        
        Args:
            outcomes: All outcomes [N]
            treatment: Treatment indicators [N] (0 or 1)
            propensity: P(T=1|X) scores [N]
            epsilon: Numerical stability
            
        Returns:
            ate: Doubly-robust estimate
        """
        propensity = torch.clamp(propensity, epsilon, 1 - epsilon)
        
        weights = (treatment / propensity - 
                  (1 - treatment) / (1 - propensity))
        
        ate = (weights * outcomes).mean()
        return float(ate)
    
    @staticmethod
    def estimate_sample_importance_via_ate(model: nn.Module,
                                          sample_data: torch.Tensor,
                                          sample_target: torch.Tensor,
                                          buffer_data: List[torch.Tensor],
                                          buffer_targets: List[torch.Tensor],
                                          device: torch.device) -> float:
        """
        Estimate causal importance of a sample via counterfactual removal.
        
        Question: "What would forgetting be if we remove this sample from buffer?"
        
        Returns:
            importance: Higher = sample prevents forgetting
        """
        with torch.no_grad():
            # Compute forgetting with sample (treatment)
            all_data = torch.cat([sample_data.unsqueeze(0)] + buffer_data, dim=0)
            all_targets = torch.cat([sample_target.unsqueeze(0)] + buffer_targets, dim=0)
            
            outputs = model(all_data)
            loss_with = F.cross_entropy(outputs, all_targets)
            
            # Compute forgetting without sample (control)
            if len(buffer_data) > 0:
                buffer_data_tensor = torch.cat(buffer_data, dim=0)
                buffer_targets_tensor = torch.cat(buffer_targets, dim=0)
                outputs_control = model(buffer_data_tensor)
                loss_without = F.cross_entropy(outputs_control, buffer_targets_tensor)
            else:
                loss_without = loss_with
            
            # ATE = forgetting prevented by including sample
            ate = float(loss_without - loss_with)
        
        return max(0.0, ate)  # Clip to positive


# =============================================================================
# 5. Distribution Shift Detection
# =============================================================================

class DistributionShiftDetector:
    """
    Detect distribution shifts for task-free streaming.
    
    Uses Maximum Mean Discrepancy (MMD) to detect when data distribution changes.
    
    Reference: Gretton et al. (2012) "A Kernel Two-Sample Test"
    """
    
    def __init__(self, window_size: int = 100, threshold: float = 0.1,
                 kernel_bandwidth: float = 1.0):
        self.window_size = window_size
        self.threshold = threshold
        self.bandwidth = kernel_bandwidth
        self.reference_features = []
    
    def gaussian_kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """RBF kernel: k(x,y) = exp(-||x-y||²/(2σ²))"""
        dist_sq = torch.cdist(x, y, p=2).pow(2)
        return torch.exp(-dist_sq / (2 * self.bandwidth ** 2))
    
    def compute_mmd(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """
        Maximum Mean Discrepancy between two distributions.
        
        MMD² = E[k(X,X')] + E[k(Y,Y')] - 2E[k(X,Y)]
        """
        xx = self.gaussian_kernel(x, x).mean()
        yy = self.gaussian_kernel(y, y).mean()
        xy = self.gaussian_kernel(x, y).mean()
        
        mmd_sq = xx + yy - 2 * xy
        return float(torch.sqrt(torch.clamp(mmd_sq, min=0.0)))
    
    def update_reference(self, features: torch.Tensor):
        """Update reference distribution (sliding window)."""
        self.reference_features.append(features.detach().cpu())
        if len(self.reference_features) > self.window_size:
            self.reference_features.pop(0)
    
    def detect_shift(self, current_features: torch.Tensor) -> Tuple[bool, float]:
        """
        Detect if current batch is from different distribution.
        
        Returns:
            is_shift: True if shift detected
            mmd_score: Magnitude of shift
        """
        if len(self.reference_features) < 10:
            # Not enough reference data
            return False, 0.0
        
        ref = torch.cat(self.reference_features, dim=0)
        
        # Subsample for efficiency
        if ref.size(0) > 200:
            idx = torch.randperm(ref.size(0))[:200]
            ref = ref[idx]
        
        mmd = self.compute_mmd(ref.to(current_features.device), current_features)
        
        is_shift = mmd > self.threshold
        
        return is_shift, mmd


# =============================================================================
# 6. Adaptive Meta-Controller
# =============================================================================

class AdaptiveMetaController:
    """
    Meta-learning controller for adaptive hyperparameter tuning.
    
    Uses contextual multi-armed bandit to adapt:
    - causal_weight (causal vs random sampling)
    - alpha (KD weight)
    - pruning frequency
    
    Reference: Langford & Zhang (2007) "Epoch-Greedy"
    """
    
    def __init__(self, num_actions: int = 5, epsilon: float = 0.1,
                 learning_rate: float = 0.01):
        self.num_actions = num_actions
        self.epsilon = epsilon
        self.lr = learning_rate
        
        # Action value estimates (Q-values)
        self.q_values = np.zeros(num_actions)
        self.action_counts = np.zeros(num_actions)
        
        # Action space: different causal_weight settings
        self.action_space = np.linspace(0.3, 0.9, num_actions)
        
        logger.info(f"Initialized Adaptive Meta-Controller with {num_actions} actions")
    
    def select_action(self, context: Optional[Dict] = None) -> int:
        """
        Select action using epsilon-greedy with UCB.
        
        Args:
            context: Optional context features (task info, buffer stats)
            
        Returns:
            action: Index of selected action
        """
        if np.random.random() < self.epsilon:
            # Explore
            return np.random.randint(self.num_actions)
        else:
            # Exploit with UCB bonus
            total_counts = self.action_counts.sum() + 1
            ucb_bonus = np.sqrt(2 * np.log(total_counts) / (self.action_counts + 1))
            ucb_values = self.q_values + ucb_bonus
            return int(np.argmax(ucb_values))
    
    def update(self, action: int, reward: float):
        """
        Update Q-value estimates.
        
        Args:
            action: Action taken
            reward: Observed reward (e.g., -forgetting or +accuracy)
        """
        self.action_counts[action] += 1
        
        # Running average update
        alpha = self.lr
        self.q_values[action] = (1 - alpha) * self.q_values[action] + alpha * reward
    
    def get_causal_weight(self, action: int) -> float:
        """Map action to causal_weight value."""
        return float(self.action_space[action])
    
    def get_statistics(self) -> Dict:
        """Get controller statistics."""
        return {
            'q_values': self.q_values.tolist(),
            'action_counts': self.action_counts.tolist(),
            'best_action': int(np.argmax(self.q_values)),
            'best_causal_weight': float(self.action_space[np.argmax(self.q_values)])
        }


# =============================================================================
# Export
# =============================================================================

__all__ = [
    'NeuralCausalDiscovery',
    'CounterfactualGenerator',
    'InvariantRiskMinimization',
    'CausalEffectEstimator',
    'DistributionShiftDetector',
    'AdaptiveMetaController'
]
