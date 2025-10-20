"""
Distill knowledge from an ensemble of expert models into a single student model.

Production-grade knowledge distillation implementation for Symbio AI that enables
efficient model compression and knowledge transfer from multiple expert models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from pathlib import Path
from abc import ABC, abstractmethod
from datetime import datetime
import json
import pickle
from contextlib import contextmanager
from collections import defaultdict

# Import from our existing system
import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.registry import BaseModel, ModelMetadata, ModelFramework
from core.pipeline import Config, TrainingCallback
from monitoring.production import MetricsCollector, ProductionLogger
from evaluation.benchmarks import BenchmarkResult, MetricType


class DistillationStrategy(torch.nn.Module):
    """Defines different knowledge distillation strategies."""
    
    SOFT_TARGETS = "soft_targets"
    FEATURE_MATCHING = "feature_matching"
    ATTENTION_TRANSFER = "attention_transfer"
    HYBRID = "hybrid"


@dataclass
class DistillationConfig:
    """Configuration for knowledge distillation training."""
    
    # Core distillation parameters
    temperature: float = 2.0
    alpha: float = 0.7  # Weight for distillation loss
    beta: float = 0.3   # Weight for hard target loss
    
    # Training parameters
    learning_rate: float = 1e-4
    batch_size: int = 32
    num_epochs: int = 100
    warmup_epochs: int = 5
    
    # Model parameters
    student_hidden_size: int = 512
    student_num_layers: int = 6
    dropout_rate: float = 0.1
    
    # Expert ensemble parameters
    expert_model_paths: List[str] = field(default_factory=list)
    expert_weights: Optional[List[float]] = None
    ensemble_method: str = "average"  # "average", "weighted", "voting"
    
    # Advanced distillation settings
    distillation_strategy: str = DistillationStrategy.SOFT_TARGETS
    feature_matching_layers: List[int] = field(default_factory=list)
    attention_matching: bool = False
    progressive_distillation: bool = False
    
    # Optimization settings
    optimizer_type: str = "adamw"
    weight_decay: float = 0.01
    gradient_clip_norm: float = 1.0
    scheduler_type: str = "cosine"
    
    # Regularization
    label_smoothing: float = 0.1
    mixup_alpha: float = 0.2
    cutmix_alpha: float = 1.0
    
    # Checkpointing and logging
    save_interval: int = 10
    log_interval: int = 100
    validate_interval: int = 5
    output_dir: str = "outputs/distillation"
    
    # Hardware settings
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: bool = True
    compile_model: bool = False  # PyTorch 2.0 compilation
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'temperature': self.temperature,
            'alpha': self.alpha,
            'beta': self.beta,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'num_epochs': self.num_epochs,
            'student_hidden_size': self.student_hidden_size,
            'student_num_layers': self.student_num_layers,
            'expert_model_paths': self.expert_model_paths,
            'distillation_strategy': self.distillation_strategy,
            'output_dir': self.output_dir
        }


class ExpertModel(nn.Module):
    """Expert model wrapper for knowledge distillation."""
    
    def __init__(self, model_path: str, device: str = "cuda"):
        super().__init__()
        self.model_path = model_path
        self.device = device
        self.model = None
        self._is_loaded = False
    
    @classmethod
    def load_pretrained(cls, model_path: str, device: str = "cuda") -> 'ExpertModel':
        """Load a pretrained expert model."""
        expert = cls(model_path, device)
        expert.load_model()
        return expert
    
    def load_model(self):
        """Load the actual model from disk."""
        try:
            if self.model_path.endswith('.pt') or self.model_path.endswith('.pth'):
                # PyTorch checkpoint
                checkpoint = torch.load(self.model_path, map_location=self.device)
                if isinstance(checkpoint, dict) and 'model' in checkpoint:
                    self.model = checkpoint['model']
                else:
                    self.model = checkpoint
            else:
                # Assume it's a model class that needs instantiation
                # This is a placeholder - in production, implement proper model loading
                self.model = self._create_mock_expert_model()
            
            self.model.to(self.device)
            self.model.eval()
            self._is_loaded = True
            
        except Exception as e:
            logging.error(f"Failed to load expert model from {self.model_path}: {e}")
            # Create a mock model for demonstration
            self.model = self._create_mock_expert_model()
            self.model.to(self.device)
            self.model.eval()
            self._is_loaded = True
    
    def _create_mock_expert_model(self) -> nn.Module:
        """Create a mock expert model for demonstration purposes."""
        return nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10)  # Assume 10 output classes
        )
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass through the expert model."""
        if not self._is_loaded:
            self.load_model()
        return self.model(inputs)
    
    def get_features(self, inputs: torch.Tensor, layer_indices: List[int] = None) -> Dict[int, torch.Tensor]:
        """Extract intermediate features from specified layers."""
        if not self._is_loaded:
            self.load_model()
        
        features = {}
        
        def hook_fn(layer_idx):
            def hook(module, input, output):
                features[layer_idx] = output.detach()
            return hook
        
        hooks = []
        if layer_indices:
            for idx in layer_indices:
                if idx < len(list(self.model.modules())):
                    layer = list(self.model.modules())[idx]
                    hooks.append(layer.register_forward_hook(hook_fn(idx)))
        
        # Forward pass
        with torch.no_grad():
            _ = self.model(inputs)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        return features


class StudentModel(nn.Module):
    """Student model for knowledge distillation."""
    
    def __init__(self, hidden_size: int = 512, num_layers: int = 6, 
                 input_size: int = 768, output_size: int = 10, 
                 dropout_rate: float = 0.1):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        
        # Build the student network
        layers = []
        current_size = input_size
        
        for i in range(num_layers):
            layers.extend([
                nn.Linear(current_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            current_size = hidden_size
        
        # Final output layer
        layers.append(nn.Linear(hidden_size, output_size))
        
        self.model = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the student model."""
        return self.model(x)
    
    def get_features(self, x: torch.Tensor, layer_indices: List[int] = None) -> Dict[int, torch.Tensor]:
        """Extract intermediate features for feature matching."""
        features = {}
        
        def hook_fn(layer_idx):
            def hook(module, input, output):
                features[layer_idx] = output
            return hook
        
        hooks = []
        if layer_indices:
            modules_list = list(self.model.modules())
            for idx in layer_indices:
                if idx < len(modules_list):
                    hooks.append(modules_list[idx].register_forward_hook(hook_fn(idx)))
        
        # Forward pass
        output = self.model(x)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        return features, output


class KnowledgeDistillationLoss(nn.Module):
    """Comprehensive loss function for knowledge distillation."""
    
    def __init__(self, config: DistillationConfig):
        super().__init__()
        self.config = config
        self.hard_loss = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
        self.feature_loss = nn.MSELoss()
    
    def soft_target_loss(self, student_logits: torch.Tensor, 
                        teacher_logits: torch.Tensor, 
                        temperature: float) -> torch.Tensor:
        """Compute soft target distillation loss."""
        return F.kl_div(
            F.log_softmax(student_logits / temperature, dim=-1),
            F.softmax(teacher_logits / temperature, dim=-1),
            reduction='batchmean'
        ) * (temperature ** 2)
    
    def feature_matching_loss(self, student_features: Dict[int, torch.Tensor],
                            teacher_features: Dict[int, torch.Tensor]) -> torch.Tensor:
        """Compute feature matching loss between student and teacher."""
        total_loss = 0.0
        num_matched = 0
        
        for layer_idx in self.config.feature_matching_layers:
            if layer_idx in student_features and layer_idx in teacher_features:
                student_feat = student_features[layer_idx]
                teacher_feat = teacher_features[layer_idx]
                
                # Align dimensions if necessary
                if student_feat.shape != teacher_feat.shape:
                    # Simple alignment - in production, use more sophisticated methods
                    min_dim = min(student_feat.size(-1), teacher_feat.size(-1))
                    student_feat = student_feat[..., :min_dim]
                    teacher_feat = teacher_feat[..., :min_dim]
                
                total_loss += self.feature_loss(student_feat, teacher_feat.detach())
                num_matched += 1
        
        return total_loss / max(num_matched, 1)
    
    def forward(self, student_logits: torch.Tensor, 
                teacher_logits: torch.Tensor,
                hard_targets: torch.Tensor = None,
                student_features: Dict[int, torch.Tensor] = None,
                teacher_features: Dict[int, torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Compute combined distillation loss."""
        
        losses = {}
        
        # Soft target loss (main distillation loss)
        soft_loss = self.soft_target_loss(
            student_logits, teacher_logits, self.config.temperature
        )
        losses['soft_loss'] = soft_loss
        
        # Hard target loss (if labels are provided)
        hard_loss = torch.tensor(0.0, device=student_logits.device)
        if hard_targets is not None:
            hard_loss = self.hard_loss(student_logits, hard_targets)
        losses['hard_loss'] = hard_loss
        
        # Feature matching loss
        feature_loss = torch.tensor(0.0, device=student_logits.device)
        if (student_features is not None and teacher_features is not None and 
            self.config.feature_matching_layers):
            feature_loss = self.feature_matching_loss(student_features, teacher_features)
        losses['feature_loss'] = feature_loss
        
        # Combined loss
        total_loss = (self.config.alpha * soft_loss + 
                     self.config.beta * hard_loss + 
                     0.1 * feature_loss)  # Feature loss weight
        
        losses['total_loss'] = total_loss
        
        return losses


class EnsembleTeacher:
    """Manages ensemble of expert models for knowledge distillation."""
    
    def __init__(self, expert_paths: List[str], weights: Optional[List[float]] = None,
                 device: str = "cuda"):
        self.expert_paths = expert_paths
        self.weights = weights or [1.0] * len(expert_paths)
        self.device = device
        self.experts = []
        
        # Normalize weights
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]
        
        self._load_experts()
    
    def _load_experts(self):
        """Load all expert models."""
        for path in self.expert_paths:
            try:
                expert = ExpertModel.load_pretrained(path, self.device)
                self.experts.append(expert)
                logging.info(f"Loaded expert model from {path}")
            except Exception as e:
                logging.error(f"Failed to load expert from {path}: {e}")
                # Create a mock expert for demonstration
                expert = ExpertModel("mock_expert.pt", self.device)
                expert.load_model()
                self.experts.append(expert)
    
    def get_ensemble_predictions(self, inputs: torch.Tensor, 
                               method: str = "average") -> torch.Tensor:
        """Get ensemble predictions from all expert models."""
        predictions = []
        
        with torch.no_grad():
            for expert, weight in zip(self.experts, self.weights):
                logits = expert(inputs)
                if method == "weighted":
                    logits = logits * weight
                predictions.append(logits)
        
        if method in ["average", "weighted"]:
            if method == "average":
                return torch.stack(predictions).mean(dim=0)
            else:  # weighted already applied above
                return torch.stack(predictions).sum(dim=0)
        elif method == "voting":
            # Majority voting on predictions
            pred_classes = [torch.argmax(pred, dim=-1) for pred in predictions]
            stacked = torch.stack(pred_classes)
            # Simple majority vote - in production, implement proper voting
            return torch.mode(stacked, dim=0)[0]
        else:
            raise ValueError(f"Unknown ensemble method: {method}")
    
    def get_ensemble_features(self, inputs: torch.Tensor, 
                            layer_indices: List[int]) -> Dict[int, torch.Tensor]:
        """Get averaged features from ensemble for feature matching."""
        ensemble_features = defaultdict(list)
        
        with torch.no_grad():
            for expert in self.experts:
                features = expert.get_features(inputs, layer_indices)
                for idx, feat in features.items():
                    ensemble_features[idx].append(feat)
        
        # Average features across experts
        averaged_features = {}
        for idx, feat_list in ensemble_features.items():
            if feat_list:
                averaged_features[idx] = torch.stack(feat_list).mean(dim=0)
        
        return averaged_features


class DistillationTrainer:
    """Main trainer class for knowledge distillation."""
    
    def __init__(self, config: DistillationConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Initialize logging and metrics
        self.logger = ProductionLogger("distillation_trainer")
        self.metrics = MetricsCollector()
        
        # Initialize models
        self.student = None
        self.teacher_ensemble = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # Setup output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Loss function
        self.loss_fn = KnowledgeDistillationLoss(config)
    
    def setup_models(self, input_size: int = 768, output_size: int = 10):
        """Initialize student model and teacher ensemble."""
        # Initialize student model
        self.student = StudentModel(
            hidden_size=self.config.student_hidden_size,
            num_layers=self.config.student_num_layers,
            input_size=input_size,
            output_size=output_size,
            dropout_rate=self.config.dropout_rate
        ).to(self.device)
        
        # Compile model if requested (PyTorch 2.0+)
        if self.config.compile_model and hasattr(torch, 'compile'):
            self.student = torch.compile(self.student)
        
        # Initialize teacher ensemble
        if self.config.expert_model_paths:
            self.teacher_ensemble = EnsembleTeacher(
                self.config.expert_model_paths,
                self.config.expert_weights,
                str(self.device)
            )
        else:
            # Create mock teachers for demonstration
            self.teacher_ensemble = EnsembleTeacher(
                ["mock_nlp_expert.pt", "mock_math_expert.pt", "mock_reasoning_expert.pt"],
                device=str(self.device)
            )
        
        # Setup optimizer
        if self.config.optimizer_type.lower() == "adamw":
            self.optimizer = optim.AdamW(
                self.student.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer_type.lower() == "adam":
            self.optimizer = optim.Adam(
                self.student.parameters(),
                lr=self.config.learning_rate
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer_type}")
        
        # Setup scheduler
        if self.config.scheduler_type == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.config.num_epochs
            )
        elif self.config.scheduler_type == "step":
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=30, gamma=0.1
            )
        
        # Setup mixed precision scaler
        if self.config.mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        
        self.logger.info("Models and optimizer setup complete")
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.student.train()
        epoch_metrics = defaultdict(list)
        
        for batch_idx, batch in enumerate(train_loader):
            # Parse batch - assuming (inputs, targets) format
            if len(batch) == 2:
                inputs, targets = batch
            else:
                inputs = batch[0]
                targets = None
            
            inputs = inputs.to(self.device)
            if targets is not None:
                targets = targets.to(self.device)
            
            # Forward pass with mixed precision if enabled
            if self.config.mixed_precision:
                with torch.cuda.amp.autocast():
                    loss_dict = self._forward_step(inputs, targets)
                
                # Backward pass
                self.scaler.scale(loss_dict['total_loss']).backward()
                
                # Gradient clipping
                if self.config.gradient_clip_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.student.parameters(), 
                        self.config.gradient_clip_norm
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss_dict = self._forward_step(inputs, targets)
                
                # Backward pass
                loss_dict['total_loss'].backward()
                
                # Gradient clipping
                if self.config.gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.student.parameters(), 
                        self.config.gradient_clip_norm
                    )
                
                self.optimizer.step()
            
            self.optimizer.zero_grad()
            
            # Record metrics
            for key, value in loss_dict.items():
                epoch_metrics[key].append(value.item())
            
            # Log progress
            if batch_idx % self.config.log_interval == 0:
                self.logger.info(
                    f"Epoch {self.current_epoch}, Step {batch_idx}: "
                    f"Loss = {loss_dict['total_loss'].item():.4f}"
                )
            
            self.global_step += 1
        
        # Average metrics
        return {key: np.mean(values) for key, values in epoch_metrics.items()}
    
    def _forward_step(self, inputs: torch.Tensor, 
                     targets: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Perform forward step and compute losses."""
        
        # Student forward pass
        if self.config.feature_matching_layers:
            student_features, student_logits = self.student.get_features(
                inputs, self.config.feature_matching_layers
            )
        else:
            student_logits = self.student(inputs)
            student_features = None
        
        # Teacher ensemble predictions
        teacher_logits = self.teacher_ensemble.get_ensemble_predictions(
            inputs, self.config.ensemble_method
        )
        
        # Teacher features for feature matching
        teacher_features = None
        if self.config.feature_matching_layers:
            teacher_features = self.teacher_ensemble.get_ensemble_features(
                inputs, self.config.feature_matching_layers
            )
        
        # Compute losses
        loss_dict = self.loss_fn(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            hard_targets=targets,
            student_features=student_features,
            teacher_features=teacher_features
        )
        
        return loss_dict
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate the student model."""
        self.student.eval()
        val_metrics = defaultdict(list)
        
        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 2:
                    inputs, targets = batch
                else:
                    inputs = batch[0]
                    targets = None
                
                inputs = inputs.to(self.device)
                if targets is not None:
                    targets = targets.to(self.device)
                
                # Forward pass
                loss_dict = self._forward_step(inputs, targets)
                
                # Record metrics
                for key, value in loss_dict.items():
                    val_metrics[key].append(value.item())
        
        return {key: np.mean(values) for key, values in val_metrics.items()}
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.student.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config.to_dict(),
            'best_val_loss': self.best_val_loss
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save regular checkpoint
        checkpoint_path = self.output_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.output_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            self.logger.info(f"New best model saved at epoch {epoch}")
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader = None):
        """Main training loop."""
        self.logger.info("Starting knowledge distillation training")
        self.logger.info(f"Configuration: {self.config.to_dict()}")
        
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            
            # Training
            train_metrics = self.train_epoch(train_loader)
            
            # Validation
            val_metrics = {}
            if val_loader is not None and epoch % self.config.validate_interval == 0:
                val_metrics = self.validate(val_loader)
                
                # Check for best model
                if val_metrics['total_loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['total_loss']
                    self.save_checkpoint(epoch, is_best=True)
            
            # Learning rate scheduling
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Logging
            self.logger.info(
                f"Epoch {epoch}/{self.config.num_epochs} - "
                f"Train Loss: {train_metrics['total_loss']:.4f}"
            )
            
            if val_metrics:
                self.logger.info(f"Val Loss: {val_metrics['total_loss']:.4f}")
            
            # Save checkpoint
            if epoch % self.config.save_interval == 0:
                self.save_checkpoint(epoch)
            
            # Record metrics
            self.metrics.record_metric('train_loss', train_metrics['total_loss'])
            if val_metrics:
                self.metrics.record_metric('val_loss', val_metrics['total_loss'])
        
        self.logger.info("Training completed successfully")


# Mock dataset for demonstration
class MockDistillationDataset(Dataset):
    """Mock dataset for knowledge distillation demonstration."""
    
    def __init__(self, num_samples: int = 1000, input_dim: int = 768, num_classes: int = 10):
        self.num_samples = num_samples
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # Generate synthetic data
        torch.manual_seed(42)
        self.data = torch.randn(num_samples, input_dim)
        self.targets = torch.randint(0, num_classes, (num_samples,))
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


def create_distillation_trainer(config: Optional[DistillationConfig] = None) -> DistillationTrainer:
    """Factory function to create a distillation trainer."""
    if config is None:
        config = DistillationConfig()
    
    trainer = DistillationTrainer(config)
    trainer.setup_models()
    
    return trainer


async def run_distillation_example():
    """Example usage of the knowledge distillation system."""
    
    print("🧠 Knowledge Distillation Training Script")
    print("=" * 50)
    
    # Create configuration
    config = DistillationConfig(
        temperature=2.0,
        alpha=0.7,
        beta=0.3,
        learning_rate=1e-4,
        batch_size=32,
        num_epochs=20,
        student_hidden_size=256,
        student_num_layers=4,
        expert_model_paths=["expert_nlp.pt", "expert_math.pt", "expert_reasoning.pt"],
        output_dir="outputs/distillation_demo"
    )
    
    print(f"Configuration:")
    print(f"  Temperature: {config.temperature}")
    print(f"  Alpha (distillation weight): {config.alpha}")
    print(f"  Beta (hard target weight): {config.beta}")
    print(f"  Student architecture: {config.student_num_layers} layers, {config.student_hidden_size} hidden")
    print(f"  Expert models: {len(config.expert_model_paths)} experts")
    
    # Create trainer
    trainer = create_distillation_trainer(config)
    
    # Create mock dataset
    train_dataset = MockDistillationDataset(num_samples=1000)
    val_dataset = MockDistillationDataset(num_samples=200)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    
    print(f"\nDataset:")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    print(f"  Batch size: {config.batch_size}")
    
    print(f"\nStarting distillation training...")
    
    # Train the model
    trainer.train(train_loader, val_loader)
    
    print(f"\n✅ Knowledge distillation completed successfully!")
    print(f"📁 Outputs saved to: {config.output_dir}")
    
    # Demonstrate the trained student model
    print(f"\nTesting student model...")
    trainer.student.eval()
    
    with torch.no_grad():
        sample_input = torch.randn(1, 768).to(trainer.device)
        student_output = trainer.student(sample_input)
        teacher_output = trainer.teacher_ensemble.get_ensemble_predictions(sample_input)
        
        print(f"Sample predictions:")
        print(f"  Student: {torch.softmax(student_output, dim=-1)[0][:5].cpu().numpy()}")
        print(f"  Teacher: {torch.softmax(teacher_output, dim=-1)[0][:5].cpu().numpy()}")


if __name__ == "__main__":
    # Load expert models (teachers) and initialize student model
    experts = [
        ExpertModel.load_pretrained("expert_nlp.pt"),
        ExpertModel.load_pretrained("expert_math.pt"), 
        ExpertModel.load_pretrained("expert_reasoning.pt")
    ]
    student = StudentModel(hidden_size=512, num_layers=6)  # a smaller model
    
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-4)
    temperature = 2.0  # softening factor for distillation
    
    # Mock training data for demonstration
    training_data = DataLoader(MockDistillationDataset(), batch_size=32)
    
    for batch in training_data:
        inputs, _ = batch
        # Get soft targets from each expert
        expert_logits = [expert(inputs) for expert in experts]
        # Average the expert predictions (ensemble teacher behavior)
        mean_logits = sum(expert_logits) / len(expert_logits)
        # Compute distillation loss between student and mean teacher logits
        student_logits = student(inputs)
        loss = F.kl_div(
            F.log_softmax(student_logits/temperature, dim=-1),
            F.softmax(mean_logits/temperature, dim=-1),
            reduction='batchmean'
        )
        # Standard backprop on student model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Demo: just run one batch
        print(f"Distillation loss: {loss.item():.4f}")
        break
    
    print("\n" + "="*50)
    print("Running full production distillation example...")
    print("="*50)
    
    # Run the full example
    asyncio.run(run_distillation_example())