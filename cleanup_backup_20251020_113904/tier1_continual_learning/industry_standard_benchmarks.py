#!/usr/bin/env python3
"""
Industry-Standard Continual Learning Benchmarks
================================================

Research implementation following published best practices:
- Proper CNN architectures (ResNet-18/34)
- Literature-standard hyperparameters
- Validation splits with early stopping
- Model checkpointing
- TensorBoard experiment tracking
- Reproducible evaluation

References:
- Kirkpatrick et al. (2017): EWC with λ ∈ [10, 400]
- Rebuffi et al. (2017): iCaRL experience replay
- Lopez-Paz & Ranzato (2017): GEM benchmark protocols
- He et al. (2016): ResNet architecture & training

Suitable for:
✓ Research prototyping and ablation studies
✓ Benchmark comparisons on standard datasets
✓ Educational purposes and continual learning exploration

Production deployment requires:
- Multi-GPU distributed training (DDP)
- Larger scale datasets (ImageNet, MS-COCO)
- Advanced monitoring and serving infrastructure
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime
import json
import sys
from collections import defaultdict, deque
import random
import yaml
import logging

# Add project to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import Advanced Continual Learning Engine
from training.advanced_continual_learning import (
    AdvancedContinualLearningEngine,
    SamplingStrategy,
    create_advanced_cl_engine
)

# Import DER++ for baseline comparison
from training.der_plus_plus import (
    DERPlusPlusWrapper,
    create_der_plus_plus_engine
)


def setup_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Setup logger with proper formatting."""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


@dataclass
class BenchmarkMetrics:
    """Standard continual learning metrics."""
    
    average_accuracy: float = 0.0
    final_accuracy: float = 0.0
    forgetting_measure: float = 0.0
    forward_transfer: float = 0.0
    backward_transfer: float = 0.0
    task_accuracies: List[float] = field(default_factory=list)
    task_learning_times: List[float] = field(default_factory=list)
    task_retention: List[float] = field(default_factory=list)
    learning_efficiency: float = 0.0
    stability_plasticity_ratio: float = 0.0


@dataclass 
class BenchmarkResult:
    """Benchmark result structure."""
    
    dataset_name: str
    method: str
    success_level: str = "NEEDS_WORK"
    overall_score: float = 0.0
    metrics: BenchmarkMetrics = field(default_factory=BenchmarkMetrics)
    num_tasks: int = 5
    epochs_per_task: int = 20
    total_time: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    success: bool = True
    error_message: str = ""
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        return result


class Conv2dBlock(nn.Module):
    """Convolutional block with BatchNorm and ReLU."""
    
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    """Standard ResNet residual block."""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, downsample: nn.Module = None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        identity = x
        
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        return self.relu(out)


class SimpleCNN(nn.Module):
    """Simple CNN backbone for smaller datasets."""
    
    def __init__(self, in_channels: int, feature_dim: int = 512):
        super().__init__()
        self.features = nn.Sequential(
            Conv2dBlock(in_channels, 64),
            nn.MaxPool2d(2),
            Conv2dBlock(64, 128),
            nn.MaxPool2d(2),
            Conv2dBlock(128, 256),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(256, feature_dim)
        self.feature_dim = feature_dim
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ResNet18Backbone(nn.Module):
    """ResNet-18 backbone for feature extraction."""
    
    def __init__(self, in_channels: int, feature_dim: int = 512):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        
        # ResNet layers
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, feature_dim) if feature_dim != 512 else nn.Identity()
        self.feature_dim = feature_dim
    
    def _make_layer(self, in_channels: int, out_channels: int, blocks: int, stride: int = 1):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride, downsample))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ReplayBuffer:
    """Experience replay buffer with reservoir sampling."""
    
    def __init__(self, max_size: int = 2000, balanced: bool = True):
        self.max_size = max_size
        self.balanced = balanced
        self.buffer = []
        self.task_buffers = defaultdict(list)
        self.total_seen = 0
    
    def add_task_samples(self, task_id: int, dataset: torch.utils.data.Dataset, samples_per_task: int = 200):
        """Add samples using reservoir sampling."""
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        
        for idx in indices[:samples_per_task]:
            data, target = dataset[idx]
            sample = (data, target, task_id)
            
            if len(self.buffer) < self.max_size:
                self.buffer.append(sample)
                self.task_buffers[task_id].append(sample)
            else:
                # Reservoir sampling
                self.total_seen += 1
                j = random.randint(0, self.total_seen)
                if j < self.max_size:
                    old_sample = self.buffer[j]
                    old_task_id = old_sample[2]
                    if old_sample in self.task_buffers[old_task_id]:
                        self.task_buffers[old_task_id].remove(old_sample)
                    
                    self.buffer[j] = sample
                    self.task_buffers[task_id].append(sample)
    
    def get_replay_batch(self, batch_size: int = 64) -> List[Tuple]:
        """Get replay batch with optional balancing."""
        if not self.buffer:
            return []
        
        if not self.balanced or len(self.task_buffers) <= 1:
            return random.sample(self.buffer, min(batch_size, len(self.buffer)))
        
        # Balance across tasks
        samples_per_task = batch_size // len(self.task_buffers)
        replay_data = []
        
        for task_id, task_samples in self.task_buffers.items():
            if task_samples:
                n_samples = min(samples_per_task, len(task_samples))
                replay_data.extend(random.sample(task_samples, n_samples))
        
        # Fill remaining
        remaining = batch_size - len(replay_data)
        if remaining > 0 and self.buffer:
            additional = random.sample(self.buffer, min(remaining, len(self.buffer)))
            replay_data.extend(additional)
        
        return replay_data


class ContinualLearningModel(nn.Module):
    """Industry-standard continual learning model."""
    
    def __init__(self, config: Dict, input_channels: int, num_classes: int, num_tasks: int = 5):
        super().__init__()
        self.config = config
        self.num_classes = num_classes
        self.num_tasks = num_tasks
        self.strategy = config['continual_learning']['strategy']
        
        # Feature extractor (CNN backbone)
        arch = config['model']['architecture']
        feature_dim = config['model']['feature_dim']
        
        if arch == 'resnet18':
            self.feature_extractor = ResNet18Backbone(input_channels, feature_dim)
        elif arch == 'simple_cnn':
            self.feature_extractor = SimpleCNN(input_channels, feature_dim)
        else:
            raise ValueError(f"Unknown architecture: {arch}")
        
        # Task-specific heads
        if config['model']['use_multihead']:
            head_dim = config['model']['head_hidden_dim']
            self.task_heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(feature_dim, head_dim),
                    nn.ReLU(),
                    nn.Dropout(config['model']['dropout']),
                    nn.Linear(head_dim, num_classes // num_tasks)
                )
                for _ in range(num_tasks)
            ])
        else:
            self.classifier = nn.Linear(feature_dim, num_classes)
        
        # EWC components
        if self.strategy in ['ewc', 'optimized']:
            self.fisher_information = {}
            self.optimal_params = {}
            self.ewc_lambda = config['continual_learning']['ewc_lambda']
            self.online_ewc = config['continual_learning']['online_ewc']
            self.ewc_gamma = config['continual_learning']['ewc_gamma']
        
        # Replay buffer
        if self.strategy in ['replay', 'optimized']:
            buffer_size = config['continual_learning']['replay_buffer_size']
            self.replay_buffer = ReplayBuffer(max_size=buffer_size, balanced=True)
        
        # Knowledge distillation
        if self.strategy == 'optimized':
            self.previous_model = None
            self.distillation_temperature = config['continual_learning']['distillation_temperature']
            self.distillation_alpha = config['continual_learning']['distillation_alpha']
        
        # Advanced Continual Learning Engine
        if self.strategy == 'advanced':
            buffer_size = config['continual_learning'].get('replay_buffer_size', 2000)
            self.advanced_cl_engine = create_advanced_cl_engine(
                buffer_capacity=buffer_size,
                strategy="uncertainty",
                use_asymmetric_ce=True,
                use_contrastive_reg=True,
                use_gradient_surgery=True,
                use_model_ensemble=True,
                distillation_weight=0.5,
                contrastive_weight=0.1
            )
            self.advanced_cl_enabled = True
        elif self.strategy == 'der++':
            # DER++ - Official baseline for comparison
            buffer_size = config['continual_learning'].get('replay_buffer_size', 2000)
            alpha = config['continual_learning'].get('distillation_alpha', 0.5)
            self.advanced_cl_engine = DERPlusPlusWrapper(alpha=alpha, buffer_size=buffer_size)
            self.advanced_cl_enabled = True
        else:
            self.advanced_cl_engine = None
            self.advanced_cl_enabled = False
    
    def forward(self, x, task_id: Optional[int] = None):
        features = self.feature_extractor(x)
        
        if self.config['model']['use_multihead']:
            if task_id is None:
                # During evaluation without task_id, use first head or raise error
                raise ValueError("task_id required for multihead model")
            return self.task_heads[task_id](features)
        else:
            return self.classifier(features)
    
    def compute_ewc_loss(self, current_loss: torch.Tensor) -> torch.Tensor:
        """Compute EWC regularization loss."""
        if not self.fisher_information:
            return current_loss
        
        ewc_loss = 0
        for name, param in self.named_parameters():
            # Skip previous_model parameters
            if name.startswith('previous_model.'):
                continue
            if name in self.fisher_information:
                fisher = self.fisher_information[name]
                optimal = self.optimal_params[name]
                ewc_loss += (fisher * (param - optimal) ** 2).sum()
        
        return current_loss + self.ewc_lambda * ewc_loss
    
    def compute_distillation_loss(self, student_output: torch.Tensor, data: torch.Tensor, task_id: int) -> torch.Tensor:
        """Compute knowledge distillation loss."""
        if self.previous_model is None:
            return torch.tensor(0.0, device=student_output.device)
        
        self.previous_model.eval()
        with torch.no_grad():
            teacher_output = self.previous_model(data, task_id)
        
        T = self.distillation_temperature
        soft_student = F.log_softmax(student_output / T, dim=1)
        soft_teacher = F.softmax(teacher_output / T, dim=1)
        
        distillation_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (T * T)
        return distillation_loss
    
    def update_ewc(self, dataset: torch.utils.data.Dataset, device: torch.device, task_id: int = None):
        """Update Fisher Information Matrix."""
        if self.strategy not in ['ewc', 'optimized']:
            return
        
        self.eval()
        
        # Filter out previous_model parameters from EWC computation
        trainable_params = {name: param for name, param in self.named_parameters() 
                           if not name.startswith('previous_model.')}
        
        if self.online_ewc and self.fisher_information:
            new_fisher = {name: torch.zeros_like(param) for name, param in trainable_params.items()}
        else:
            new_fisher = {name: torch.zeros_like(param) for name, param in trainable_params.items()}
        
        loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
        
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            
            self.zero_grad()
            output = self(data, task_id=task_id)
            loss = F.cross_entropy(output, target)
            loss.backward()
            
            for name, param in trainable_params.items():
                if param.grad is not None:
                    new_fisher[name] += param.grad ** 2
        
        # Normalize
        for name in new_fisher:
            new_fisher[name] /= len(dataset)
        
        # Online update
        if self.online_ewc and self.fisher_information:
            for name in new_fisher:
                if name in self.fisher_information:
                    self.fisher_information[name] = (
                        self.ewc_gamma * self.fisher_information[name] + 
                        (1 - self.ewc_gamma) * new_fisher[name]
                    )
                else:
                    self.fisher_information[name] = new_fisher[name]
        else:
            self.fisher_information = new_fisher
        
        self.optimal_params = {name: param.clone().detach() for name, param in trainable_params.items()}
    
    def save_for_distillation(self, config: Dict, input_channels: int):
        """Save model for distillation."""
        if self.strategy == 'optimized':
            self.previous_model = ContinualLearningModel(config, input_channels, self.num_classes, self.num_tasks)
            # Use strict=False to avoid key mismatch errors
            self.previous_model.load_state_dict(self.state_dict(), strict=False)
            self.previous_model.eval()
            for param in self.previous_model.parameters():
                param.requires_grad = False


class MappedTaskDataset(torch.utils.data.Dataset):
    """Dataset wrapper for task-incremental learning."""
    
    def __init__(self, base_dataset, indices, task_classes, task_id):
        self.base_dataset = base_dataset
        self.indices = indices
        self.task_classes = task_classes
        self.task_id = task_id
        self.label_mapping = {cls: idx for idx, cls in enumerate(task_classes)}
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        data, original_label = self.base_dataset[real_idx]
        mapped_label = self.label_mapping[original_label]
        return data, mapped_label


class IndustryStandardBenchmarkSuite:
    """Industry-standard benchmarking suite with best practices."""
    
    def __init__(self, config_path: str = None):
        """Initialize with config file."""
        if config_path is None:
            config_path = Path(__file__).parent / "config.yaml"
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup logger
        self.logger = setup_logger(__name__, self.config['experiment']['log_level'])
        
        # Set seed for reproducibility
        if self.config['experiment'].get('deterministic', True):
            set_seed(self.config['experiment']['seed'])
        
        # Device setup
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.logger.info("Using CUDA GPU")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device('mps')
            self.logger.info("Using Apple Silicon GPU (MPS)")
        else:
            self.device = torch.device('cpu')
            self.logger.info("Using CPU")
        
        # Setup directories
        self.data_dir = Path(self.config['datasets']['data_dir'])
        self.data_dir.mkdir(exist_ok=True)
        
        self.checkpoint_dir = Path(self.config['experiment']['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.results_dir = Path(self.config['experiment']['results_dir'])
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # TensorBoard
        if self.config['experiment']['use_tensorboard']:
            tensorboard_dir = Path(self.config['experiment']['tensorboard_dir'])
            tensorboard_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.writer = SummaryWriter(tensorboard_dir / f"run_{timestamp}")
        else:
            self.writer = None
    
    def get_transforms(self, dataset_name: str, train: bool = True):
        """Get appropriate transforms for dataset."""
        dataset_config = self.config['datasets'].get(dataset_name, {})
        
        if train and self.config['datasets']['use_augmentation']:
            if dataset_name in ['mnist', 'fashion_mnist', 'omniglot']:
                # Grayscale datasets (28x28 or 105x105)
                transform_list = []
                if dataset_name == 'omniglot':
                    transform_list.append(transforms.Resize((32, 32)))
                transform_list.extend([
                    transforms.RandomRotation(10),
                    transforms.RandomAffine(0, translate=(0.1, 0.1)),
                    transforms.ToTensor(),
                    transforms.Normalize(dataset_config['mean'], dataset_config['std'])
                ])
                transform = transforms.Compose(transform_list)
            elif dataset_name in ['cifar10', 'cifar100', 'svhn']:
                # Standard CIFAR augmentation (32x32 RGB)
                transform = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(0.1, 0.1, 0.1),
                    transforms.ToTensor(),
                    transforms.Normalize(dataset_config['mean'], dataset_config['std'])
                ])
            elif dataset_name == 'tiny_imagenet':
                # TinyImageNet augmentation (64x64 RGB)
                transform = transforms.Compose([
                    transforms.RandomCrop(64, padding=8),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
                    transforms.RandomRotation(15),
                    transforms.ToTensor(),
                    transforms.Normalize(dataset_config['mean'], dataset_config['std'])
                ])
            else:
                # Fallback
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(dataset_config['mean'], dataset_config['std'])
                ])
        else:
            # Validation/test transforms
            if dataset_name == 'omniglot':
                transform = transforms.Compose([
                    transforms.Resize((32, 32)),
                    transforms.ToTensor(),
                    transforms.Normalize(dataset_config['mean'], dataset_config['std'])
                ])
            else:
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(dataset_config['mean'], dataset_config['std'])
                ])
        
        return transform
    
    def load_dataset(self, dataset_name: str, train: bool = True):
        """Load dataset with proper transforms."""
        transform = self.get_transforms(dataset_name, train)
        
        if dataset_name == 'mnist':
            return torchvision.datasets.MNIST(
                root=str(self.data_dir), train=train, transform=transform, download=True
            )
        elif dataset_name == 'fashion_mnist':
            return torchvision.datasets.FashionMNIST(
                root=str(self.data_dir), train=train, transform=transform, download=True
            )
        elif dataset_name == 'cifar10':
            return torchvision.datasets.CIFAR10(
                root=str(self.data_dir), train=train, transform=transform, download=True
            )
        elif dataset_name == 'cifar100':
            return torchvision.datasets.CIFAR100(
                root=str(self.data_dir), train=train, transform=transform, download=True
            )
        elif dataset_name == 'svhn':
            split = 'train' if train else 'test'
            return torchvision.datasets.SVHN(
                root=str(self.data_dir), split=split, transform=transform, download=True
            )
        elif dataset_name == 'omniglot':
            # Omniglot has background/evaluation sets, we'll use background for train
            # and evaluation for test
            background = train
            return torchvision.datasets.Omniglot(
                root=str(self.data_dir), background=background, transform=transform, download=True
            )
        elif dataset_name == 'tiny_imagenet':
            return self._load_tiny_imagenet(train, transform)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
    
    def _load_tiny_imagenet(self, train: bool, transform):
        """Load TinyImageNet dataset."""
        tiny_imagenet_path = self.data_dir / "tiny-imagenet-200"
        
        # Check if dataset exists
        if not tiny_imagenet_path.exists():
            self.logger.warning(
                f"TinyImageNet not found at {tiny_imagenet_path}. "
                "Please download from http://cs231n.stanford.edu/tiny-imagenet-200.zip "
                "and extract to {self.data_dir}"
            )
            raise FileNotFoundError(
                f"TinyImageNet dataset not found. Download from: "
                "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
            )
        
        if train:
            data_path = tiny_imagenet_path / "train"
        else:
            data_path = tiny_imagenet_path / "val"
        
        return torchvision.datasets.ImageFolder(root=str(data_path), transform=transform)
    
    def create_task_sequence(self, dataset, num_tasks: int):
        """Create task sequence with train/val split."""
        # Get targets
        if hasattr(dataset, 'targets'):
            targets = dataset.targets
            if not isinstance(targets, torch.Tensor):
                targets = torch.tensor(targets)
        else:
            targets = torch.tensor([dataset[i][1] for i in range(len(dataset))])
        
        unique_classes = torch.unique(targets).tolist()
        classes_per_task = len(unique_classes) // num_tasks
        
        tasks = []
        for task_idx in range(num_tasks):
            start_class = task_idx * classes_per_task
            if task_idx == num_tasks - 1:
                task_classes = unique_classes[start_class:]
            else:
                task_classes = unique_classes[start_class:start_class + classes_per_task]
            
            task_mask = torch.zeros(len(targets), dtype=torch.bool)
            for cls in task_classes:
                task_mask |= (targets == cls)
            
            task_indices = torch.where(task_mask)[0].tolist()
            
            # Train/val split
            val_split = self.config['training']['val_split']
            random.shuffle(task_indices)
            split_idx = int(len(task_indices) * (1 - val_split))
            
            train_indices = task_indices[:split_idx]
            val_indices = task_indices[split_idx:]
            
            train_dataset = MappedTaskDataset(dataset, train_indices, task_classes, task_idx)
            val_dataset = MappedTaskDataset(dataset, val_indices, task_classes, task_idx)
            
            tasks.append((train_dataset, val_dataset))
        
        return tasks
    
    def train_on_task(self, model: ContinualLearningModel, train_dataset, val_dataset, task_id: int, global_step: int):
        """Train model on task with validation and early stopping."""
        config = self.config['training']
        
        model.train()
        model = model.to(self.device)
        
        # Add to replay buffer
        if hasattr(model, 'replay_buffer'):
            model.replay_buffer.add_task_samples(task_id, train_dataset, samples_per_task=200)
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config['num_workers'],
            pin_memory=True
        )
        
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.config['evaluation']['eval_batch_size'],
            shuffle=False,
            num_workers=config['num_workers'],
            pin_memory=True
        )
        
        # Optimizer
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # Scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['epochs_per_task'],
            eta_min=config['learning_rate'] * config['lr_min_factor']
        )
        
        best_val_acc = 0.0
        patience_counter = 0
        start_time = time.time()
        
        for epoch in range(config['epochs_per_task']):
            model.train()
            epoch_loss = 0.0
            correct = 0
            total = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                if self.config['model']['use_multihead']:
                    output = model(data, task_id=task_id)
                else:
                    output = model(data)
                
                loss = F.cross_entropy(output, target)
                
                # Continual learning losses
                if model.strategy in ['advanced', 'der++'] and model.advanced_cl_enabled:
                    # Use Advanced Continual Learning Engine or DER++
                    replay_batch_size = int(config['batch_size'] * self.config['continual_learning'].get('replay_batch_ratio', 0.5))
                    
                    # Compute replay loss with all advanced features
                    loss, loss_info = model.advanced_cl_engine.compute_replay_loss(
                        model=model,
                        current_data=data,
                        current_target=target,
                        current_loss=loss,
                        task_id=task_id,
                        replay_batch_size=replay_batch_size
                    )
                    
                    # Log advanced metrics
                    if batch_idx % self.config['experiment']['log_interval'] == 0 and self.writer:
                        strategy_name = 'der++' if model.strategy == 'der++' else 'advanced'
                        for key, value in loss_info.items():
                            if isinstance(value, (int, float)):
                                self.writer.add_scalar(f'Task{task_id}/{strategy_name}_{key}', value, global_step)
                
                elif model.strategy == 'optimized':
                    loss = model.compute_ewc_loss(loss)
                    
                    if task_id > 0:
                        distill_loss = model.compute_distillation_loss(output, data, task_id)
                        loss = loss + model.distillation_alpha * distill_loss
                    
                    if hasattr(model, 'replay_buffer'):
                        replay_batch_size = int(config['batch_size'] * self.config['continual_learning']['replay_batch_ratio'])
                        replay_data = model.replay_buffer.get_replay_batch(replay_batch_size)
                        if replay_data:
                            replay_loss = self._compute_replay_loss(model, replay_data)
                            loss = loss + 0.5 * replay_loss
                
                elif model.strategy == 'ewc':
                    loss = model.compute_ewc_loss(loss)
                
                elif model.strategy == 'replay' and hasattr(model, 'replay_buffer'):
                    replay_batch_size = int(config['batch_size'] * 0.5)
                    replay_data = model.replay_buffer.get_replay_batch(replay_batch_size)
                    if replay_data:
                        replay_loss = self._compute_replay_loss(model, replay_data)
                        loss = loss + 0.5 * replay_loss
                
                # Check for NaN/Inf loss before backward (numerical stability)
                if torch.isnan(loss) or torch.isinf(loss):
                    self.logger.warning(f"Task {task_id} Epoch {epoch} Batch {batch_idx}: Loss is {loss.item()}, skipping batch")
                    continue
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip_norm'])
                optimizer.step()
                
                # Store experiences in advanced CL engine or DER++
                if model.strategy in ['advanced', 'der++'] and model.advanced_cl_enabled:
                    # Store a subset of the batch (e.g., 10% to avoid memory issues)
                    store_fraction = 0.1
                    num_to_store = max(1, int(data.size(0) * store_fraction))
                    indices = torch.randperm(data.size(0))[:num_to_store]
                    
                    for idx in indices:
                        model.advanced_cl_engine.store_experience(
                            model=model,
                            data=data[idx:idx+1],
                            target=target[idx:idx+1],
                            task_id=task_id,
                            compute_features=True
                        )
                
                epoch_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += (pred == target).sum().item()
                total += target.size(0)
                
                # Logging
                if batch_idx % self.config['experiment']['log_interval'] == 0:
                    if self.writer:
                        self.writer.add_scalar(f'Task{task_id}/train_loss', loss.item(), global_step)
                        self.writer.add_scalar(f'Task{task_id}/learning_rate', scheduler.get_last_lr()[0], global_step)
                    global_step += 1
            
            scheduler.step()
            
            # Validation
            val_acc = self.evaluate_on_task(model, val_loader, task_id)
            train_acc = correct / total if total > 0 else 0.0
            
            self.logger.info(f"Task {task_id+1}, Epoch {epoch+1}/{config['epochs_per_task']}: "
                           f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Loss: {epoch_loss/len(train_loader):.4f}")
            
            if self.writer:
                self.writer.add_scalar(f'Task{task_id}/train_accuracy', train_acc, epoch)
                self.writer.add_scalar(f'Task{task_id}/val_accuracy', val_acc, epoch)
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                
                # Save best model
                if self.config['experiment']['save_checkpoints']:
                    checkpoint_path = self.checkpoint_dir / f"task{task_id}_best.pt"
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'epoch': epoch,
                        'val_acc': val_acc
                    }, checkpoint_path)
            else:
                patience_counter += 1
                if patience_counter >= config['early_stopping_patience']:
                    self.logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        training_time = time.time() - start_time
        
        # Update continual learning components
        if model.strategy in ['advanced', 'der++'] and model.advanced_cl_enabled:
            # Finalize task with advanced CL engine or DER++
            model.advanced_cl_engine.finish_task(
                model=model,
                task_id=task_id,
                performance=best_val_acc
            )
            
            # Log statistics
            stats = model.advanced_cl_engine.get_statistics()
            strategy_name = "DER++" if model.strategy == 'der++' else "Advanced CL"
            self.logger.info(f"{strategy_name} Statistics after Task {task_id+1}:")
            self.logger.info(f"  Buffer utilization: {stats['buffer']['utilization']:.2%}")
            if model.strategy == 'advanced':
                self.logger.info(f"  Buffer quality score: {stats['buffer']['quality_score']:.4f}")
                self.logger.info(f"  Ensemble size: {stats['ensemble']['size']}")
                self.logger.info(f"  Gradient surgeries: {stats['training']['gradient_surgeries']}")
            else:
                self.logger.info(f"  Alpha (distillation weight): {stats.get('alpha', 0.5)}")
        
        elif model.strategy in ['ewc', 'optimized']:
            model.update_ewc(train_dataset, self.device, task_id=task_id)
        
        if model.strategy == 'optimized':
            # Get input channels from first sample
            sample_data, _ = train_dataset[0]
            input_channels = sample_data.shape[0]
            model.save_for_distillation(self.config, input_channels)
        
        return {'training_time': training_time, 'best_val_acc': best_val_acc}, global_step
    
    def _compute_replay_loss(self, model, replay_data):
        """Compute replay loss."""
        if not replay_data:
            return torch.tensor(0.0, device=self.device)
        
        batch_data = torch.stack([d for d, _, _ in replay_data]).to(self.device)
        batch_targets = torch.tensor([t for _, t, _ in replay_data], device=self.device)
        batch_task_ids = [tid for _, _, tid in replay_data]
        
        if self.config['model']['use_multihead']:
            replay_loss = 0.0
            count = 0
            unique_tasks = set(batch_task_ids)
            for task_id in unique_tasks:
                task_mask = torch.tensor([tid == task_id for tid in batch_task_ids], device=self.device)
                if task_mask.any():
                    task_data = batch_data[task_mask]
                    task_targets = batch_targets[task_mask]
                    output = model(task_data, task_id=task_id)
                    replay_loss += F.cross_entropy(output, task_targets)
                    count += 1
            return replay_loss / count if count > 0 else torch.tensor(0.0, device=self.device)
        else:
            output = model(batch_data)
            return F.cross_entropy(output, batch_targets)
    
    def evaluate_on_task(self, model, data_loader, task_id: int) -> float:
        """Evaluate model on task."""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                if self.config['model']['use_multihead']:
                    output = model(data, task_id=task_id)
                else:
                    output = model(data)
                
                pred = output.argmax(dim=1)
                correct += (pred == target).sum().item()
                total += target.size(0)
        
        return correct / total if total > 0 else 0.0
    
    def run_benchmark(self, dataset_name: str, strategy: str = None):
        """Run industry-standard benchmark."""
        if strategy:
            self.config['continual_learning']['strategy'] = strategy
        
        strategy = self.config['continual_learning']['strategy']
        num_tasks = self.config['continual_learning']['num_tasks']
        
        self.logger.info(f"Starting benchmark: {dataset_name}, Strategy: {strategy}")
        self.logger.info(f"Config: {json.dumps(self.config, indent=2)}")
        
        try:
            start_time = time.time()
            
            # Load datasets
            train_dataset = self.load_dataset(dataset_name, train=True)
            test_dataset = self.load_dataset(dataset_name, train=False)
            
            # Create task sequences
            train_val_tasks = self.create_task_sequence(train_dataset, num_tasks)
            test_tasks = self.create_task_sequence(test_dataset, num_tasks)
            
            # Get dataset info
            sample_data, _ = train_dataset[0]
            input_channels = sample_data.shape[0]
            num_classes = len(set([train_dataset[i][1] for i in range(len(train_dataset))]))
            
            # Create model
            model = ContinualLearningModel(self.config, input_channels, num_classes, num_tasks)
            total_params = sum(p.numel() for p in model.parameters())
            
            self.logger.info(f"Model: {self.config['model']['architecture']}, Parameters: {total_params:,}")
            
            # Training loop
            task_learning_times = []
            all_task_accuracies = []
            global_step = 0
            
            for task_idx in range(num_tasks):
                self.logger.info(f"\n{'='*60}")
                self.logger.info(f"Task {task_idx + 1}/{num_tasks}")
                self.logger.info(f"{'='*60}")
                
                train_dataset, val_dataset = train_val_tasks[task_idx]
                
                train_result, global_step = self.train_on_task(
                    model, train_dataset, val_dataset, task_idx, global_step
                )
                task_learning_times.append(train_result['training_time'])
                
                # Evaluate on all test tasks
                current_accuracies = []
                for eval_task_idx in range(task_idx + 1):
                    _, test_val_dataset = test_tasks[eval_task_idx]
                    test_loader = torch.utils.data.DataLoader(
                        test_val_dataset,
                        batch_size=self.config['evaluation']['eval_batch_size'],
                        shuffle=False
                    )
                    acc = self.evaluate_on_task(model, test_loader, eval_task_idx)
                    current_accuracies.append(acc)
                    self.logger.info(f"Task {eval_task_idx + 1} test accuracy: {acc:.4f}")
                
                all_task_accuracies.append(current_accuracies)
            
            # Compute metrics
            metrics = self._compute_metrics(all_task_accuracies, task_learning_times)
            
            # Determine success level using dataset-specific thresholds if available
            dataset_thresholds = self.config['evaluation'].get('dataset_thresholds', {})
            if dataset_name in dataset_thresholds:
                thresholds = dataset_thresholds[dataset_name]
                self.logger.info(f"Using dataset-specific thresholds for {dataset_name}")
            else:
                thresholds = self.config['evaluation']['thresholds']
                self.logger.info(f"Using default thresholds for {dataset_name}")
            
            if (metrics.average_accuracy >= thresholds['excellent']['avg_accuracy'] and 
                metrics.forgetting_measure <= thresholds['excellent']['forgetting']):
                success_level = "EXCELLENT"
                overall_score = 0.95
            elif (metrics.average_accuracy >= thresholds['good']['avg_accuracy'] and 
                  metrics.forgetting_measure <= thresholds['good']['forgetting']):
                success_level = "GOOD"
                overall_score = 0.8
            elif (metrics.average_accuracy >= thresholds['acceptable']['avg_accuracy'] and 
                  metrics.forgetting_measure <= thresholds['acceptable']['forgetting']):
                success_level = "ACCEPTABLE"
                overall_score = 0.65
            else:
                success_level = "NEEDS_WORK"
                overall_score = 0.5
            
            total_time = time.time() - start_time
            
            result = BenchmarkResult(
                dataset_name=dataset_name,
                method=strategy,
                success_level=success_level,
                overall_score=overall_score,
                metrics=metrics,
                num_tasks=num_tasks,
                epochs_per_task=self.config['training']['epochs_per_task'],
                total_time=total_time,
                success=True
            )
            
            self.logger.info(f"\n{'='*80}")
            self.logger.info(f"RESULTS: {success_level}")
            self.logger.info(f"{'='*80}")
            self.logger.info(f"Overall Score: {overall_score:.2f}")
            self.logger.info(f"Average Accuracy: {metrics.average_accuracy:.4f}")
            self.logger.info(f"Forgetting Measure: {metrics.forgetting_measure:.4f}")
            self.logger.info(f"Forward Transfer: {metrics.forward_transfer:.4f}")
            self.logger.info(f"Total Time: {total_time:.2f}s")
            
            # Save results
            self._save_result(result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Benchmark failed: {e}", exc_info=True)
            return BenchmarkResult(
                dataset_name=dataset_name,
                method=strategy,
                success=False,
                error_message=str(e)
            )
    
    def _compute_metrics(self, all_task_accuracies, task_times):
        """Compute continual learning metrics."""
        num_tasks = len(all_task_accuracies)
        
        final_accuracies = all_task_accuracies[-1]
        average_accuracy = np.mean(final_accuracies)
        
        # Forgetting
        forgetting_scores = []
        for task_idx in range(num_tasks - 1):
            peak_acc = all_task_accuracies[task_idx][task_idx]
            final_acc = all_task_accuracies[-1][task_idx]
            forgetting = max(0, peak_acc - final_acc)
            forgetting_scores.append(forgetting)
        forgetting_measure = np.mean(forgetting_scores) if forgetting_scores else 0.0
        
        # Forward transfer
        forward_transfer_scores = []
        random_baseline = 1.0 / self.config['model']['num_classes']
        for task_idx in range(1, num_tasks):
            first_acc = all_task_accuracies[task_idx][task_idx]
            forward_transfer = max(0, first_acc - random_baseline)
            forward_transfer_scores.append(forward_transfer)
        forward_transfer = np.mean(forward_transfer_scores) if forward_transfer_scores else 0.0
        
        # Task retention
        task_retention = []
        for task_idx in range(num_tasks):
            initial = all_task_accuracies[task_idx][task_idx]
            final = all_task_accuracies[-1][task_idx]
            retention = final / initial if initial > 0 else 0.0
            task_retention.append(retention)
        
        total_time = sum(task_times) if task_times else 1.0
        learning_efficiency = average_accuracy / (total_time / 60.0)
        
        avg_retention = np.mean(task_retention) if task_retention else 0.0
        stability_plasticity_ratio = avg_retention * average_accuracy
        
        return BenchmarkMetrics(
            average_accuracy=average_accuracy,
            final_accuracy=final_accuracies[-1] if final_accuracies else 0.0,
            forgetting_measure=forgetting_measure,
            forward_transfer=forward_transfer,
            task_accuracies=final_accuracies,
            task_learning_times=task_times,
            task_retention=task_retention,
            learning_efficiency=learning_efficiency,
            stability_plasticity_ratio=stability_plasticity_ratio
        )
    
    def _save_result(self, result: BenchmarkResult):
        """Save result to JSON."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.results_dir / f"{result.dataset_name}_{result.method}_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
        
        self.logger.info(f"Results saved to: {output_file}")
    
    def __del__(self):
        """Cleanup."""
        if self.writer:
            self.writer.close()


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Industry-Standard Continual Learning Benchmarks')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--dataset', type=str, 
                       choices=['mnist', 'fashion_mnist', 'cifar10', 'cifar100', 'svhn', 'omniglot', 'tiny_imagenet'],
                       default='mnist', help='Dataset to benchmark')
    parser.add_argument('--strategy', type=str, 
                       choices=['naive', 'replay', 'ewc', 'multihead', 'optimized', 'advanced', 'der++'],
                       help='Strategy to test (der++ is the SOTA baseline)')
    
    args = parser.parse_args()
    
    print("Industry-Standard Continual Learning Benchmarks")
    print("="*80)
    print("Following published best practices and literature standards")
    if args.strategy == 'der++':
        print("Running DER++ (Buzzega et al. 2020) - SOTA baseline")
    
    suite = IndustryStandardBenchmarkSuite(config_path=args.config)
    result = suite.run_benchmark(args.dataset, strategy=args.strategy)
    
    if result.success:
        print(f"\n✅ Benchmark complete: {result.success_level} ({result.overall_score:.2f})")
    else:
        print(f"\n❌ Benchmark failed: {result.error_message}")


if __name__ == '__main__':
    main()
