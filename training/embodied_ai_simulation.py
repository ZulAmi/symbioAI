"""
Embodied AI Simulation System

Agents that learn through interaction in simulated environments with:
- Physics-aware world models
- Tool use and manipulation learning
- Spatial reasoning and navigation
- Sensorimotor grounding of concepts

This bridges pure language AI to physical understanding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import random
from collections import deque
import logging


# ============================================================================
# Core Data Structures
# ============================================================================

class ActionType(Enum):
    """Types of actions in the embodied environment."""
    MOVE_FORWARD = "move_forward"
    MOVE_BACKWARD = "move_backward"
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"
    GRASP = "grasp"
    RELEASE = "release"
    PUSH = "push"
    PULL = "pull"
    LIFT = "lift"
    PLACE = "place"
    USE_TOOL = "use_tool"
    NAVIGATE_TO = "navigate_to"


class SensorModality(Enum):
    """Sensor modalities for perception."""
    VISION = "vision"
    DEPTH = "depth"
    PROPRIOCEPTION = "proprioception"
    TOUCH = "touch"
    FORCE = "force"
    POSITION = "position"
    ORIENTATION = "orientation"


class ObjectProperty(Enum):
    """Physical properties of objects."""
    MASS = "mass"
    FRICTION = "friction"
    ELASTICITY = "elasticity"
    GRASPABLE = "graspable"
    MOVABLE = "movable"
    RIGID = "rigid"
    SOFT = "soft"


@dataclass
class PhysicsState:
    """Complete physics state of the environment."""
    positions: Dict[str, np.ndarray]  # object_id -> [x, y, z]
    velocities: Dict[str, np.ndarray]  # object_id -> [vx, vy, vz]
    orientations: Dict[str, np.ndarray]  # object_id -> quaternion
    forces: Dict[str, np.ndarray]  # object_id -> [fx, fy, fz]
    contacts: List[Tuple[str, str]]  # pairs of objects in contact
    timestamp: float


@dataclass
class SensoryInput:
    """Multi-modal sensory input from the environment."""
    vision: Optional[torch.Tensor] = None  # RGB image (C, H, W)
    depth: Optional[torch.Tensor] = None  # Depth map (1, H, W)
    proprioception: Optional[torch.Tensor] = None  # Joint angles/positions
    touch: Optional[torch.Tensor] = None  # Touch sensors
    force: Optional[torch.Tensor] = None  # Force sensors
    position: Optional[torch.Tensor] = None  # Agent position [x, y, z]
    orientation: Optional[torch.Tensor] = None  # Agent orientation (quaternion)


@dataclass
class Action:
    """Action to be executed in the environment."""
    action_type: ActionType
    parameters: Dict[str, Any]  # Action-specific parameters
    duration: float = 0.1  # Time duration for action


@dataclass
class ToolUseSkill:
    """Learned skill for using a specific tool."""
    tool_name: str
    affordances: List[str]  # What the tool can do
    prerequisites: List[str]  # Required conditions
    action_sequence: List[Action]  # Sequence of actions
    success_rate: float
    learning_iterations: int


@dataclass
class SpatialMap:
    """Spatial map of the environment."""
    occupancy_grid: np.ndarray  # 3D grid (occupied/free)
    object_locations: Dict[str, np.ndarray]  # object_id -> position
    navigation_graph: Dict[str, List[str]]  # Location -> reachable locations
    explored_area: float  # Percentage explored


# ============================================================================
# Physics-Aware World Model
# ============================================================================

class PhysicsEngine:
    """
    Simplified physics engine for simulation.
    Models forces, collisions, gravity, friction.
    """
    
    def __init__(self, gravity: float = -9.81, dt: float = 0.01):
        self.gravity = gravity
        self.dt = dt
        self.objects: Dict[str, Dict[str, Any]] = {}
        self.logger = logging.getLogger(__name__)
    
    def add_object(
        self,
        object_id: str,
        position: np.ndarray,
        mass: float,
        friction: float = 0.5,
        elasticity: float = 0.3,
        is_static: bool = False
    ):
        """Add an object to the physics simulation."""
        self.objects[object_id] = {
            'position': position.copy(),
            'velocity': np.zeros(3),
            'orientation': np.array([1.0, 0.0, 0.0, 0.0]),  # Quaternion
            'mass': mass,
            'friction': friction,
            'elasticity': elasticity,
            'is_static': is_static,
            'forces': np.zeros(3),
            'contacts': []
        }
    
    def apply_force(self, object_id: str, force: np.ndarray):
        """Apply force to an object."""
        if object_id in self.objects:
            self.objects[object_id]['forces'] += force
    
    def step(self) -> PhysicsState:
        """Simulate one physics step."""
        # Apply gravity
        for obj_id, obj in self.objects.items():
            if not obj['is_static']:
                obj['forces'][2] += obj['mass'] * self.gravity
        
        # Update velocities and positions
        for obj_id, obj in self.objects.items():
            if not obj['is_static']:
                # F = ma -> a = F/m
                acceleration = obj['forces'] / obj['mass']
                obj['velocity'] += acceleration * self.dt
                
                # Apply friction
                friction_force = -obj['velocity'] * obj['friction']
                obj['velocity'] += friction_force * self.dt
                
                # Update position
                obj['position'] += obj['velocity'] * self.dt
                
                # Reset forces
                obj['forces'] = np.zeros(3)
        
        # Detect collisions (simplified)
        contacts = self._detect_collisions()
        
        # Create physics state
        return PhysicsState(
            positions={oid: obj['position'].copy() for oid, obj in self.objects.items()},
            velocities={oid: obj['velocity'].copy() for oid, obj in self.objects.items()},
            orientations={oid: obj['orientation'].copy() for oid, obj in self.objects.items()},
            forces={oid: obj['forces'].copy() for oid, obj in self.objects.items()},
            contacts=contacts,
            timestamp=0.0
        )
    
    def _detect_collisions(self) -> List[Tuple[str, str]]:
        """Detect collisions between objects (simplified sphere collision)."""
        contacts = []
        obj_ids = list(self.objects.keys())
        
        for i, id1 in enumerate(obj_ids):
            for id2 in obj_ids[i+1:]:
                pos1 = self.objects[id1]['position']
                pos2 = self.objects[id2]['position']
                distance = np.linalg.norm(pos1 - pos2)
                
                # Simplified: assume radius of 0.5 for all objects
                if distance < 1.0:
                    contacts.append((id1, id2))
        
        return contacts


class WorldModelNetwork(nn.Module):
    """
    Neural network that predicts future states given actions.
    Learns physics implicitly from data.
    """
    
    def __init__(
        self,
        state_dim: int = 128,
        action_dim: int = 32,
        hidden_dim: int = 256,
        num_layers: int = 4
    ):
        super().__init__()
        
        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Action encoder
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2)
        )
        
        # Dynamics model (predicts next state)
        layers = []
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim)
            ])
        self.dynamics = nn.Sequential(*layers)
        
        # State decoder
        self.state_decoder = nn.Linear(hidden_dim, state_dim)
        
        # Reward predictor
        self.reward_predictor = nn.Linear(hidden_dim, 1)
    
    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict next state and reward."""
        # Encode
        state_enc = self.state_encoder(state)
        action_enc = self.action_encoder(action)
        
        # Combine and predict dynamics
        combined = torch.cat([state_enc, action_enc], dim=-1)
        dynamics_out = self.dynamics(combined)
        
        # Decode
        next_state = self.state_decoder(dynamics_out)
        reward = self.reward_predictor(dynamics_out)
        
        return next_state, reward
    
    def rollout(
        self,
        initial_state: torch.Tensor,
        actions: List[torch.Tensor],
        horizon: int = 10
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Rollout the world model for planning."""
        states = [initial_state]
        rewards = []
        
        state = initial_state
        for action in actions[:horizon]:
            next_state, reward = self.forward(state, action)
            states.append(next_state)
            rewards.append(reward)
            state = next_state
        
        return states, rewards


# ============================================================================
# Sensorimotor System
# ============================================================================

class SensorimotorEncoder(nn.Module):
    """
    Encodes multi-modal sensory inputs into a unified representation.
    Grounds concepts in sensorimotor experience.
    """
    
    def __init__(
        self,
        vision_shape: Tuple[int, int, int] = (3, 64, 64),
        depth_shape: Tuple[int, int, int] = (1, 64, 64),
        proprioception_dim: int = 12,
        touch_dim: int = 8,
        embedding_dim: int = 256
    ):
        super().__init__()
        
        # Vision encoder (CNN)
        self.vision_encoder = nn.Sequential(
            nn.Conv2d(vision_shape[0], 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, embedding_dim)
        )
        
        # Depth encoder
        self.depth_encoder = nn.Sequential(
            nn.Conv2d(depth_shape[0], 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, embedding_dim // 2)
        )
        
        # Proprioception encoder
        self.proprio_encoder = nn.Sequential(
            nn.Linear(proprioception_dim, 64),
            nn.ReLU(),
            nn.Linear(64, embedding_dim // 4)
        )
        
        # Touch encoder
        self.touch_encoder = nn.Sequential(
            nn.Linear(touch_dim, 32),
            nn.ReLU(),
            nn.Linear(32, embedding_dim // 4)
        )
        
        # Fusion layer
        total_dim = embedding_dim + embedding_dim // 2 + embedding_dim // 4 + embedding_dim // 4
        self.fusion = nn.Sequential(
            nn.Linear(total_dim, embedding_dim * 2),
            nn.ReLU(),
            nn.LayerNorm(embedding_dim * 2),
            nn.Linear(embedding_dim * 2, embedding_dim)
        )
    
    def forward(self, sensory_input: SensoryInput) -> torch.Tensor:
        """Encode multi-modal sensory input."""
        embeddings = []
        
        if sensory_input.vision is not None:
            vision_emb = self.vision_encoder(sensory_input.vision)
            embeddings.append(vision_emb)
        
        if sensory_input.depth is not None:
            depth_emb = self.depth_encoder(sensory_input.depth)
            embeddings.append(depth_emb)
        
        if sensory_input.proprioception is not None:
            proprio_emb = self.proprio_encoder(sensory_input.proprioception)
            embeddings.append(proprio_emb)
        
        if sensory_input.touch is not None:
            touch_emb = self.touch_encoder(sensory_input.touch)
            embeddings.append(touch_emb)
        
        # Concatenate all embeddings
        combined = torch.cat(embeddings, dim=-1)
        
        # Fuse into unified representation
        fused = self.fusion(combined)
        
        return fused


class ConceptGrounder(nn.Module):
    """
    Grounds abstract concepts in sensorimotor experience.
    Maps language to physical affordances.
    """
    
    def __init__(
        self,
        sensorimotor_dim: int = 256,
        language_dim: int = 512,
        concept_dim: int = 128,
        num_concepts: int = 100
    ):
        super().__init__()
        
        # Sensorimotor to concept
        self.sensorimotor_to_concept = nn.Sequential(
            nn.Linear(sensorimotor_dim, concept_dim * 2),
            nn.ReLU(),
            nn.Linear(concept_dim * 2, concept_dim)
        )
        
        # Language to concept
        self.language_to_concept = nn.Sequential(
            nn.Linear(language_dim, concept_dim * 2),
            nn.ReLU(),
            nn.Linear(concept_dim * 2, concept_dim)
        )
        
        # Concept dictionary (learnable)
        self.concept_embeddings = nn.Parameter(torch.randn(num_concepts, concept_dim))
        
        # Grounding score predictor
        self.grounding_scorer = nn.Sequential(
            nn.Linear(concept_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def ground_concept(
        self,
        language_input: torch.Tensor,
        sensorimotor_input: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Ground a language concept in sensorimotor experience."""
        # Map to concept space
        language_concept = self.language_to_concept(language_input)
        sensorimotor_concept = self.sensorimotor_to_concept(sensorimotor_input)
        
        # Compute grounding score
        combined = torch.cat([language_concept, sensorimotor_concept], dim=-1)
        grounding_score = self.grounding_scorer(combined)
        
        # Find closest concept in dictionary
        distances = torch.cdist(language_concept, self.concept_embeddings)
        closest_concept_idx = torch.argmin(distances, dim=-1)
        
        return closest_concept_idx, grounding_score
    
    def retrieve_affordances(
        self,
        concept_idx: torch.Tensor
    ) -> torch.Tensor:
        """Retrieve affordances for a grounded concept."""
        # Get concept embedding
        concept_emb = self.concept_embeddings[concept_idx]
        return concept_emb


# ============================================================================
# Spatial Reasoning & Navigation
# ============================================================================

class SpatialReasoningModule(nn.Module):
    """
    Spatial reasoning for navigation and scene understanding.
    Builds cognitive maps and plans paths.
    """
    
    def __init__(
        self,
        feature_dim: int = 256,
        map_size: int = 64,
        num_layers: int = 4
    ):
        super().__init__()
        
        self.map_size = map_size
        
        # Spatial memory (2D grid)
        self.spatial_memory = nn.Parameter(torch.zeros(1, feature_dim, map_size, map_size))
        
        # Feature extractor for current observation
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(feature_dim, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, feature_dim, kernel_size=3, padding=1)
        )
        
        # Spatial transformer network
        self.spatial_transformer = nn.Sequential(
            *[self._make_conv_block(feature_dim) for _ in range(num_layers)]
        )
        
        # Path planner
        self.path_planner = nn.Sequential(
            nn.Conv2d(feature_dim, 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 4, kernel_size=1)  # 4 directions: up, down, left, right
        )
    
    def _make_conv_block(self, channels: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.LayerNorm([channels, self.map_size, self.map_size])
        )
    
    def update_map(
        self,
        observation: torch.Tensor,
        agent_position: Tuple[int, int]
    ):
        """Update spatial map with new observation."""
        # Extract features from observation
        features = self.feature_extractor(observation)
        
        # Update spatial memory at agent position
        x, y = agent_position
        self.spatial_memory[:, :, x:x+features.shape[2], y:y+features.shape[3]] = features
    
    def plan_path(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int]
    ) -> List[Tuple[int, int]]:
        """Plan path from start to goal."""
        # Transform spatial memory
        transformed = self.spatial_transformer(self.spatial_memory)
        
        # Get direction probabilities
        direction_probs = F.softmax(self.path_planner(transformed), dim=1)
        
        # A* search using learned heuristic (simplified)
        path = self._astar_search(direction_probs, start, goal)
        
        return path
    
    def _astar_search(
        self,
        direction_probs: torch.Tensor,
        start: Tuple[int, int],
        goal: Tuple[int, int]
    ) -> List[Tuple[int, int]]:
        """Simplified A* search."""
        # For demonstration, return straight line path
        path = []
        current = start
        
        while current != goal:
            x, y = current
            gx, gy = goal
            
            # Move towards goal
            if x < gx:
                current = (x + 1, y)
            elif x > gx:
                current = (x - 1, y)
            elif y < gy:
                current = (x, y + 1)
            elif y > gy:
                current = (x, y - 1)
            
            path.append(current)
            
            if len(path) > 100:  # Prevent infinite loops
                break
        
        return path


# ============================================================================
# Tool Use & Manipulation Learning
# ============================================================================

class ToolUseLearner:
    """
    Learns to use tools through trial and error.
    Discovers affordances and action sequences.
    """
    
    def __init__(
        self,
        state_dim: int = 256,
        action_dim: int = 32,
        num_tools: int = 10
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_tools = num_tools
        
        # Tool affordance network
        self.affordance_network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_tools),
            nn.Sigmoid()
        )
        
        # Tool use policy (per tool)
        self.tool_policies: Dict[str, nn.Module] = {}
        
        # Learned skills
        self.skills: Dict[str, ToolUseSkill] = {}
        
        # Experience buffer
        self.experience_buffer: deque = deque(maxlen=10000)
    
    def detect_affordances(
        self,
        state: torch.Tensor,
        available_tools: List[str]
    ) -> Dict[str, float]:
        """Detect which tools afford which actions in current state."""
        # Get affordance scores
        affordance_scores = self.affordance_network(state)
        
        # Map to tool names
        affordances = {}
        for i, tool_name in enumerate(available_tools[:self.num_tools]):
            affordances[tool_name] = affordance_scores[i].item()
        
        return affordances
    
    def learn_tool_use(
        self,
        tool_name: str,
        state_sequence: List[torch.Tensor],
        action_sequence: List[Action],
        success: bool
    ):
        """Learn from a tool use episode."""
        # Store experience
        self.experience_buffer.append({
            'tool': tool_name,
            'states': state_sequence,
            'actions': action_sequence,
            'success': success
        })
        
        # Update skill if it exists
        if tool_name in self.skills:
            skill = self.skills[tool_name]
            skill.learning_iterations += 1
            
            # Update success rate
            alpha = 0.1  # Learning rate
            skill.success_rate = (1 - alpha) * skill.success_rate + alpha * (1.0 if success else 0.0)
            
            # Update action sequence if successful
            if success and random.random() < 0.3:
                skill.action_sequence = action_sequence
        else:
            # Create new skill
            self.skills[tool_name] = ToolUseSkill(
                tool_name=tool_name,
                affordances=["manipulate", "interact"],
                prerequisites=[],
                action_sequence=action_sequence,
                success_rate=1.0 if success else 0.0,
                learning_iterations=1
            )
    
    def get_tool_policy(
        self,
        tool_name: str,
        state: torch.Tensor
    ) -> Action:
        """Get action for using a specific tool."""
        if tool_name not in self.tool_policies:
            # Create new policy for this tool
            self.tool_policies[tool_name] = nn.Sequential(
                nn.Linear(self.state_dim, 128),
                nn.ReLU(),
                nn.Linear(128, self.action_dim),
                nn.Tanh()
            )
        
        # Get action from policy
        action_params = self.tool_policies[tool_name](state)
        
        # Convert to Action object (simplified)
        action = Action(
            action_type=ActionType.USE_TOOL,
            parameters={'tool': tool_name, 'params': action_params.tolist()},
            duration=0.1
        )
        
        return action


class ManipulationController(nn.Module):
    """
    Neural controller for object manipulation.
    Learns inverse kinematics and grasp policies.
    """
    
    def __init__(
        self,
        observation_dim: int = 256,
        action_dim: int = 7,  # 6 DOF + gripper
        hidden_dim: int = 128
    ):
        super().__init__()
        
        # Visual features to manipulation action
        self.policy_network = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
        
        # Grasp quality predictor
        self.grasp_predictor = nn.Sequential(
            nn.Linear(observation_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        observation: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute manipulation action and grasp quality."""
        action = self.policy_network(observation)
        grasp_quality = self.grasp_predictor(observation)
        
        return action, grasp_quality


# ============================================================================
# Embodied Agent
# ============================================================================

class EmbodiedAgent:
    """
    Complete embodied agent that integrates all components:
    - Sensorimotor perception
    - World model
    - Spatial reasoning
    - Tool use learning
    - Manipulation control
    """
    
    def __init__(
        self,
        vision_shape: Tuple[int, int, int] = (3, 64, 64),
        state_dim: int = 256,
        action_dim: int = 32,
        use_physics: bool = True
    ):
        # Core components
        self.sensorimotor_encoder = SensorimotorEncoder(
            vision_shape=vision_shape,
            embedding_dim=state_dim
        )
        
        self.world_model = WorldModelNetwork(
            state_dim=state_dim,
            action_dim=action_dim
        )
        
        self.concept_grounder = ConceptGrounder(
            sensorimotor_dim=state_dim
        )
        
        self.spatial_reasoner = SpatialReasoningModule(
            feature_dim=state_dim
        )
        
        self.tool_learner = ToolUseLearner(
            state_dim=state_dim,
            action_dim=action_dim
        )
        
        self.manipulation_controller = ManipulationController(
            observation_dim=state_dim,
            action_dim=7
        )
        
        # Physics engine (optional)
        self.physics_engine = PhysicsEngine() if use_physics else None
        
        # State
        self.current_state: Optional[torch.Tensor] = None
        self.position = np.array([0.0, 0.0, 0.0])
        self.orientation = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Metrics
        self.total_steps = 0
        self.successful_manipulations = 0
        self.total_manipulations = 0
        
        self.logger = logging.getLogger(__name__)
    
    def perceive(self, sensory_input: SensoryInput) -> torch.Tensor:
        """Process multi-modal sensory input."""
        # Encode sensory input
        state = self.sensorimotor_encoder(sensory_input)
        
        self.current_state = state
        return state
    
    def plan_action(
        self,
        goal: Optional[str] = None,
        available_tools: Optional[List[str]] = None
    ) -> Action:
        """Plan next action based on current state and goal."""
        if self.current_state is None:
            raise ValueError("No current state - call perceive() first")
        
        # Check if tool use is beneficial
        if available_tools:
            affordances = self.tool_learner.detect_affordances(
                self.current_state,
                available_tools
            )
            
            # Select best tool
            best_tool = max(affordances, key=affordances.get)
            if affordances[best_tool] > 0.5:
                return self.tool_learner.get_tool_policy(best_tool, self.current_state)
        
        # Otherwise, use manipulation controller
        action_params, grasp_quality = self.manipulation_controller(self.current_state)
        
        # Determine action type based on grasp quality
        if grasp_quality > 0.7:
            action_type = ActionType.GRASP
        else:
            action_type = ActionType.MOVE_FORWARD
        
        return Action(
            action_type=action_type,
            parameters={'params': action_params.tolist()},
            duration=0.1
        )
    
    def navigate_to(
        self,
        goal_position: Tuple[int, int]
    ) -> List[Tuple[int, int]]:
        """Navigate to goal position using spatial reasoning."""
        current_pos = (int(self.position[0]), int(self.position[1]))
        path = self.spatial_reasoner.plan_path(current_pos, goal_position)
        
        return path
    
    def ground_language_concept(
        self,
        language_input: torch.Tensor
    ) -> Tuple[torch.Tensor, float]:
        """Ground a language concept in sensorimotor experience."""
        if self.current_state is None:
            raise ValueError("No current state")
        
        concept_idx, grounding_score = self.concept_grounder.ground_concept(
            language_input,
            self.current_state
        )
        
        return concept_idx, grounding_score.item()
    
    def learn_from_experience(
        self,
        state_sequence: List[torch.Tensor],
        action_sequence: List[Action],
        reward_sequence: List[float],
        tool_used: Optional[str] = None
    ):
        """Learn from interaction experience."""
        # Learn tool use if applicable
        if tool_used:
            success = sum(reward_sequence) > 0
            self.tool_learner.learn_tool_use(
                tool_used,
                state_sequence,
                action_sequence,
                success
            )
        
        # Update world model (simplified)
        # In practice, you'd train the world model with backprop
        
        self.total_steps += len(action_sequence)
        
        self.logger.info(f"Learned from {len(action_sequence)} steps, total steps: {self.total_steps}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get agent statistics."""
        return {
            'total_steps': self.total_steps,
            'successful_manipulations': self.successful_manipulations,
            'total_manipulations': self.total_manipulations,
            'manipulation_success_rate': (
                self.successful_manipulations / max(1, self.total_manipulations)
            ),
            'tools_learned': len(self.tool_learner.skills),
            'tool_skills': {
                name: {
                    'success_rate': skill.success_rate,
                    'iterations': skill.learning_iterations
                }
                for name, skill in self.tool_learner.skills.items()
            }
        }


# ============================================================================
# Training & Simulation Environment
# ============================================================================

class SimulationEnvironment:
    """
    Simulated environment for embodied learning.
    Provides physics, objects, and tasks.
    """
    
    def __init__(
        self,
        grid_size: Tuple[int, int, int] = (10, 10, 5),
        num_objects: int = 5,
        num_tools: int = 3
    ):
        self.grid_size = grid_size
        self.physics = PhysicsEngine()
        
        # Spawn objects
        self.objects: Dict[str, Dict[str, Any]] = {}
        for i in range(num_objects):
            obj_id = f"object_{i}"
            position = np.array([
                random.uniform(0, grid_size[0]),
                random.uniform(0, grid_size[1]),
                0.5
            ])
            self.physics.add_object(
                obj_id,
                position,
                mass=random.uniform(0.1, 2.0),
                friction=random.uniform(0.3, 0.8)
            )
            self.objects[obj_id] = {
                'type': 'box',
                'size': random.uniform(0.1, 0.5)
            }
        
        # Spawn tools
        self.tools: List[str] = []
        for i in range(num_tools):
            tool_name = f"tool_{i}"
            self.tools.append(tool_name)
            position = np.array([
                random.uniform(0, grid_size[0]),
                random.uniform(0, grid_size[1]),
                0.5
            ])
            self.physics.add_object(
                tool_name,
                position,
                mass=0.5,
                friction=0.4
            )
        
        # Agent
        self.agent_position = np.array([grid_size[0] / 2, grid_size[1] / 2, 0.5])
        self.physics.add_object(
            "agent",
            self.agent_position,
            mass=1.0,
            friction=0.5
        )
    
    def step(self, action: Action) -> Tuple[SensoryInput, float, bool]:
        """Execute action and return observation, reward, done."""
        # Execute action in physics
        if action.action_type == ActionType.MOVE_FORWARD:
            force = np.array([1.0, 0.0, 0.0])
            self.physics.apply_force("agent", force)
        elif action.action_type == ActionType.TURN_LEFT:
            # Apply rotational force (simplified)
            pass
        # ... other action types
        
        # Step physics
        physics_state = self.physics.step()
        
        # Update agent position
        self.agent_position = physics_state.positions["agent"]
        
        # Generate sensory input
        sensory_input = self._generate_sensory_input()
        
        # Compute reward (task-dependent)
        reward = self._compute_reward(action, physics_state)
        
        # Check if episode is done
        done = False  # Implement termination condition
        
        return sensory_input, reward, done
    
    def _generate_sensory_input(self) -> SensoryInput:
        """Generate simulated sensory input."""
        # Vision: simple rendered view (simplified)
        vision = torch.randn(1, 3, 64, 64)
        
        # Depth: distance map
        depth = torch.randn(1, 1, 64, 64)
        
        # Proprioception: joint angles (simplified)
        proprioception = torch.randn(1, 12)
        
        # Touch: contact sensors
        touch = torch.zeros(1, 8)
        
        return SensoryInput(
            vision=vision,
            depth=depth,
            proprioception=proprioception,
            touch=touch,
            position=torch.tensor(self.agent_position).unsqueeze(0),
            orientation=torch.tensor([1.0, 0.0, 0.0, 0.0]).unsqueeze(0)
        )
    
    def _compute_reward(self, action: Action, physics_state: PhysicsState) -> float:
        """Compute reward for the action."""
        # Simple reward: encourage exploration and interaction
        reward = 0.01  # Base reward for survival
        
        # Reward for moving
        if action.action_type in [ActionType.MOVE_FORWARD, ActionType.MOVE_BACKWARD]:
            reward += 0.05
        
        # Reward for using tools
        if action.action_type == ActionType.USE_TOOL:
            reward += 0.1
        
        # Reward for successful grasps
        if action.action_type == ActionType.GRASP:
            # Check if near object
            for obj_id, pos in physics_state.positions.items():
                if obj_id != "agent":
                    dist = np.linalg.norm(self.agent_position - pos)
                    if dist < 1.0:
                        reward += 0.5
        
        return reward
    
    def reset(self) -> SensoryInput:
        """Reset environment to initial state."""
        # Reset physics
        self.physics = PhysicsEngine()
        
        # Re-spawn everything
        self.__init__(self.grid_size, len(self.objects), len(self.tools))
        
        return self._generate_sensory_input()


# ============================================================================
# Factory Function
# ============================================================================

def create_embodied_agent(
    vision_shape: Tuple[int, int, int] = (3, 64, 64),
    state_dim: int = 256,
    action_dim: int = 32,
    use_physics: bool = True
) -> EmbodiedAgent:
    """Create an embodied AI agent."""
    return EmbodiedAgent(
        vision_shape=vision_shape,
        state_dim=state_dim,
        action_dim=action_dim,
        use_physics=use_physics
    )


def create_simulation_environment(
    grid_size: Tuple[int, int, int] = (10, 10, 5),
    num_objects: int = 5,
    num_tools: int = 3
) -> SimulationEnvironment:
    """Create a simulation environment."""
    return SimulationEnvironment(
        grid_size=grid_size,
        num_objects=num_objects,
        num_tools=num_tools
    )
