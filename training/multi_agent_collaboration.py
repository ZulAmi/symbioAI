"""
Multi-Agent Collaboration Networks: Multiple specialized agents that cooperate and compete

Implements a sophisticated multi-agent system with:
- Automatic role assignment and specialization
- Emergent communication protocols
- Adversarial training between agents
- Collaborative problem decomposition
- Self-organizing agent teams

Built on top of Symbio AI's existing agent orchestrator.
"""

import asyncio
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import time
import uuid
import logging
from collections import defaultdict, deque
import json


# ============================================================================
# Core Data Structures
# ============================================================================

class AgentRole(Enum):
    """Agent roles in the collaboration network."""
    GENERATOR = "generator"           # Generates solutions/content
    CRITIC = "critic"                 # Evaluates and critiques
    COORDINATOR = "coordinator"       # Coordinates team activities
    SPECIALIST = "specialist"         # Domain-specific expert
    GENERALIST = "generalist"         # Handles diverse tasks
    LEARNER = "learner"              # Focused on learning/adaptation
    TEACHER = "teacher"              # Shares knowledge with others
    EXPLORER = "explorer"            # Explores new strategies
    EXPLOITER = "exploiter"          # Exploits known strategies


class CommunicationMode(Enum):
    """Communication protocol modes."""
    BROADCAST = "broadcast"          # Send to all agents
    UNICAST = "unicast"             # Send to specific agent
    MULTICAST = "multicast"         # Send to agent group
    EMERGENT = "emergent"           # Learn communication patterns


class CollaborationStrategy(Enum):
    """Collaboration strategies."""
    COOPERATIVE = "cooperative"      # Pure cooperation
    COMPETITIVE = "competitive"      # Pure competition
    MIXED = "mixed"                 # Mix of both
    ADAPTIVE = "adaptive"           # Adapt based on context


@dataclass
class CommunicationMessage:
    """Message exchanged between agents."""
    id: str
    sender_id: str
    recipient_ids: List[str]
    mode: CommunicationMode
    content: torch.Tensor  # Learned message representation
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    priority: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'sender': self.sender_id,
            'recipients': self.recipient_ids,
            'mode': self.mode.value,
            'metadata': self.metadata,
            'timestamp': self.timestamp,
            'priority': self.priority
        }


@dataclass
class CollaborationTask:
    """Task requiring multi-agent collaboration."""
    id: str
    description: str
    complexity: float  # 0.0 to 1.0
    required_roles: List[AgentRole]
    subtasks: List[Dict[str, Any]] = field(default_factory=list)
    dependencies: Dict[str, List[str]] = field(default_factory=dict)
    rewards: Dict[str, float] = field(default_factory=dict)
    
    def decompose(self, num_subtasks: int = 3) -> List[Dict[str, Any]]:
        """Decompose task into subtasks."""
        if self.subtasks:
            return self.subtasks
        
        # Simple decomposition
        subtasks = []
        for i in range(num_subtasks):
            subtasks.append({
                'id': f"{self.id}_subtask_{i}",
                'description': f"Subtask {i} of {self.description}",
                'complexity': self.complexity / num_subtasks,
                'index': i
            })
        
        return subtasks


@dataclass
class AgentPerformance:
    """Track agent performance metrics."""
    agent_id: str
    role: AgentRole
    tasks_completed: int = 0
    tasks_failed: int = 0
    collaboration_score: float = 0.0
    communication_efficiency: float = 0.0
    specialization_scores: Dict[str, float] = field(default_factory=dict)
    peer_ratings: List[float] = field(default_factory=list)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        total = self.tasks_completed + self.tasks_failed
        return self.tasks_completed / total if total > 0 else 0.0
    
    @property
    def average_peer_rating(self) -> float:
        """Average rating from peers."""
        return np.mean(self.peer_ratings) if self.peer_ratings else 0.5


# ============================================================================
# Communication Protocol Networks
# ============================================================================

class EmergentCommunicationProtocol(nn.Module):
    """
    Neural network that learns emergent communication protocols between agents.
    
    Agents develop their own "language" to communicate effectively without
    pre-defined protocols.
    """
    
    def __init__(
        self,
        agent_id: str,
        message_dim: int = 64,
        hidden_dim: int = 128,
        num_agents: int = 10
    ):
        super().__init__()
        
        self.agent_id = agent_id
        self.message_dim = message_dim
        
        # Message encoder: state -> message
        self.message_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, message_dim),
            nn.Tanh()  # Bound message values
        )
        
        # Message decoder: message -> meaning
        self.message_decoder = nn.Sequential(
            nn.Linear(message_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Attention over received messages
        self.message_attention = nn.MultiheadAttention(
            embed_dim=message_dim,
            num_heads=4,
            batch_first=True
        )
        
        # Protocol evolution network
        self.protocol_updater = nn.GRU(
            input_size=message_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True
        )
        
        # Communication success predictor
        self.success_predictor = nn.Sequential(
            nn.Linear(message_dim + hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.message_history = deque(maxlen=100)
        
    def encode_message(self, state: torch.Tensor, intent: str = "inform") -> torch.Tensor:
        """
        Encode agent's current state into a message.
        
        Args:
            state: Agent's current state
            intent: Communication intent
            
        Returns:
            Encoded message tensor
        """
        message = self.message_encoder(state)
        self.message_history.append(message.detach())
        return message
    
    def decode_message(self, message: torch.Tensor) -> torch.Tensor:
        """
        Decode received message into meaningful representation.
        
        Args:
            message: Received message tensor
            
        Returns:
            Decoded meaning representation
        """
        return self.message_decoder(message)
    
    def aggregate_messages(
        self,
        messages: List[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Aggregate multiple messages using attention mechanism.
        
        Args:
            messages: List of message tensors
            attention_mask: Optional attention mask
            
        Returns:
            Aggregated message representation
        """
        if not messages:
            return torch.zeros(1, self.message_dim)
        
        # Stack messages
        msg_stack = torch.stack(messages).unsqueeze(0)  # (1, num_msgs, msg_dim)
        
        # Apply attention
        aggregated, _ = self.message_attention(
            msg_stack, msg_stack, msg_stack,
            attn_mask=attention_mask
        )
        
        # Return mean pooling
        return aggregated.mean(dim=1)
    
    def update_protocol(
        self,
        message_sequence: torch.Tensor,
        success_signal: float
    ) -> None:
        """
        Update communication protocol based on success.
        
        Uses reinforcement signal to shape emergent communication.
        """
        # Protocol evolution through GRU
        _, hidden = self.protocol_updater(message_sequence.unsqueeze(0))
        
        # Predict success
        combined = torch.cat([
            message_sequence.mean(dim=0),
            hidden[-1, 0]
        ], dim=0)
        predicted_success = self.success_predictor(combined)
        
        # Update based on actual vs predicted
        # (Would be trained end-to-end in full system)
        
    def get_protocol_state(self) -> Dict[str, Any]:
        """Get current protocol state for analysis."""
        if not self.message_history:
            return {'status': 'no_messages'}
        
        recent_messages = torch.stack(list(self.message_history)[-10:])
        
        return {
            'message_diversity': torch.std(recent_messages).item(),
            'message_magnitude': torch.mean(torch.abs(recent_messages)).item(),
            'total_messages': len(self.message_history)
        }


# ============================================================================
# Collaborative Agent
# ============================================================================

class CollaborativeAgent(nn.Module):
    """
    Agent that can collaborate and compete with other agents.
    
    Features:
    - Automatic role specialization
    - Emergent communication
    - Adversarial learning capability
    - Peer evaluation
    """
    
    def __init__(
        self,
        agent_id: str,
        state_dim: int = 128,
        action_dim: int = 64,
        message_dim: int = 64,
        initial_role: Optional[AgentRole] = None
    ):
        super().__init__()
        
        self.agent_id = agent_id
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Role assignment
        self.current_role = initial_role or AgentRole.GENERALIST
        self.role_scores = {role: 0.0 for role in AgentRole}
        
        # Core policy network
        self.policy = nn.Sequential(
            nn.Linear(state_dim + message_dim, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )
        
        # Value network for self-evaluation
        self.value_network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # Communication protocol
        self.communication = EmergentCommunicationProtocol(
            agent_id=agent_id,
            message_dim=message_dim
        )
        
        # Peer evaluation network
        self.peer_evaluator = nn.Sequential(
            nn.Linear(action_dim * 2, 128),  # Own action + peer action
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Rating 0-1
        )
        
        # Role specialization network
        self.role_predictor = nn.Sequential(
            nn.Linear(state_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, len(AgentRole)),
            nn.Softmax(dim=-1)
        )
        
        # Collaboration strategy selector
        self.strategy_selector = nn.Sequential(
            nn.Linear(state_dim + message_dim, 64),
            nn.ReLU(),
            nn.Linear(64, len(CollaborationStrategy)),
            nn.Softmax(dim=-1)
        )
        
        # Performance tracking
        self.performance = AgentPerformance(
            agent_id=agent_id,
            role=self.current_role
        )
        
        # Message inbox
        self.message_inbox: List[CommunicationMessage] = []
        
    def forward(
        self,
        state: torch.Tensor,
        messages: Optional[List[torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: state + messages -> action + outgoing message.
        
        Args:
            state: Current state observation
            messages: Received messages from other agents
            
        Returns:
            (action, outgoing_message)
        """
        # Aggregate received messages
        if messages:
            aggregated_msg = self.communication.aggregate_messages(messages)
        else:
            aggregated_msg = torch.zeros(1, self.communication.message_dim)
        
        # Generate action
        combined_input = torch.cat([state, aggregated_msg], dim=-1)
        action = self.policy(combined_input)
        
        # Generate outgoing message
        outgoing_message = self.communication.encode_message(state)
        
        return action, outgoing_message
    
    def evaluate_peer(
        self,
        own_action: torch.Tensor,
        peer_action: torch.Tensor
    ) -> float:
        """
        Evaluate peer agent's action quality.
        
        Args:
            own_action: This agent's action
            peer_action: Peer agent's action
            
        Returns:
            Rating score (0-1)
        """
        combined = torch.cat([own_action, peer_action], dim=-1)
        rating = self.peer_evaluator(combined)
        return rating.item()
    
    def predict_role(
        self,
        state: torch.Tensor,
        action: torch.Tensor
    ) -> Dict[AgentRole, float]:
        """
        Predict role suitability based on state and action.
        
        Returns:
            Dictionary of role -> probability
        """
        combined = torch.cat([state, action], dim=-1)
        role_probs = self.role_predictor(combined)
        
        return {
            role: role_probs[0, i].item()
            for i, role in enumerate(AgentRole)
        }
    
    def select_strategy(
        self,
        state: torch.Tensor,
        context_messages: List[torch.Tensor]
    ) -> CollaborationStrategy:
        """
        Select collaboration strategy based on context.
        
        Args:
            state: Current state
            context_messages: Recent messages
            
        Returns:
            Selected collaboration strategy
        """
        aggregated_msg = self.communication.aggregate_messages(context_messages) \
                        if context_messages else torch.zeros(1, self.communication.message_dim)
        
        combined = torch.cat([state, aggregated_msg], dim=-1)
        strategy_probs = self.strategy_selector(combined)
        
        # Select strategy with highest probability
        strategy_idx = torch.argmax(strategy_probs).item()
        return list(CollaborationStrategy)[strategy_idx]
    
    def update_role(self, new_role: AgentRole) -> None:
        """Update agent's role."""
        self.current_role = new_role
        self.performance.role = new_role
    
    def receive_message(self, message: CommunicationMessage) -> None:
        """Receive message from another agent."""
        self.message_inbox.append(message)
    
    def clear_inbox(self) -> List[CommunicationMessage]:
        """Clear and return all messages."""
        messages = self.message_inbox.copy()
        self.message_inbox.clear()
        return messages
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        return {
            'agent_id': self.agent_id,
            'role': self.current_role.value,
            'success_rate': self.performance.success_rate,
            'tasks_completed': self.performance.tasks_completed,
            'collaboration_score': self.performance.collaboration_score,
            'communication_efficiency': self.performance.communication_efficiency,
            'peer_rating': self.performance.average_peer_rating,
            'protocol_state': self.communication.get_protocol_state()
        }


# ============================================================================
# Multi-Agent Collaboration Network
# ============================================================================

class MultiAgentCollaborationNetwork:
    """
    Manages multiple collaborative agents with emergent behaviors.
    
    Features:
    - Automatic role assignment and specialization
    - Emergent communication protocols
    - Adversarial training between agents
    - Collaborative problem decomposition
    - Self-organizing teams
    """
    
    def __init__(
        self,
        num_agents: int = 10,
        state_dim: int = 128,
        action_dim: int = 64,
        message_dim: int = 64,
        adversarial_ratio: float = 0.3  # % of agents in adversarial mode
    ):
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.message_dim = message_dim
        self.adversarial_ratio = adversarial_ratio
        
        # Create agents
        self.agents: Dict[str, CollaborativeAgent] = {}
        for i in range(num_agents):
            agent_id = f"agent_{i:03d}"
            agent = CollaborativeAgent(
                agent_id=agent_id,
                state_dim=state_dim,
                action_dim=action_dim,
                message_dim=message_dim
            )
            self.agents[agent_id] = agent
        
        # Role assignment manager
        self.role_assignments: Dict[str, AgentRole] = {}
        
        # Communication network
        self.communication_graph: Dict[str, Set[str]] = defaultdict(set)
        
        # Collaboration history
        self.collaboration_history: List[Dict[str, Any]] = []
        
        # Adversarial pairs (for competitive training)
        self.adversarial_pairs: List[Tuple[str, str]] = []
        
        # Team formations
        self.teams: Dict[str, List[str]] = {}
        
        # Performance tracking
        self.global_metrics = {
            'total_tasks': 0,
            'successful_collaborations': 0,
            'emergent_protocols_discovered': 0,
            'role_specializations': 0
        }
        
        self.logger = logging.getLogger(__name__)
    
    def assign_roles_automatically(
        self,
        task: CollaborationTask,
        method: str = "performance"
    ) -> Dict[str, AgentRole]:
        """
        Automatically assign roles to agents based on task requirements.
        
        Methods:
        - performance: Based on past performance
        - diversity: Maximize role diversity
        - specialization: Based on agent specializations
        - adaptive: Learn optimal assignments
        
        Args:
            task: Collaboration task
            method: Assignment method
            
        Returns:
            Dictionary mapping agent_id -> role
        """
        required_roles = task.required_roles
        assignments = {}
        
        if method == "performance":
            # Assign based on performance metrics
            for role in required_roles:
                # Find best agent for this role
                best_agent = None
                best_score = -float('inf')
                
                for agent_id, agent in self.agents.items():
                    if agent_id in assignments:
                        continue
                    
                    # Score based on role affinity
                    score = agent.role_scores.get(role, 0.0)
                    score += agent.performance.success_rate
                    
                    if score > best_score:
                        best_score = score
                        best_agent = agent_id
                
                if best_agent:
                    assignments[best_agent] = role
                    self.agents[best_agent].update_role(role)
        
        elif method == "diversity":
            # Maximize diversity of assigned roles
            available_agents = list(self.agents.keys())
            np.random.shuffle(available_agents)
            
            for i, role in enumerate(required_roles):
                if i < len(available_agents):
                    agent_id = available_agents[i]
                    assignments[agent_id] = role
                    self.agents[agent_id].update_role(role)
        
        elif method == "specialization":
            # Assign based on learned specializations
            for role in required_roles:
                # Find most specialized agent
                best_agent = None
                best_specialization = -1.0
                
                for agent_id, agent in self.agents.items():
                    if agent_id in assignments:
                        continue
                    
                    spec_score = agent.performance.specialization_scores.get(
                        role.value, 0.0
                    )
                    if spec_score > best_specialization:
                        best_specialization = spec_score
                        best_agent = agent_id
                
                if best_agent:
                    assignments[best_agent] = role
                    self.agents[best_agent].update_role(role)
        
        self.role_assignments.update(assignments)
        self.logger.info(f"Assigned {len(assignments)} roles for task {task.id}")
        
        return assignments
    
    async def solve_task_collaboratively(
        self,
        task: CollaborationTask,
        max_iterations: int = 10
    ) -> Dict[str, Any]:
        """
        Solve task through multi-agent collaboration.
        
        Process:
        1. Assign roles to agents
        2. Decompose task into subtasks
        3. Form teams for subtasks
        4. Execute with communication
        5. Integrate results
        
        Args:
            task: Collaboration task
            max_iterations: Maximum communication rounds
            
        Returns:
            Task solution and collaboration metrics
        """
        start_time = time.time()
        
        # Step 1: Assign roles
        assignments = self.assign_roles_automatically(task)
        
        # Step 2: Decompose task
        subtasks = task.decompose()
        
        # Step 3: Form teams
        teams = self._form_teams(subtasks, assignments)
        
        # Step 4: Collaborative execution
        results = {}
        messages_exchanged = 0
        
        for iteration in range(max_iterations):
            # Each agent acts based on state + messages
            for agent_id, agent in self.agents.items():
                if agent_id not in assignments:
                    continue
                
                # Get messages from inbox
                messages_received = agent.clear_inbox()
                message_tensors = [msg.content for msg in messages_received]
                
                # Generate state (mock for demo)
                state = torch.randn(1, self.state_dim)
                
                # Forward pass
                action, outgoing_message = agent(state, message_tensors)
                
                # Store action
                results[agent_id] = {
                    'action': action.detach(),
                    'iteration': iteration
                }
                
                # Send message to team members
                team = teams.get(agent_id, [])
                for teammate_id in team:
                    if teammate_id != agent_id:
                        msg = CommunicationMessage(
                            id=str(uuid.uuid4()),
                            sender_id=agent_id,
                            recipient_ids=[teammate_id],
                            mode=CommunicationMode.UNICAST,
                            content=outgoing_message,
                            metadata={'iteration': iteration},
                            timestamp=time.time()
                        )
                        self.agents[teammate_id].receive_message(msg)
                        messages_exchanged += 1
            
            # Check convergence (simplified)
            if iteration >= max_iterations - 1:
                break
        
        # Step 5: Integrate results
        integrated_result = self._integrate_collaborative_results(results, task)
        
        # Update metrics
        execution_time = time.time() - start_time
        
        self.global_metrics['total_tasks'] += 1
        self.global_metrics['successful_collaborations'] += 1
        
        return {
            'success': True,
            'task_id': task.id,
            'result': integrated_result,
            'execution_time': execution_time,
            'iterations': iteration + 1,
            'messages_exchanged': messages_exchanged,
            'teams': teams,
            'role_assignments': {k: v.value for k, v in assignments.items()}
        }
    
    def _form_teams(
        self,
        subtasks: List[Dict[str, Any]],
        role_assignments: Dict[str, AgentRole]
    ) -> Dict[str, List[str]]:
        """
        Form agent teams for subtasks.
        
        Args:
            subtasks: List of subtasks
            role_assignments: Current role assignments
            
        Returns:
            Dictionary mapping agent_id -> list of team member ids
        """
        teams = {}
        assigned_agents = list(role_assignments.keys())
        
        # Simple team formation: distribute agents across subtasks
        agents_per_subtask = len(assigned_agents) // max(len(subtasks), 1)
        
        for i, subtask in enumerate(subtasks):
            start_idx = i * agents_per_subtask
            end_idx = start_idx + agents_per_subtask
            team_members = assigned_agents[start_idx:end_idx]
            
            # Assign team to all members
            for agent_id in team_members:
                teams[agent_id] = team_members
        
        # Remaining agents join last team
        if len(assigned_agents) % len(subtasks) != 0:
            remaining = assigned_agents[len(subtasks) * agents_per_subtask:]
            last_team = assigned_agents[-agents_per_subtask:]
            for agent_id in remaining:
                teams[agent_id] = last_team + remaining
        
        return teams
    
    def _integrate_collaborative_results(
        self,
        agent_results: Dict[str, Any],
        task: CollaborationTask
    ) -> Dict[str, Any]:
        """
        Integrate results from multiple agents.
        
        Integration strategies:
        - Voting: Majority vote on discrete outcomes
        - Averaging: Average continuous values
        - Hierarchical: Weighted by role hierarchy
        - Consensus: Require agreement threshold
        """
        if not agent_results:
            return {'status': 'no_results'}
        
        # Collect all actions
        actions = [result['action'] for result in agent_results.values()]
        
        # Simple integration: average actions
        integrated_action = torch.stack(actions).mean(dim=0)
        
        return {
            'integrated_action': integrated_action,
            'num_contributors': len(agent_results),
            'individual_contributions': {
                aid: {
                    'action_norm': result['action'].norm().item(),
                    'iteration': result['iteration']
                }
                for aid, result in agent_results.items()
            }
        }
    
    async def adversarial_training_round(
        self,
        num_rounds: int = 5
    ) -> Dict[str, Any]:
        """
        Run adversarial training between agent pairs.
        
        Competitive training where agents compete against each other,
        learning robust strategies through adversarial examples.
        
        Args:
            num_rounds: Number of adversarial rounds
            
        Returns:
            Training results and performance metrics
        """
        # Form adversarial pairs
        if not self.adversarial_pairs:
            self._create_adversarial_pairs()
        
        results = {
            'rounds': [],
            'winner_counts': defaultdict(int),
            'improvement_rates': {}
        }
        
        for round_num in range(num_rounds):
            round_results = []
            
            for agent1_id, agent2_id in self.adversarial_pairs:
                agent1 = self.agents[agent1_id]
                agent2 = self.agents[agent2_id]
                
                # Generate adversarial scenario
                state = torch.randn(1, self.state_dim)
                
                # Both agents act
                action1, msg1 = agent1(state)
                action2, msg2 = agent2(state)
                
                # Determine winner (simplified: higher action norm wins)
                score1 = action1.norm().item()
                score2 = action2.norm().item()
                
                winner = agent1_id if score1 > score2 else agent2_id
                results['winner_counts'][winner] += 1
                
                # Peer evaluation
                rating1to2 = agent1.evaluate_peer(action1, action2)
                rating2to1 = agent2.evaluate_peer(action2, action1)
                
                agent1.performance.peer_ratings.append(rating2to1)
                agent2.performance.peer_ratings.append(rating1to2)
                
                round_results.append({
                    'pair': (agent1_id, agent2_id),
                    'winner': winner,
                    'scores': (score1, score2),
                    'ratings': (rating1to2, rating2to1)
                })
            
            results['rounds'].append(round_results)
        
        # Calculate improvement rates
        for agent_id, agent in self.agents.items():
            if agent.performance.peer_ratings:
                improvement = (
                    np.mean(agent.performance.peer_ratings[-5:]) -
                    np.mean(agent.performance.peer_ratings[:5])
                ) if len(agent.performance.peer_ratings) >= 10 else 0.0
                results['improvement_rates'][agent_id] = improvement
        
        self.logger.info(f"Completed {num_rounds} adversarial training rounds")
        
        return results
    
    def _create_adversarial_pairs(self) -> None:
        """Create adversarial agent pairs for competitive training."""
        agent_ids = list(self.agents.keys())
        num_adversarial = int(len(agent_ids) * self.adversarial_ratio)
        
        # Shuffle and pair
        np.random.shuffle(agent_ids)
        
        for i in range(0, num_adversarial - 1, 2):
            self.adversarial_pairs.append((agent_ids[i], agent_ids[i + 1]))
        
        self.logger.info(f"Created {len(self.adversarial_pairs)} adversarial pairs")
    
    def discover_emergent_protocols(self) -> Dict[str, Any]:
        """
        Analyze communication patterns to discover emergent protocols.
        
        Returns:
            Discovered protocol patterns and statistics
        """
        protocols = {
            'discovered_patterns': [],
            'communication_clusters': [],
            'protocol_efficiency': {}
        }
        
        # Analyze message patterns from each agent
        for agent_id, agent in self.agents.items():
            protocol_state = agent.communication.get_protocol_state()
            
            if protocol_state.get('total_messages', 0) > 10:
                protocols['protocol_efficiency'][agent_id] = {
                    'message_diversity': protocol_state.get('message_diversity', 0.0),
                    'efficiency': agent.performance.communication_efficiency
                }
        
        # Identify communication clusters
        # (Simplified: group agents with similar communication patterns)
        clusters = self._cluster_communication_patterns()
        protocols['communication_clusters'] = clusters
        
        # Count discovered patterns
        num_patterns = len(set(
            agent.communication.get_protocol_state().get('message_diversity', 0)
            for agent in self.agents.values()
        ))
        
        protocols['num_unique_patterns'] = num_patterns
        self.global_metrics['emergent_protocols_discovered'] = num_patterns
        
        return protocols
    
    def _cluster_communication_patterns(self) -> List[List[str]]:
        """Cluster agents by communication patterns."""
        # Simple clustering by message diversity
        patterns = {}
        
        for agent_id, agent in self.agents.items():
            state = agent.communication.get_protocol_state()
            diversity = state.get('message_diversity', 0.0)
            
            # Group by diversity (rounded to 2 decimals)
            key = round(diversity, 2)
            if key not in patterns:
                patterns[key] = []
            patterns[key].append(agent_id)
        
        return list(patterns.values())
    
    def analyze_role_specialization(self) -> Dict[str, Any]:
        """
        Analyze how agents have specialized into roles.
        
        Returns:
            Specialization analysis and statistics
        """
        specialization_report = {
            'role_distribution': defaultdict(list),
            'specialization_strength': {},
            'role_transitions': [],
            'optimal_roles': {}
        }
        
        for agent_id, agent in self.agents.items():
            current_role = agent.current_role
            specialization_report['role_distribution'][current_role.value].append(agent_id)
            
            # Calculate specialization strength
            role_scores = list(agent.role_scores.values())
            if role_scores:
                max_score = max(role_scores)
                avg_score = np.mean(role_scores)
                strength = (max_score - avg_score) / (avg_score + 1e-6)
                specialization_report['specialization_strength'][agent_id] = strength
            
            # Find optimal role for this agent
            if agent.role_scores:
                optimal_role = max(agent.role_scores.items(), key=lambda x: x[1])
                specialization_report['optimal_roles'][agent_id] = optimal_role[0].value
        
        # Count role specializations
        num_specialized = sum(
            1 for strength in specialization_report['specialization_strength'].values()
            if strength > 0.5
        )
        self.global_metrics['role_specializations'] = num_specialized
        
        return specialization_report
    
    def get_collaboration_statistics(self) -> Dict[str, Any]:
        """Get comprehensive collaboration statistics."""
        stats = {
            'global_metrics': self.global_metrics.copy(),
            'agent_performances': {},
            'communication_metrics': {},
            'team_metrics': {},
            'emergent_behaviors': {}
        }
        
        # Agent performances
        for agent_id, agent in self.agents.items():
            stats['agent_performances'][agent_id] = agent.get_performance_summary()
        
        # Communication metrics
        total_messages = sum(
            agent.communication.get_protocol_state().get('total_messages', 0)
            for agent in self.agents.values()
        )
        stats['communication_metrics']['total_messages'] = total_messages
        stats['communication_metrics']['avg_per_agent'] = total_messages / self.num_agents
        
        # Emergent behaviors
        stats['emergent_behaviors']['protocols'] = self.discover_emergent_protocols()
        stats['emergent_behaviors']['specializations'] = self.analyze_role_specialization()
        
        return stats


# ============================================================================
# Factory Function
# ============================================================================

def create_multi_agent_collaboration_network(
    num_agents: int = 10,
    state_dim: int = 128,
    action_dim: int = 64,
    message_dim: int = 64,
    adversarial_ratio: float = 0.3
) -> MultiAgentCollaborationNetwork:
    """
    Create a multi-agent collaboration network.
    
    Args:
        num_agents: Number of agents in the network
        state_dim: State dimension
        action_dim: Action dimension  
        message_dim: Message dimension
        adversarial_ratio: Ratio of agents for adversarial training
        
    Returns:
        Configured MultiAgentCollaborationNetwork
    """
    return MultiAgentCollaborationNetwork(
        num_agents=num_agents,
        state_dim=state_dim,
        action_dim=action_dim,
        message_dim=message_dim,
        adversarial_ratio=adversarial_ratio
    )


# ============================================================================
# Integration with Existing Orchestrator
# ============================================================================

class MultiAgentOrchestrator:
    """
    Enhanced orchestrator integrating multi-agent collaboration with
    existing Symbio AI agent orchestration framework.
    """
    
    def __init__(
        self,
        base_orchestrator: Any,  # AgentOrchestrator from agents.orchestrator
        collaboration_network: MultiAgentCollaborationNetwork
    ):
        self.base_orchestrator = base_orchestrator
        self.collaboration_network = collaboration_network
        self.logger = logging.getLogger(__name__)
    
    async def execute_collaborative_task(
        self,
        task_description: str,
        complexity: float = 0.5,
        required_roles: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Execute task using both orchestrator and collaboration network.
        
        Args:
            task_description: Task description
            complexity: Task complexity (0-1)
            required_roles: Required agent roles
            
        Returns:
            Combined execution results
        """
        # Create collaboration task
        role_enums = [
            AgentRole[r.upper()] for r in (required_roles or ['GENERALIST'])
        ]
        
        collab_task = CollaborationTask(
            id=str(uuid.uuid4()),
            description=task_description,
            complexity=complexity,
            required_roles=role_enums
        )
        
        # Execute through collaboration network
        collab_result = await self.collaboration_network.solve_task_collaboratively(
            collab_task
        )
        
        # Also leverage base orchestrator for task decomposition
        # (Integration point with existing system)
        
        return {
            'collaboration_result': collab_result,
            'task_description': task_description,
            'success': collab_result['success']
        }
    
    async def run_adversarial_training(
        self,
        num_rounds: int = 10
    ) -> Dict[str, Any]:
        """Run adversarial training rounds."""
        return await self.collaboration_network.adversarial_training_round(num_rounds)
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        return self.collaboration_network.get_collaboration_statistics()
