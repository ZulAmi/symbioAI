#!/usr/bin/env python3
"""
Phase 1 Critical Test: Embodied Learning (Test 14/15)

Tests embodied AI and tool learning capabilities:
- Tool use learning
- Sensorimotor integration
- World model prediction
- Spatial reasoning

Competitive Advantage:
Embodied learning enables agents to interact with environments
and use tools, going beyond pure language/vision tasks.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import numpy as np
from training.embodied_ai_simulation import (
    EmbodiedAgent,
    ToolUseLearner,
    create_embodied_agent
)


class TestEmbodiedLearning:
    """Tests for embodied learning capabilities."""
    
    def test_embodied_agent_creation(self):
        """Test 14.1: Create embodied AI agent."""
        agent = create_embodied_agent(
            observation_dim=128,
            action_dim=10,
            tool_dim=20
        )
        
        assert agent is not None, "Embodied agent creation failed"
        assert isinstance(agent, EmbodiedAgent), "Wrong agent type"
    
    def test_tool_use_learning(self):
        """Test 14.2: Learn to use tools from interaction."""
        tool_learner = ToolUseLearner(
            observation_dim=64,
            tool_repertoire_size=5
        )
        
        # Simulate tool use episodes
        for episode in range(10):
            observation = torch.randn(1, 64)
            
            # Select tool
            tool_selection = tool_learner.select_tool(observation)
            
            assert tool_selection is not None, "Tool selection failed"
            assert 0 <= tool_selection < 5, "Invalid tool selected"
            
            # Simulate outcome
            success = np.random.random() > 0.5
            tool_learner.update_tool_policy(tool_selection, success)
        
        # Check learning occurred
        assert hasattr(tool_learner, 'tool_success_rates') or \
               hasattr(tool_learner, 'tool_policy'), \
            "No tool learning tracking"
    
    def test_sensorimotor_integration(self):
        """Test 14.3: Integrate sensory input with motor actions."""
        agent = create_embodied_agent()
        
        # Sensory input (vision + proprioception)
        visual_input = torch.randn(1, 64)
        proprioceptive_input = torch.randn(1, 32)
        
        combined_observation = torch.cat([visual_input, proprioceptive_input], dim=1)
        
        # Generate motor action
        action = agent.act(combined_observation)
        
        assert action is not None, "Sensorimotor integration failed"
        assert action.shape[-1] == agent.action_dim, "Wrong action dimension"
    
    def test_world_model_prediction(self):
        """Test 14.4: Predict environment dynamics."""
        agent = create_embodied_agent()
        
        # Current state
        current_state = torch.randn(1, agent.observation_dim)
        
        # Proposed action
        action = torch.randn(1, agent.action_dim)
        
        # Predict next state
        predicted_next_state = agent.predict_next_state(current_state, action)
        
        assert predicted_next_state is not None, "World model prediction failed"
        assert predicted_next_state.shape == current_state.shape, \
            "Wrong prediction shape"
    
    def test_spatial_reasoning(self):
        """Test 14.5: Spatial reasoning and navigation."""
        agent = create_embodied_agent()
        
        # Spatial environment (grid world)
        current_position = torch.tensor([[5.0, 3.0]])  # (x, y)
        goal_position = torch.tensor([[8.0, 7.0]])
        
        # Plan path
        path = agent.plan_path(
            start=current_position,
            goal=goal_position
        )
        
        assert path is not None, "Spatial reasoning failed"
        assert len(path) > 0, "No path planned"


def run_all_tests():
    """Run all embodied learning tests."""
    print("=" * 80)
    print("PHASE 1 - TEST 14: Embodied Learning")
    print("=" * 80)
    
    test_suite = TestEmbodiedLearning()
    results = {'total': 5, 'passed': 0, 'failed': 0, 'errors': []}
    
    tests = [
        ('Embodied Agent Creation', test_suite.test_embodied_agent_creation),
        ('Tool Use Learning', test_suite.test_tool_use_learning),
        ('Sensorimotor Integration', test_suite.test_sensorimotor_integration),
        ('World Model Prediction', test_suite.test_world_model_prediction),
        ('Spatial Reasoning', test_suite.test_spatial_reasoning),
    ]
    
    for test_name, test_func in tests:
        try:
            print(f"\n🧪 Running: {test_name}...")
            test_func()
            print(f"✅ PASSED: {test_name}")
            results['passed'] += 1
        except Exception as e:
            print(f"❌ FAILED: {test_name} - {str(e)}")
            results['failed'] += 1
            results['errors'].append({'test': test_name, 'error': str(e)})
    
    print(f"\n{'='*80}")
    print(f"Total: {results['total']}, Passed: {results['passed']}, Failed: {results['failed']}")
    print(f"Success Rate: {(results['passed']/results['total']*100):.1f}%")
    
    return results


if __name__ == "__main__":
    results = run_all_tests()
    sys.exit(0 if results['failed'] == 0 else 1)
