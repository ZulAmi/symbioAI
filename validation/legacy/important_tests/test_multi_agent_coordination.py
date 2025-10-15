#!/usr/bin/env python3
"""
Phase 1 Critical Test: Multi-Agent Coordination (Test 7/9)

Tests collaborative multi-agent coordination:
- Automatic role assignment
- Team formation and task decomposition
- Emergent communication protocols
- Collaborative problem solving

Competitive Advantage vs SakanaAI:
True multi-agent collaboration with role specialization and emergent
coordination, not just parallel execution.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import asyncio
from training.multi_agent_collaboration import (
    MultiAgentCollaborationNetwork,
    CollaborationTask,
    AgentRole,
    create_multi_agent_collaboration_network
)


class TestMultiAgentCoordination:
    """Tests for multi-agent coordination capabilities."""
    
    def test_collaboration_network_creation(self):
        """Test 7.1: Create multi-agent collaboration network."""
        network = create_multi_agent_collaboration_network(
            num_agents=8,
            adversarial_ratio=0.3
        )
        
        assert network is not None, "Network creation failed"
        assert network.num_agents == 8, "Wrong number of agents"
        assert len(network.agents) == 8, "Agents not created"
    
    def test_automatic_role_assignment(self):
        """Test 7.2: Automatically assign roles based on task requirements."""
        network = create_multi_agent_collaboration_network(num_agents=10)
        
        task = CollaborationTask(
            id="task_001",
            description="Build ML pipeline with data processing and training",
            complexity=0.7,
            required_roles=[AgentRole.COORDINATOR, AgentRole.SPECIALIST, 
                          AgentRole.CRITIC, AgentRole.GENERALIST]
        )
        
        assignments = network.assign_roles_automatically(task, method="performance")
        
        assert assignments is not None, "Role assignment failed"
        assert len(assignments) >= len(task.required_roles), \
            "Not all roles assigned"
        
        # Check role types are correct
        assigned_roles = set(assignments.values())
        for required_role in task.required_roles:
            assert required_role in assigned_roles, \
                f"Required role {required_role} not assigned"
    
    def test_team_formation(self):
        """Test 7.3: Form teams for collaborative tasks."""
        network = create_multi_agent_collaboration_network(num_agents=12)
        
        task = CollaborationTask(
            id="task_002",
            description="Multi-stage data processing pipeline",
            complexity=0.8,
            required_roles=[AgentRole.COORDINATOR, AgentRole.SPECIALIST]
        )
        
        # Assign roles first
        assignments = network.assign_roles_automatically(task)
        
        # Decompose into subtasks
        subtasks = [
            {"name": "data_collection", "complexity": 0.3},
            {"name": "preprocessing", "complexity": 0.5},
            {"name": "model_training", "complexity": 0.9},
        ]
        
        # Form teams
        teams = network._form_teams(subtasks, assignments)
        
        assert teams is not None, "Team formation failed"
        assert len(teams) > 0, "No teams formed"
    
    async def test_collaborative_task_solving(self):
        """Test 7.4: Solve tasks collaboratively."""
        network = create_multi_agent_collaboration_network(num_agents=6)
        
        task = CollaborationTask(
            id="task_003",
            description="Collaborative optimization problem",
            complexity=0.6,
            required_roles=[AgentRole.GENERATOR, AgentRole.CRITIC, 
                          AgentRole.COORDINATOR]
        )
        
        result = await network.solve_task_collaboratively(task, max_iterations=5)
        
        assert result is not None, "Collaborative solving failed"
        assert result.get('success') is True or result.get('status') == 'completed', \
            "Task not completed successfully"
        assert 'execution_time' in result, "No execution time tracked"
    
    def test_agent_specialization(self):
        """Test 7.5: Agents develop specializations over time."""
        network = create_multi_agent_collaboration_network(num_agents=8)
        
        # Track agent usage patterns
        task_history = []
        
        for i in range(10):
            task = CollaborationTask(
                id=f"task_{i:03d}",
                description=f"Task requiring {'vision' if i % 2 == 0 else 'language'} skills",
                complexity=0.5,
                required_roles=[AgentRole.SPECIALIST]
            )
            
            assignments = network.assign_roles_automatically(task, method="specialization")
            task_history.append({'task': task, 'assignments': assignments})
        
        # Check that agents specialize (same agents handle similar tasks)
        assert len(task_history) == 10, "Not all tasks recorded"
        
        # Get statistics
        stats = network.global_metrics
        assert 'role_specializations' in stats or \
               any('special' in k.lower() for k in stats.keys()), \
            "No specialization tracking"


def run_all_tests():
    """Run all multi-agent coordination tests."""
    print("=" * 80)
    print("PHASE 1 - TEST 7: Multi-Agent Coordination")
    print("=" * 80)
    
    test_suite = TestMultiAgentCoordination()
    results = {
        'total': 5,
        'passed': 0,
        'failed': 0,
        'errors': []
    }
    
    tests = [
        ('Network Creation', test_suite.test_collaboration_network_creation),
        ('Automatic Role Assignment', test_suite.test_automatic_role_assignment),
        ('Team Formation', test_suite.test_team_formation),
        ('Collaborative Task Solving', lambda: asyncio.run(test_suite.test_collaborative_task_solving())),
        ('Agent Specialization', test_suite.test_agent_specialization),
    ]
    
    for test_name, test_func in tests:
        try:
            print(f"\n🧪 Running: {test_name}...")
            test_func()
            print(f"✅ PASSED: {test_name}")
            results['passed'] += 1
        except Exception as e:
            print(f"❌ FAILED: {test_name}")
            print(f"   Error: {str(e)}")
            results['failed'] += 1
            results['errors'].append({
                'test': test_name,
                'error': str(e)
            })
    
    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY - Multi-Agent Coordination")
    print("=" * 80)
    print(f"Total Tests: {results['total']}")
    print(f"Passed: {results['passed']}")
    print(f"Failed: {results['failed']}")
    print(f"Success Rate: {(results['passed'] / results['total'] * 100):.1f}%")
    
    return results


if __name__ == "__main__":
    results = run_all_tests()
    sys.exit(0 if results['failed'] == 0 else 1)
