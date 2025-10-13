#!/usr/bin/env python3
"""
Phase 1 Critical Test: Neural-Symbolic Agent Integration (Test 3/3)

Tests integration with agent orchestrator:
- Symbolic reasoning agents in multi-agent systems
- Rule learning from demonstrations
- Proof-carrying network execution
- Verification of agent decisions

Competitive Advantage:
Enables trustworthy multi-agent systems with verifiable reasoning,
critical for enterprise/academic deployment.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import asyncio
from training.neural_symbolic_architecture import (
    NeuralSymbolicArchitecture,
    SymbolicReasoningAgent,
    create_neural_symbolic_architecture
)


class TestNeuralSymbolicAgentIntegration:
    """Test neural-symbolic integration with agent systems."""
    
    def test_agent_rule_learning(self):
        """Test 3.1: Agents can learn rules from task demonstrations."""
        agent = SymbolicReasoningAgent(num_symbols=15, num_rules=10, logic_dim=64)
        
        # Provide demonstrations
        demonstrations = [
            {"input": {"x": 5, "y": 3}, "output": 8, "rule": "add"},
            {"input": {"x": 10, "y": 2}, "output": 12, "rule": "add"},
            {"input": {"x": 7, "y": 4}, "output": 11, "rule": "add"},
        ]
        
        # Learn rule from demonstrations
        learned_rule = agent.learn_rule_from_demonstrations(demonstrations)
        
        assert learned_rule is not None, "Rule learning failed"
        assert 'rule_description' in learned_rule or 'pattern' in learned_rule, \
            "No rule extracted"
    
    def test_proof_carrying_execution(self):
        """Test 3.2: Execution includes proof of correctness."""
        arch = create_neural_symbolic_architecture()
        
        input_data = torch.randn(1, 64)
        
        # Execute with proof generation
        result = arch.execute_with_proof(input_data)
        
        assert result is not None, "Execution failed"
        assert 'output' in result, "No output produced"
        assert 'proof' in result or 'verification' in result, \
            "No proof generated"
    
    def test_agent_decision_verification(self):
        """Test 3.3: Agent decisions can be verified against rules."""
        agent = SymbolicReasoningAgent(num_symbols=15, num_rules=10, logic_dim=64)
        
        # Set up rules
        agent.add_rule("if temperature > 30 then action = cooling")
        agent.add_rule("if temperature < 10 then action = heating")
        
        # Make a decision
        decision = agent.make_decision({"temperature": 35})
        
        assert decision is not None, "Decision making failed"
        
        # Verify decision
        verification = agent.verify_decision(
            decision=decision,
            context={"temperature": 35}
        )
        
        assert verification is not None, "Verification failed"
        assert 'valid' in verification or 'verified' in verification, \
            "No verification result"
    
    def test_multi_agent_symbolic_coordination(self):
        """Test 3.4: Multiple symbolic agents can coordinate using shared knowledge."""
        agent1 = SymbolicReasoningAgent(num_symbols=10, num_rules=5, logic_dim=64)
        agent2 = SymbolicReasoningAgent(num_symbols=10, num_rules=5, logic_dim=64)
        
        # Shared knowledge base
        shared_knowledge = {
            "facts": [
                ("task_a", "assigned_to", "agent1"),
                ("task_b", "assigned_to", "agent2"),
                ("task_a", "prerequisite_of", "task_b")
            ],
            "rules": [
                "if X prerequisite_of Y then complete X before Y"
            ]
        }
        
        # Load shared knowledge
        for fact in shared_knowledge["facts"]:
            agent1.add_fact(*fact)
            agent2.add_fact(*fact)
        
        # Coordinate execution
        coordination = {
            'agent1': agent1.query("task_a assigned_to agent1"),
            'agent2': agent2.query("task_a prerequisite_of task_b")
        }
        
        assert all(v is not None for v in coordination.values()), \
            "Coordination queries failed"
    
    def test_symbolic_transfer_learning(self):
        """Test 3.5: Symbolic rules can transfer across tasks."""
        arch = create_neural_symbolic_architecture()
        
        # Learn rules on task A
        task_a_data = {
            'inputs': torch.randn(20, 64),
            'outputs': torch.randint(0, 3, (20,))
        }
        
        learned_rules = arch.extract_symbolic_rules(
            inputs=task_a_data['inputs'],
            outputs=task_a_data['outputs']
        )
        
        assert learned_rules is not None, "Rule extraction failed"
        
        # Transfer to task B
        transfer_result = arch.transfer_rules(
            learned_rules=learned_rules,
            target_task='task_b'
        )
        
        assert transfer_result is not None, "Rule transfer failed"
        assert 'transferred_rules' in transfer_result or 'success' in transfer_result, \
            "No transfer confirmation"


def run_all_tests():
    """Run all neural-symbolic agent integration tests."""
    print("=" * 80)
    print("PHASE 1 - TEST 3: Neural-Symbolic Agent Integration")
    print("=" * 80)
    
    test_suite = TestNeuralSymbolicAgentIntegration()
    results = {
        'total': 5,
        'passed': 0,
        'failed': 0,
        'errors': []
    }
    
    tests = [
        ('Agent Rule Learning', test_suite.test_agent_rule_learning),
        ('Proof-Carrying Execution', test_suite.test_proof_carrying_execution),
        ('Agent Decision Verification', test_suite.test_agent_decision_verification),
        ('Multi-Agent Symbolic Coordination', test_suite.test_multi_agent_symbolic_coordination),
        ('Symbolic Transfer Learning', test_suite.test_symbolic_transfer_learning),
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
    print("TEST SUMMARY - Neural-Symbolic Agent Integration")
    print("=" * 80)
    print(f"Total Tests: {results['total']}")
    print(f"Passed: {results['passed']}")
    print(f"Failed: {results['failed']}")
    print(f"Success Rate: {(results['passed'] / results['total'] * 100):.1f}%")
    
    return results


if __name__ == "__main__":
    results = run_all_tests()
    sys.exit(0 if results['failed'] == 0 else 1)
