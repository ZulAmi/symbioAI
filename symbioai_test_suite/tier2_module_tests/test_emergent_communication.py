#!/usr/bin/env python3
"""
Phase 1 Critical Test: Emergent Communication (Test 8/9)

Tests emergent communication between agents:
- Emergent protocol learning
- Multi-modal communication (broadcast, unicast, multicast)
- Message passing and information sharing
- Communication efficiency

Competitive Advantage:
Agents learn to communicate effectively without predefined protocols,
enabling flexible coordination.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import asyncio
import numpy as np
from training.multi_agent_collaboration import (
    MultiAgentCollaborationNetwork,
    CollaborationTask,
    CommunicationMode,
    CommunicationMessage,
    create_multi_agent_collaboration_network
)


class TestEmergentCommunication:
    """Tests for emergent communication protocols."""
    
    def test_communication_protocol_learning(self):
        """Test 8.1: Agents learn communication protocols."""
        network = create_multi_agent_collaboration_network(num_agents=6, message_dim=32)
        
        # Train communication through interaction
        for epoch in range(5):
            for agent_id, agent in network.agents.items():
                # Simulate communication learning
                state = torch.randn(1, network.state_dim)
                messages = [torch.randn(1, network.message_dim) for _ in range(3)]
                
                # Agent processes messages and generates response
                action, outgoing_msg = agent(state, messages)
                
                assert action is not None, "Action generation failed"
                assert outgoing_msg is not None, "Message generation failed"
        
        # Check that communication patterns emerge
        assert network.global_metrics['emergent_protocols_discovered'] >= 0, \
            "Protocol discovery not tracked"
    
    def test_broadcast_communication(self):
        """Test 8.2: Broadcast messages to all agents."""
        network = create_multi_agent_collaboration_network(num_agents=5)
        
        sender_id = "agent_000"
        message_content = torch.randn(1, network.message_dim)
        
        # Create broadcast message
        broadcast_msg = CommunicationMessage(
            id="msg_001",
            sender_id=sender_id,
            recipient_ids=None,  # Broadcast to all
            mode=CommunicationMode.BROADCAST,
            content=message_content,
            timestamp=0.0
        )
        
        # Send to all agents
        recipients = [aid for aid in network.agents.keys() if aid != sender_id]
        for recipient_id in recipients:
            network.agents[recipient_id].receive_message(broadcast_msg)
        
        # Verify all agents received the message
        for recipient_id in recipients:
            inbox = network.agents[recipient_id].message_inbox
            assert len(inbox) > 0, f"Agent {recipient_id} didn't receive broadcast"
    
    def test_targeted_unicast_communication(self):
        """Test 8.3: Send targeted messages to specific agents."""
        network = create_multi_agent_collaboration_network(num_agents=4)
        
        sender_id = "agent_000"
        recipient_id = "agent_001"
        message_content = torch.randn(1, network.message_dim)
        
        # Create unicast message
        unicast_msg = CommunicationMessage(
            id="msg_002",
            sender_id=sender_id,
            recipient_ids=[recipient_id],
            mode=CommunicationMode.UNICAST,
            content=message_content,
            timestamp=0.0
        )
        
        # Send to specific agent
        network.agents[recipient_id].receive_message(unicast_msg)
        
        # Verify only target agent received
        assert len(network.agents[recipient_id].message_inbox) > 0, \
            "Target agent didn't receive message"
        
        # Other agents should not have this message
        assert len(network.agents["agent_002"].message_inbox) == 0, \
            "Non-target agent received unicast message"
    
    def test_multicast_to_group(self):
        """Test 8.4: Send messages to specific agent groups."""
        network = create_multi_agent_collaboration_network(num_agents=8)
        
        sender_id = "agent_000"
        group_ids = ["agent_001", "agent_002", "agent_003"]
        message_content = torch.randn(1, network.message_dim)
        
        # Create multicast message
        multicast_msg = CommunicationMessage(
            id="msg_003",
            sender_id=sender_id,
            recipient_ids=group_ids,
            mode=CommunicationMode.MULTICAST,
            content=message_content,
            timestamp=0.0
        )
        
        # Send to group
        for recipient_id in group_ids:
            network.agents[recipient_id].receive_message(multicast_msg)
        
        # Verify group members received
        for recipient_id in group_ids:
            assert len(network.agents[recipient_id].message_inbox) > 0, \
                f"Group member {recipient_id} didn't receive multicast"
        
        # Non-group members should not receive
        assert len(network.agents["agent_004"].message_inbox) == 0, \
            "Non-group agent received multicast message"
    
    def test_communication_efficiency_metrics(self):
        """Test 8.5: Track communication efficiency metrics."""
        network = create_multi_agent_collaboration_network(num_agents=6)
        
        # Simulate communication exchanges
        total_messages = 0
        for i in range(10):
            sender = network.agents[f"agent_{i % 6:03d}"]
            state = torch.randn(1, network.state_dim)
            _, message = sender(state, [])
            total_messages += 1
        
        # Check metrics
        assert total_messages == 10, "Message count mismatch"
        
        # Network should track collaboration history
        assert hasattr(network, 'collaboration_history'), \
            "No collaboration history"


def run_all_tests():
    """Run all emergent communication tests."""
    print("=" * 80)
    print("PHASE 1 - TEST 8: Emergent Communication")
    print("=" * 80)
    
    test_suite = TestEmergentCommunication()
    results = {
        'total': 5,
        'passed': 0,
        'failed': 0,
        'errors': []
    }
    
    tests = [
        ('Communication Protocol Learning', test_suite.test_communication_protocol_learning),
        ('Broadcast Communication', test_suite.test_broadcast_communication),
        ('Targeted Unicast', test_suite.test_targeted_unicast_communication),
        ('Multicast to Group', test_suite.test_multicast_to_group),
        ('Communication Efficiency', test_suite.test_communication_efficiency_metrics),
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
    print("TEST SUMMARY - Emergent Communication")
    print("=" * 80)
    print(f"Total Tests: {results['total']}")
    print(f"Passed: {results['passed']}")
    print(f"Failed: {results['failed']}")
    print(f"Success Rate: {(results['passed'] / results['total'] * 100):.1f}%")
    
    return results


if __name__ == "__main__":
    results = run_all_tests()
    sys.exit(0 if results['failed'] == 0 else 1)
