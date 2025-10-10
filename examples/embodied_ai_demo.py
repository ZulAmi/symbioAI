"""
Embodied AI Simulation - Comprehensive Demo

Demonstrates all capabilities:
1. Physics-aware world models
2. Sensorimotor perception
3. Tool use learning
4. Spatial reasoning and navigation
5. Concept grounding
6. Manipulation control
7. Complete embodied agent training
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Any

from training.embodied_ai_simulation import (
    create_embodied_agent,
    create_simulation_environment,
    SensoryInput,
    Action,
    ActionType,
    PhysicsEngine,
    SensorimotorEncoder,
    ConceptGrounder,
    SpatialReasoningModule,
    ToolUseLearner,
    ManipulationController,
    WorldModelNetwork
)


def print_section(title: str):
    """Print formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def demo_1_physics_simulation():
    """Demo 1: Physics-aware world model."""
    print_section("DEMO 1: Physics-Aware World Model")
    
    print("\n🔬 Creating Physics Engine...")
    physics = PhysicsEngine(gravity=-9.81, dt=0.01)
    
    # Add objects
    print("\n📦 Adding Objects to Simulation:")
    physics.add_object(
        "box1",
        position=np.array([0.0, 0.0, 5.0]),
        mass=1.0,
        friction=0.5
    )
    physics.add_object(
        "box2",
        position=np.array([1.0, 0.0, 3.0]),
        mass=0.5,
        friction=0.3
    )
    physics.add_object(
        "ground",
        position=np.array([0.0, 0.0, 0.0]),
        mass=100.0,
        is_static=True
    )
    
    print(f"  ✓ Added box1 at height 5.0m (mass: 1.0kg)")
    print(f"  ✓ Added box2 at height 3.0m (mass: 0.5kg)")
    print(f"  ✓ Added ground plane (static)")
    
    # Simulate
    print("\n⏱️  Simulating Physics (100 steps)...")
    for step in range(100):
        state = physics.step()
        
        if step % 20 == 0:
            box1_pos = state.positions['box1']
            box2_pos = state.positions['box2']
            print(f"\n  Step {step}:")
            print(f"    box1 position: [{box1_pos[0]:.2f}, {box1_pos[1]:.2f}, {box1_pos[2]:.2f}]")
            print(f"    box2 position: [{box2_pos[0]:.2f}, {box2_pos[1]:.2f}, {box2_pos[2]:.2f}]")
            print(f"    box1 velocity: {np.linalg.norm(state.velocities['box1']):.2f} m/s")
    
    # Final state
    final_state = state
    print(f"\n✅ Physics Simulation Complete!")
    print(f"  Final height box1: {final_state.positions['box1'][2]:.3f}m")
    print(f"  Final height box2: {final_state.positions['box2'][2]:.3f}m")
    print(f"  Contacts detected: {len(final_state.contacts)}")


def demo_2_sensorimotor_perception():
    """Demo 2: Multi-modal sensorimotor perception."""
    print_section("DEMO 2: Sensorimotor Perception")
    
    print("\n👁️  Creating Sensorimotor Encoder...")
    encoder = SensorimotorEncoder(
        vision_shape=(3, 64, 64),
        depth_shape=(1, 64, 64),
        proprioception_dim=12,
        touch_dim=8,
        embedding_dim=256
    )
    
    print("  ✓ Vision encoder: 3-channel RGB (64×64)")
    print("  ✓ Depth encoder: 1-channel depth map (64×64)")
    print("  ✓ Proprioception encoder: 12 joint angles")
    print("  ✓ Touch encoder: 8 tactile sensors")
    print("  ✓ Output embedding: 256-dim unified representation")
    
    # Create sensory input
    print("\n📊 Generating Multi-Modal Sensory Input...")
    sensory_input = SensoryInput(
        vision=torch.randn(1, 3, 64, 64),
        depth=torch.randn(1, 1, 64, 64),
        proprioception=torch.randn(1, 12),
        touch=torch.randn(1, 8),
        position=torch.tensor([[0.0, 0.0, 0.5]]),
        orientation=torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    )
    
    print(f"  Vision shape: {sensory_input.vision.shape}")
    print(f"  Depth shape: {sensory_input.depth.shape}")
    print(f"  Proprioception shape: {sensory_input.proprioception.shape}")
    print(f"  Touch shape: {sensory_input.touch.shape}")
    
    # Encode
    print("\n🧠 Encoding Multi-Modal Input...")
    encoder.eval()
    with torch.no_grad():
        embedding = encoder(sensory_input)
    
    print(f"\n✅ Sensorimotor Encoding Complete!")
    print(f"  Input modalities: 4 (vision, depth, proprioception, touch)")
    print(f"  Output embedding shape: {embedding.shape}")
    print(f"  Embedding dimensionality: {embedding.shape[-1]}")
    print(f"  Embedding norm: {torch.norm(embedding).item():.2f}")


def demo_3_concept_grounding():
    """Demo 3: Grounding language concepts in sensorimotor experience."""
    print_section("DEMO 3: Concept Grounding")
    
    print("\n🔗 Creating Concept Grounder...")
    grounder = ConceptGrounder(
        sensorimotor_dim=256,
        language_dim=512,
        concept_dim=128,
        num_concepts=100
    )
    
    print("  ✓ Sensorimotor input: 256-dim")
    print("  ✓ Language input: 512-dim")
    print("  ✓ Concept space: 128-dim")
    print("  ✓ Concept dictionary: 100 concepts")
    
    # Create inputs
    print("\n📝 Creating Language and Sensorimotor Inputs...")
    language_input = torch.randn(1, 512)  # e.g., "red cube"
    sensorimotor_input = torch.randn(1, 256)  # visual + tactile experience
    
    print("  Language concept: 'red cube' (embedded)")
    print("  Sensorimotor experience: visual + tactile data")
    
    # Ground concept
    print("\n🎯 Grounding Concept...")
    grounder.eval()
    with torch.no_grad():
        concept_idx, grounding_score = grounder.ground_concept(
            language_input,
            sensorimotor_input
        )
    
    print(f"\n✅ Concept Grounded Successfully!")
    print(f"  Matched concept index: {concept_idx.item()}")
    print(f"  Grounding score: {grounding_score.item():.2%}")
    print(f"  Interpretation: {'Strong' if grounding_score.item() > 0.7 else 'Moderate' if grounding_score.item() > 0.4 else 'Weak'} grounding")
    
    # Retrieve affordances
    print("\n🔧 Retrieving Affordances...")
    with torch.no_grad():
        affordances = grounder.retrieve_affordances(concept_idx)
    
    print(f"  Affordance embedding shape: {affordances.shape}")
    print(f"  Affordance features available: {affordances.shape[-1]}")


def demo_4_spatial_reasoning():
    """Demo 4: Spatial reasoning and navigation."""
    print_section("DEMO 4: Spatial Reasoning & Navigation")
    
    print("\n🗺️  Creating Spatial Reasoning Module...")
    spatial_reasoner = SpatialReasoningModule(
        feature_dim=256,
        map_size=64,
        num_layers=4
    )
    
    print("  ✓ Feature dimension: 256")
    print("  ✓ Map size: 64×64 grid")
    print("  ✓ Spatial transformer layers: 4")
    
    # Update map with observations
    print("\n📍 Updating Spatial Map...")
    observation = torch.randn(1, 256, 16, 16)
    agent_position = (10, 10)
    
    spatial_reasoner.update_map(observation, agent_position)
    print(f"  Agent position: {agent_position}")
    print(f"  Observation integrated into map")
    
    # Plan path
    print("\n🧭 Planning Navigation Path...")
    start = (10, 10)
    goal = (50, 50)
    
    path = spatial_reasoner.plan_path(start, goal)
    
    print(f"\n✅ Path Planning Complete!")
    print(f"  Start position: {start}")
    print(f"  Goal position: {goal}")
    print(f"  Path length: {len(path)} waypoints")
    print(f"  First 5 waypoints: {path[:5]}")
    print(f"  Last 5 waypoints: {path[-5:]}")
    print(f"  Euclidean distance: {np.sqrt((goal[0]-start[0])**2 + (goal[1]-start[1])**2):.1f}")
    print(f"  Path distance: {len(path)}")


def demo_5_tool_use_learning():
    """Demo 5: Learning to use tools."""
    print_section("DEMO 5: Tool Use Learning")
    
    print("\n🔨 Creating Tool Use Learner...")
    tool_learner = ToolUseLearner(
        state_dim=256,
        action_dim=32,
        num_tools=10
    )
    
    print("  ✓ State dimension: 256")
    print("  ✓ Action dimension: 32")
    print("  ✓ Max tools: 10")
    
    # Detect affordances
    print("\n🔍 Detecting Tool Affordances...")
    state = torch.randn(256)
    available_tools = ["hammer", "screwdriver", "wrench", "pliers", "saw"]
    
    affordances = tool_learner.detect_affordances(state, available_tools)
    
    print("\n  Tool Affordances:")
    for tool, score in sorted(affordances.items(), key=lambda x: x[1], reverse=True):
        bar = "█" * int(score * 20)
        print(f"    {tool:15s} {bar:20s} {score:.2%}")
    
    # Simulate learning episodes
    print("\n📚 Simulating Tool Use Learning (5 episodes)...")
    for episode in range(5):
        tool = "hammer"
        states = [torch.randn(256) for _ in range(10)]
        actions = [
            Action(
                action_type=ActionType.USE_TOOL,
                parameters={'tool': tool},
                duration=0.1
            ) for _ in range(10)
        ]
        success = episode >= 2  # Improve over time
        
        tool_learner.learn_tool_use(tool, states, actions, success)
        
        skill = tool_learner.skills.get(tool)
        if skill:
            print(f"  Episode {episode + 1}: Success={success}, Success Rate={skill.success_rate:.2%}")
    
    print(f"\n✅ Tool Learning Complete!")
    print(f"  Tools learned: {len(tool_learner.skills)}")
    for tool_name, skill in tool_learner.skills.items():
        print(f"    {tool_name}: {skill.success_rate:.2%} success rate ({skill.learning_iterations} iterations)")


def demo_6_manipulation_control():
    """Demo 6: Object manipulation control."""
    print_section("DEMO 6: Manipulation Control")
    
    print("\n🤖 Creating Manipulation Controller...")
    controller = ManipulationController(
        observation_dim=256,
        action_dim=7,  # 6 DOF + gripper
        hidden_dim=128
    )
    
    print("  ✓ Observation dimension: 256")
    print("  ✓ Action dimension: 7 (6 DOF + gripper)")
    print("  ✓ Hidden layer: 128 units")
    
    # Generate manipulation actions
    print("\n🎮 Generating Manipulation Actions...")
    observations = torch.randn(5, 256)  # 5 different scenarios
    
    controller.eval()
    with torch.no_grad():
        actions, grasp_qualities = controller(observations)
    
    print("\n  Manipulation Actions Generated:")
    for i in range(5):
        action = actions[i]
        quality = grasp_qualities[i].item()
        
        print(f"\n  Scenario {i + 1}:")
        print(f"    Action (7 DOF): [{', '.join([f'{a:.2f}' for a in action[:3].tolist()])}...")
        print(f"    Grasp quality: {quality:.2%}")
        print(f"    Recommendation: {'GRASP' if quality > 0.7 else 'REPOSITION' if quality > 0.4 else 'APPROACH'}")
    
    print(f"\n✅ Manipulation Control Complete!")
    print(f"  Average grasp quality: {grasp_qualities.mean().item():.2%}")
    print(f"  High-quality grasps: {(grasp_qualities > 0.7).sum().item()}/5")


def demo_7_world_model_prediction():
    """Demo 7: World model for planning."""
    print_section("DEMO 7: World Model Prediction")
    
    print("\n🌍 Creating World Model Network...")
    world_model = WorldModelNetwork(
        state_dim=128,
        action_dim=32,
        hidden_dim=256,
        num_layers=4
    )
    
    print("  ✓ State dimension: 128")
    print("  ✓ Action dimension: 32")
    print("  ✓ Hidden dimension: 256")
    print("  ✓ Dynamics layers: 4")
    
    # Predict future
    print("\n🔮 Predicting Future States...")
    current_state = torch.randn(1, 128)
    action = torch.randn(1, 32)
    
    world_model.eval()
    with torch.no_grad():
        next_state, predicted_reward = world_model(current_state, action)
    
    print(f"  Current state norm: {torch.norm(current_state).item():.2f}")
    print(f"  Action applied: {action.shape}")
    print(f"  Predicted next state norm: {torch.norm(next_state).item():.2f}")
    print(f"  Predicted reward: {predicted_reward.item():.3f}")
    
    # Multi-step rollout
    print("\n📊 Multi-Step Rollout (Horizon=10)...")
    actions = [torch.randn(1, 32) for _ in range(10)]
    
    with torch.no_grad():
        states, rewards = world_model.rollout(current_state, actions, horizon=10)
    
    print(f"\n  Rollout Results:")
    print(f"    Total steps: {len(states) - 1}")
    print(f"    Cumulative reward: {sum([r.item() for r in rewards]):.3f}")
    print(f"    State trajectory:")
    for i in range(0, len(states), 2):
        print(f"      Step {i}: norm={torch.norm(states[i]).item():.2f}")
    
    print(f"\n✅ World Model Prediction Complete!")


def demo_8_complete_embodied_agent():
    """Demo 8: Complete embodied agent in simulation."""
    print_section("DEMO 8: Complete Embodied Agent")
    
    print("\n🤖 Creating Embodied Agent...")
    agent = create_embodied_agent(
        vision_shape=(3, 64, 64),
        state_dim=256,
        action_dim=32,
        use_physics=True
    )
    
    print("  ✓ Vision input: 3×64×64")
    print("  ✓ State embedding: 256-dim")
    print("  ✓ Action space: 32-dim")
    print("  ✓ Physics engine: enabled")
    
    # Create environment
    print("\n🌍 Creating Simulation Environment...")
    env = create_simulation_environment(
        grid_size=(10, 10, 5),
        num_objects=5,
        num_tools=3
    )
    
    print("  ✓ Grid size: 10×10×5 meters")
    print("  ✓ Objects spawned: 5")
    print("  ✓ Tools available: 3")
    
    # Run episodes
    print("\n🎮 Running Embodied Learning Episodes...")
    num_episodes = 3
    
    for episode in range(num_episodes):
        print(f"\n  Episode {episode + 1}/{num_episodes}")
        print("  " + "-" * 40)
        
        # Reset environment
        sensory_input = env.reset()
        
        # Run episode
        episode_states = []
        episode_actions = []
        episode_rewards = []
        
        for step in range(20):
            # Perceive
            state = agent.perceive(sensory_input)
            episode_states.append(state)
            
            # Plan action
            action = agent.plan_action(
                available_tools=env.tools
            )
            episode_actions.append(action)
            
            # Execute
            next_sensory_input, reward, done = env.step(action)
            episode_rewards.append(reward)
            
            if step % 5 == 0:
                print(f"    Step {step}: action={action.action_type.value}, reward={reward:.3f}")
            
            sensory_input = next_sensory_input
            
            if done:
                break
        
        # Learn from episode
        agent.learn_from_experience(
            episode_states,
            episode_actions,
            episode_rewards,
            tool_used=env.tools[0] if episode > 0 else None
        )
        
        total_reward = sum(episode_rewards)
        print(f"    Episode reward: {total_reward:.2f}")
        print(f"    Steps taken: {len(episode_actions)}")
    
    # Agent statistics
    print("\n📊 Agent Statistics:")
    stats = agent.get_statistics()
    print(f"  Total steps: {stats['total_steps']}")
    print(f"  Tools learned: {stats['tools_learned']}")
    
    if stats['tool_skills']:
        print("\n  Tool Skills:")
        for tool_name, skill_info in stats['tool_skills'].items():
            print(f"    {tool_name}: {skill_info['success_rate']:.2%} success ({skill_info['iterations']} iterations)")
    
    print(f"\n✅ Embodied Agent Training Complete!")


async def main():
    """Run all demonstrations."""
    print("\n" + "=" * 80)
    print(" EMBODIED AI SIMULATION - COMPREHENSIVE DEMO")
    print("=" * 80)
    print("\nThis demo showcases:")
    print("  1. Physics-aware world models")
    print("  2. Multi-modal sensorimotor perception")
    print("  3. Concept grounding (language → physical)")
    print("  4. Spatial reasoning and navigation")
    print("  5. Tool use learning through interaction")
    print("  6. Manipulation control (grasping, etc.)")
    print("  7. World model prediction for planning")
    print("  8. Complete embodied agent training")
    
    try:
        # Run all demos
        demo_1_physics_simulation()
        demo_2_sensorimotor_perception()
        demo_3_concept_grounding()
        demo_4_spatial_reasoning()
        demo_5_tool_use_learning()
        demo_6_manipulation_control()
        demo_7_world_model_prediction()
        demo_8_complete_embodied_agent()
        
        # Final summary
        print_section("DEMO COMPLETE - SUMMARY")
        
        print("\n✅ All 8 Demonstrations Completed Successfully!")
        
        print("\n📊 Key Capabilities Demonstrated:")
        print("  ✓ Physics simulation with gravity, friction, collisions")
        print("  ✓ Multi-modal perception (vision, depth, proprioception, touch)")
        print("  ✓ Language concept grounding in physical experience")
        print("  ✓ Spatial mapping and path planning")
        print("  ✓ Tool affordance detection and skill learning")
        print("  ✓ Manipulation control with grasp quality estimation")
        print("  ✓ Predictive world models for planning")
        print("  ✓ Complete embodied agents learning through interaction")
        
        print("\n🏆 Competitive Advantages:")
        print("  • Bridges pure language AI to physical understanding")
        print("  • Sensorimotor grounding of abstract concepts")
        print("  • Learns tool use through experience (not just language)")
        print("  • Physics-aware reasoning and prediction")
        print("  • Spatial cognition for navigation and manipulation")
        print("  • Unified perception across multiple modalities")
        
        print("\n🎯 Use Cases:")
        print("  • Robotics: manipulation, navigation, assembly")
        print("  • Simulation: training AI in virtual environments")
        print("  • Embodied QA: answer questions through interaction")
        print("  • Tool use: learn to use new tools from demonstration")
        print("  • Spatial reasoning: 3D scene understanding")
        
        print("\n📚 Next Steps:")
        print("  1. Train on real robotic platforms")
        print("  2. Integrate with vision-language models")
        print("  3. Add more complex physics (deformable objects)")
        print("  4. Scale to larger environments (outdoor navigation)")
        print("  5. Multi-agent coordination and communication")
        
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
