#!/usr/bin/env python3
"""
Simple Multi-Agent Orchestration Framework Demo
This demonstrates the exact prompt structure you requested, integrated with our production system.

Based on your prompt:
```
Agent Orchestrator: Coordinate multiple specialized agents to solve complex tasks.
```
"""

import asyncio
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agents.orchestrator import AgentOrchestrator, ReasoningAgent, CodingAgent, VisionAgent


class SimpleAgentFactory:
    """Simple factory for creating agents as requested in your prompt."""
    
    @staticmethod
    def create_agent(cfg):
        """Create agent from configuration - matches your prompt's agents.create_agent(cfg)"""
        agent_type = cfg.get("type", "reasoning")
        agent_id = cfg.get("id", f"{agent_type}_agent")
        
        if agent_type == "reasoning":
            return ReasoningAgent(agent_id, cfg)
        elif agent_type == "coding":
            return CodingAgent(agent_id, cfg)
        elif agent_type == "vision":
            return VisionAgent(agent_id, cfg)
        else:
            # Default to reasoning agent
            return ReasoningAgent(agent_id, cfg)


# Mock the agents module for this demo (in production, this would be a real module)
class MockAgentsModule:
    create_agent = SimpleAgentFactory.create_agent


# Simple version following your exact prompt structure
class SimpleAgentOrchestrator:
    """Manages a team of AI agents with different specialties."""
    
    def __init__(self, agent_configs):
        # Initialize agents (e.g., reasoning_agent, coding_agent, vision_agent)
        # Using our factory instead of the global agents module
        self.agents = [SimpleAgentFactory.create_agent(cfg) for cfg in agent_configs]
        self.agent_lookup = {agent.agent_id: agent for agent in self.agents}
    
    async def solve_task(self, task):
        """Distribute sub-tasks among agents and aggregate results."""
        print(f"🎯 Solving task: {task}")
        
        # Example: break task into parts and assign to best-suited agent
        plan = await self._plan_task(task)
        partial_results = {}
        
        for subtask, agent in plan:
            print(f"  📋 Assigning '{subtask['description']}' to {agent.agent_id}")
            result = await agent.handle(subtask)
            partial_results[agent.agent_id] = result
        
        # Combine results from all agents
        final_result = await self._integrate(partial_results)
        print(f"✅ Task completed with result: {final_result.get('status', 'unknown')}")
        return final_result
    
    async def _plan_task(self, task):
        """Analyze task and create a plan mapping subtasks to agents."""
        # Simple task decomposition logic
        task_lower = task.lower()
        plan = []
        
        # Reasoning task
        if any(word in task_lower for word in ["analyze", "think", "reason", "problem"]):
            reasoning_agent = self._find_agent_by_type("reasoning")
            if reasoning_agent:
                subtask = {
                    "id": "reasoning_subtask",
                    "type": "reasoning", 
                    "description": f"Analyze and reason about: {task}"
                }
                plan.append((subtask, reasoning_agent))
        
        # Coding task
        if any(word in task_lower for word in ["code", "program", "develop", "script"]):
            coding_agent = self._find_agent_by_type("coding")
            if coding_agent:
                subtask = {
                    "id": "coding_subtask",
                    "type": "coding",
                    "description": f"Generate code for: {task}"
                }
                plan.append((subtask, coding_agent))
        
        # Vision task
        if any(word in task_lower for word in ["image", "visual", "picture", "photo"]):
            vision_agent = self._find_agent_by_type("vision")
            if vision_agent:
                subtask = {
                    "id": "vision_subtask", 
                    "type": "image_preprocessing",
                    "description": f"Process visual content for: {task}"
                }
                plan.append((subtask, vision_agent))
        
        # If no specific subtasks, assign to reasoning agent as default
        if not plan:
            reasoning_agent = self._find_agent_by_type("reasoning")
            if reasoning_agent:
                subtask = {
                    "id": "general_subtask",
                    "type": "reasoning",
                    "description": task
                }
                plan.append((subtask, reasoning_agent))
        
        return plan
    
    async def _integrate(self, partial_results):
        """Integrate partial results from agents into a final answer."""
        if not partial_results:
            return {"status": "no_results", "message": "No partial results to integrate"}
        
        # Simple voting/aggregation strategy
        successful_results = []
        failed_results = []
        
        for agent_name, result in partial_results.items():
            if result.get("status") == "completed":
                successful_results.append({
                    "agent": agent_name,
                    "result": result.get("result", {}),
                    "confidence": result.get("result", {}).get("confidence_score", 0.5)
                })
            else:
                failed_results.append({"agent": agent_name, "error": result.get("error")})
        
        if not successful_results:
            return {
                "status": "failed", 
                "message": "All agents failed",
                "failures": failed_results
            }
        
        # Combine successful results
        combined_result = {
            "status": "completed",
            "strategy": "simple_aggregation", 
            "agent_results": successful_results,
            "summary": f"Successfully integrated results from {len(successful_results)} agents",
            "confidence": sum(r["confidence"] for r in successful_results) / len(successful_results)
        }
        
        if failed_results:
            combined_result["partial_failures"] = failed_results
        
        return combined_result
    
    def _find_agent_by_type(self, agent_type):
        """Find an agent by its type."""
        for agent in self.agents:
            if hasattr(agent, 'agent_type') and agent.agent_type == agent_type:
                return agent
        return None


async def demonstrate_orchestration():
    """Demonstrate the multi-agent orchestration system."""
    print("🚀 Multi-Agent Orchestration Framework Demo")
    print("=" * 50)
    
    # Configure agents following your prompt structure
    agent_configs = [
        {
            "type": "reasoning", 
            "id": "reasoning_agent",
            "reasoning_depth": 3,
            "logic_framework": "deductive"
        },
        {
            "type": "coding", 
            "id": "coding_agent", 
            "languages": ["python", "javascript"],
            "code_style": "clean_code"
        },
        {
            "type": "vision", 
            "id": "vision_agent",
            "model_type": "cnn",
            "formats": ["jpg", "png"]
        }
    ]
    
    # Initialize orchestrator
    orchestrator = SimpleAgentOrchestrator(agent_configs)
    print(f"✅ Initialized orchestrator with {len(orchestrator.agents)} agents")
    
    # Test cases
    test_tasks = [
        "Analyze the problem of optimizing machine learning model performance",
        "Write Python code to implement a binary search algorithm", 
        "Process and classify images from a dataset",
        "Create a comprehensive solution for data pipeline optimization with code and analysis"
    ]
    
    for i, task in enumerate(test_tasks, 1):
        print(f"\n🔍 Test Case {i}:")
        print("-" * 30)
        result = await orchestrator.solve_task(task)
        print(f"📊 Final Result Summary: {result.get('summary', 'No summary available')}")
        print(f"🎯 Confidence: {result.get('confidence', 0):.2f}")
    
    print("\n🎉 Multi-Agent Orchestration Demo Complete!")
    print("💡 This demonstrates adaptive task assignment and collective intelligence")


if __name__ == "__main__":
    asyncio.run(demonstrate_orchestration())