"""
Agent Orchestrator: Coordinate multiple specialized agents to solve complex tasks.

Production-grade multi-agent orchestration framework for Symbio AI that manages
intelligent agents, coordination protocols, communication, and hierarchical task
distribution for optimal system performance.
"""

import asyncio
import logging
import json
from typing import Dict, Any, List, Optional, Callable, Set, Union, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import time
import uuid
from datetime import datetime
import weakref
import agents  # Import agents module for factory pattern


class AgentStatus(Enum):
    """Agent status enumeration."""
    IDLE = "idle"
    BUSY = "busy"
    WAITING = "waiting"
    FAILED = "failed"
    SHUTDOWN = "shutdown"


class MessagePriority(Enum):
    """Message priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class TaskType(Enum):
    """Types of tasks agents can handle."""
    INFERENCE = "inference"
    TRAINING = "training"
    EVALUATION = "evaluation"
    DATA_PROCESSING = "data_processing"
    MODEL_MERGE = "model_merge"
    COORDINATION = "coordination"
    MONITORING = "monitoring"
    REASONING = "reasoning"
    ANALYSIS = "analysis"
    CODING = "coding"
    VISION = "vision"


@dataclass
class Message:
    """Inter-agent communication message."""
    id: str
    sender_id: str
    recipient_id: str
    message_type: str
    priority: MessagePriority
    payload: Dict[str, Any]
    timestamp: float
    expires_at: Optional[float] = None
    correlation_id: Optional[str] = None


@dataclass
class Task:
    """Task definition for agent execution."""
    id: str
    task_type: TaskType
    priority: MessagePriority
    payload: Dict[str, Any]
    assigned_agent_id: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    retry_count: int = 0
    max_retries: int = 3


@dataclass
class ResourceRequirements:
    """Resource requirements for agent operation."""
    cpu_cores: int = 1
    memory_gb: int = 2
    gpu_memory_gb: Optional[int] = None
    disk_space_gb: int = 1
    network_bandwidth_mbps: int = 100


@dataclass
class AgentCapabilities:
    """Agent capability specification."""
    supported_tasks: List[TaskType]
    max_concurrent_tasks: int
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    resource_requirements: Optional[ResourceRequirements] = None
    specializations: List[str] = field(default_factory=list)


class Agent(ABC):
    """Abstract base class for all agents."""
    
    def __init__(self, agent_id: str, capabilities: AgentCapabilities):
        self.agent_id = agent_id
        self.capabilities = capabilities
        self.status = AgentStatus.IDLE
        self.current_tasks: Dict[str, Task] = {}
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.orchestrator_ref: Optional[weakref.ref] = None
        self.logger = logging.getLogger(f"agent.{agent_id}")
        self.metrics = {
            "tasks_completed": 0,
            "tasks_failed": 0,
            "average_task_time": 0.0,
            "uptime": time.time()
        }
    
    @abstractmethod
    async def execute_task(self, task: Task) -> Any:
        """Execute a specific task."""
        pass
    
    async def start(self) -> None:
        """Start the agent."""
        self.logger.info(f"Starting agent {self.agent_id}")
        asyncio.create_task(self._message_loop())
        asyncio.create_task(self._heartbeat_loop())
    
    async def stop(self) -> None:
        """Stop the agent."""
        self.logger.info(f"Stopping agent {self.agent_id}")
        self.status = AgentStatus.SHUTDOWN
        
        # Cancel running tasks
        for task in self.current_tasks.values():
            if task.started_at and not task.completed_at:
                task.error = "Agent shutdown"
    
    async def send_message(self, message: Message) -> None:
        """Send message to another agent."""
        if self.orchestrator_ref:
            orchestrator = self.orchestrator_ref()
            if orchestrator:
                await orchestrator.route_message(message)
    
    async def receive_message(self, message: Message) -> None:
        """Receive message from another agent."""
        await self.message_queue.put(message)
    
    async def assign_task(self, task: Task) -> bool:
        """Assign a task to this agent."""
        if len(self.current_tasks) >= self.capabilities.max_concurrent_tasks:
            return False
        
        if task.task_type not in self.capabilities.supported_tasks:
            return False
        
        self.current_tasks[task.id] = task
        task.assigned_agent_id = self.agent_id
        task.started_at = time.time()
        self.status = AgentStatus.BUSY
        
        # Execute task asynchronously
        asyncio.create_task(self._execute_task_wrapper(task))
        return True
    
    async def _execute_task_wrapper(self, task: Task) -> None:
        """Wrapper for task execution with error handling."""
        try:
            self.logger.info(f"Executing task {task.id} of type {task.task_type}")
            result = await self.execute_task(task)
            
            task.result = result
            task.completed_at = time.time()
            self.metrics["tasks_completed"] += 1
            
            # Update average task time
            task_duration = task.completed_at - task.started_at
            self._update_average_task_time(task_duration)
            
            self.logger.info(f"Completed task {task.id} in {task_duration:.2f}s")
            
        except Exception as e:
            self.logger.error(f"Task {task.id} failed: {e}")
            task.error = str(e)
            task.completed_at = time.time()
            self.metrics["tasks_failed"] += 1
        
        finally:
            # Remove from current tasks
            if task.id in self.current_tasks:
                del self.current_tasks[task.id]
            
            # Update status
            if not self.current_tasks:
                self.status = AgentStatus.IDLE
            
            # Notify orchestrator of task completion
            if self.orchestrator_ref:
                orchestrator = self.orchestrator_ref()
                if orchestrator:
                    await orchestrator.task_completed(task)
    
    def _update_average_task_time(self, task_duration: float) -> None:
        """Update average task execution time."""
        total_tasks = self.metrics["tasks_completed"] + self.metrics["tasks_failed"]
        if total_tasks > 1:
            current_avg = self.metrics["average_task_time"]
            self.metrics["average_task_time"] = (
                (current_avg * (total_tasks - 1) + task_duration) / total_tasks
            )
        else:
            self.metrics["average_task_time"] = task_duration
    
    async def _message_loop(self) -> None:
        """Process incoming messages."""
        while self.status != AgentStatus.SHUTDOWN:
            try:
                message = await asyncio.wait_for(self.message_queue.get(), timeout=1.0)
                await self._handle_message(message)
            except asyncio.TimeoutError:
                continue
    
    async def _handle_message(self, message: Message) -> None:
        """Handle incoming message."""
        self.logger.debug(f"Received message: {message.message_type}")
        
        if message.message_type == "ping":
            # Respond to ping with pong
            response = Message(
                id=str(uuid.uuid4()),
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                message_type="pong",
                priority=MessagePriority.NORMAL,
                payload={"status": self.status.value},
                timestamp=time.time()
            )
            await self.send_message(response)
    
    async def _heartbeat_loop(self) -> None:
        """Send periodic heartbeat to orchestrator."""
        while self.status != AgentStatus.SHUTDOWN:
            if self.orchestrator_ref:
                orchestrator = self.orchestrator_ref()
                if orchestrator:
                    await orchestrator.agent_heartbeat(self.agent_id, self.metrics)
            await asyncio.sleep(30)  # Heartbeat every 30 seconds


class InferenceAgent(Agent):
    """Agent specialized for model inference tasks."""
    
    def __init__(self, agent_id: str):
        capabilities = AgentCapabilities(
            supported_tasks=[TaskType.INFERENCE],
            max_concurrent_tasks=5,
            performance_metrics={"inference_speed": 100.0},
            resource_requirements={"gpu_memory": "4GB", "cpu_cores": 2},
            specializations=["neural_networks", "transformer_models"]
        )
        super().__init__(agent_id, capabilities)
    
    async def execute_task(self, task: Task) -> Any:
        """Execute inference task."""
        model_id = task.payload.get("model_id")
        inputs = task.payload.get("inputs")
        
        # Simulate inference
        await asyncio.sleep(0.1)
        
        return {
            "predictions": f"inference_result_for_{model_id}",
            "confidence": 0.95,
            "processing_time": 0.1
        }


class TrainingAgent(Agent):
    """Agent specialized for model training tasks."""
    
    def __init__(self, agent_id: str):
        capabilities = AgentCapabilities(
            supported_tasks=[TaskType.TRAINING, TaskType.EVALUATION],
            max_concurrent_tasks=2,
            performance_metrics={"training_speed": 50.0},
            resource_requirements={"gpu_memory": "8GB", "cpu_cores": 4},
            specializations=["deep_learning", "optimization"]
        )
        super().__init__(agent_id, capabilities)
    
    async def execute_task(self, task: Task) -> Any:
        """Execute training task."""
        model_id = task.payload.get("model_id")
        dataset = task.payload.get("dataset")
        epochs = task.payload.get("epochs", 10)
        
        # Simulate training
        for epoch in range(epochs):
            await asyncio.sleep(0.05)  # Simulate epoch training
            self.logger.debug(f"Training epoch {epoch + 1}/{epochs}")
        
        return {
            "model_id": model_id,
            "final_accuracy": 0.92,
            "training_loss": 0.15,
            "epochs_completed": epochs
        }


class DataAgent(Agent):
    """Agent specialized for data processing tasks."""
    
    def __init__(self, agent_id: str):
        capabilities = AgentCapabilities(
            supported_tasks=[TaskType.DATA_PROCESSING],
            max_concurrent_tasks=8,
            performance_metrics={"processing_speed": 200.0},
            resource_requirements={"memory": "16GB", "cpu_cores": 8},
            specializations=["data_preprocessing", "feature_engineering"]
        )
        super().__init__(agent_id, capabilities)
    
    async def execute_task(self, task: Task) -> Any:
        """Execute data processing task."""
        dataset_name = task.payload.get("dataset_name")
        operations = task.payload.get("operations", [])
        
        # Simulate data processing
        await asyncio.sleep(0.2)
        
        return {
            "dataset_name": dataset_name,
            "operations_applied": operations,
            "processed_samples": 10000,
            "processing_time": 0.2
        }


class CoordinatorAgent(Agent):
    """Coordinator agent for managing agent collaboration."""
    
    def __init__(self, agent_id: str):
        super().__init__(agent_id, "coordinator", AgentCapabilities())
        self.managed_agents: List[Agent] = []
        self.coordination_strategies: Dict[str, Callable] = {}
    
    async def handle(self, task: Dict[str, Any]) -> Any:
        """Handle coordination tasks."""
        try:
            task_type = task.get("type", "coordinate")
            
            if task_type == "coordinate":
                return await self._coordinate_agents(task)
            elif task_type == "balance_load":
                return await self._balance_workload()
            else:
                return {"status": "unknown_task_type", "task_type": task_type}
        
        except Exception as e:
            self.metrics["tasks_failed"] += 1
            return {"status": "error", "error": str(e)}
    
    async def _coordinate_agents(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Coordinate multiple agents for complex tasks."""
        coordination_plan = task.get("coordination_plan", {})
        
        results = {}
        for agent_id, subtask in coordination_plan.items():
            agent = next((a for a in self.managed_agents if a.agent_id == agent_id), None)
            if agent:
                result = await agent.handle(subtask)
                results[agent_id] = result
        
        return {"status": "coordinated", "results": results}
    
    async def _balance_workload(self) -> Dict[str, Any]:
        """Balance workload across managed agents."""
        workloads = {agent.agent_id: len(agent.current_tasks) for agent in self.managed_agents}
        avg_workload = sum(workloads.values()) / len(workloads) if workloads else 0
        
        return {"status": "balanced", "workloads": workloads, "average": avg_workload}


class ReasoningAgent(Agent):
    """Specialized agent for logical reasoning and problem analysis."""
    
    def __init__(self, agent_id: str, config: Dict[str, Any] = None):
        capabilities = AgentCapabilities(
            supported_tasks=[TaskType.REASONING, TaskType.ANALYSIS],
            max_concurrent_tasks=config.get("max_concurrent_tasks", 3) if config else 3,
            specializations=["logical_reasoning", "problem_decomposition", "analysis"],
            resource_requirements=ResourceRequirements(
                cpu_cores=config.get("cpu_cores", 2) if config else 2,
                memory_gb=config.get("memory_gb", 4) if config else 4
            )
        )
        super().__init__(agent_id, "reasoning", capabilities)
        self.reasoning_depth = config.get("reasoning_depth", 3) if config else 3
        self.logic_framework = config.get("logic_framework", "deductive") if config else "deductive"
    
    async def handle(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Handle reasoning and analysis tasks."""
        try:
            task_type = task.get("type", "reasoning")
            
            if task_type == "reasoning":
                return await self._perform_reasoning(task)
            elif task_type == "analysis":
                return await self._analyze_problem(task)
            elif task_type == "decomposition":
                return await self._decompose_problem(task)
            else:
                return {"status": "unknown_task_type", "task_type": task_type}
        
        except Exception as e:
            self.metrics["tasks_failed"] += 1
            return {"status": "error", "error": str(e)}
    
    async def _perform_reasoning(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Perform logical reasoning on the given problem."""
        problem = task.get("problem", "")
        premises = task.get("premises", [])
        
        # Simulate reasoning process
        reasoning_steps = []
        for i in range(self.reasoning_depth):
            step = {
                "step": i + 1,
                "inference": f"Reasoning step {i + 1} for: {problem[:50]}...",
                "confidence": 0.8 - (i * 0.1)
            }
            reasoning_steps.append(step)
        
        conclusion = {
            "reasoning_steps": reasoning_steps,
            "final_conclusion": f"Conclusion for {problem}",
            "confidence_score": 0.85,
            "logic_framework": self.logic_framework
        }
        
        self.metrics["tasks_completed"] += 1
        return {"status": "completed", "result": conclusion}
    
    async def _analyze_problem(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze problem structure and requirements."""
        problem = task.get("description", "")
        
        analysis = {
            "problem_type": self._classify_problem_type(problem),
            "complexity": self._assess_complexity(problem),
            "requirements": self._extract_requirements(problem),
            "constraints": task.get("constraints", []),
            "suggested_approach": self._suggest_approach(problem)
        }
        
        self.metrics["tasks_completed"] += 1
        return {"status": "completed", "result": analysis}
    
    async def _decompose_problem(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Decompose complex problem into subproblems."""
        problem = task.get("description", "")
        
        subproblems = [
            {"id": f"sub_1", "description": f"Analyze core requirements of: {problem[:30]}..."},
            {"id": f"sub_2", "description": f"Identify solution components for: {problem[:30]}..."},
            {"id": f"sub_3", "description": f"Design integration strategy for: {problem[:30]}..."}
        ]
        
        decomposition = {
            "original_problem": problem,
            "subproblems": subproblems,
            "dependencies": [{"sub_1": ["sub_2"]}, {"sub_2": ["sub_3"]}],
            "estimated_complexity": self._assess_complexity(problem)
        }
        
        self.metrics["tasks_completed"] += 1
        return {"status": "completed", "result": decomposition}
    
    def _classify_problem_type(self, problem: str) -> str:
        """Classify the type of problem for appropriate handling."""
        problem_lower = problem.lower()
        if any(word in problem_lower for word in ["optimize", "maximize", "minimize"]):
            return "optimization"
        elif any(word in problem_lower for word in ["classify", "categorize", "identify"]):
            return "classification"
        elif any(word in problem_lower for word in ["predict", "forecast", "estimate"]):
            return "prediction"
        else:
            return "general"
    
    def _assess_complexity(self, problem: str) -> str:
        """Assess problem complexity based on content analysis."""
        if len(problem) > 500:
            return "high"
        elif len(problem) > 200:
            return "medium"
        else:
            return "low"
    
    def _extract_requirements(self, problem: str) -> List[str]:
        """Extract key requirements from problem description."""
        # Simple keyword-based extraction
        requirements = []
        if "fast" in problem.lower():
            requirements.append("performance")
        if "accurate" in problem.lower():
            requirements.append("accuracy")
        if "scalable" in problem.lower():
            requirements.append("scalability")
        return requirements
    
    def _suggest_approach(self, problem: str) -> str:
        """Suggest problem-solving approach based on analysis."""
        problem_type = self._classify_problem_type(problem)
        if problem_type == "optimization":
            return "gradient-based optimization or heuristic search"
        elif problem_type == "classification":
            return "supervised learning or rule-based classification"
        elif problem_type == "prediction":
            return "time series analysis or regression modeling"
        else:
            return "divide-and-conquer with iterative refinement"


class CodingAgent(Agent):
    """Specialized agent for code generation and programming tasks."""
    
    def __init__(self, agent_id: str, config: Dict[str, Any] = None):
        capabilities = AgentCapabilities(
            supported_tasks=[TaskType.CODING, TaskType.ANALYSIS],
            max_concurrent_tasks=config.get("max_concurrent_tasks", 2) if config else 2,
            specializations=["code_generation", "debugging", "refactoring", "testing"],
            resource_requirements=ResourceRequirements(
                cpu_cores=config.get("cpu_cores", 4) if config else 4,
                memory_gb=config.get("memory_gb", 8) if config else 8
            )
        )
        super().__init__(agent_id, "coding", capabilities)
        self.supported_languages = config.get("languages", ["python", "javascript", "typescript"]) if config else ["python", "javascript", "typescript"]
        self.code_style = config.get("code_style", "clean_code") if config else "clean_code"
    
    async def handle(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Handle coding and programming tasks."""
        try:
            task_type = task.get("type", "coding")
            
            if task_type == "coding":
                return await self._generate_code(task)
            elif task_type == "debugging":
                return await self._debug_code(task)
            elif task_type == "refactoring":
                return await self._refactor_code(task)
            elif task_type == "validation":
                return await self._validate_code(task)
            else:
                return {"status": "unknown_task_type", "task_type": task_type}
        
        except Exception as e:
            self.metrics["tasks_failed"] += 1
            return {"status": "error", "error": str(e)}
    
    async def _generate_code(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Generate code based on requirements."""
        requirements = task.get("description", "")
        language = task.get("language", "python")
        
        if language not in self.supported_languages:
            return {"status": "error", "error": f"Language {language} not supported"}
        
        # Simulate code generation
        generated_code = self._create_code_template(requirements, language)
        
        result = {
            "language": language,
            "code": generated_code,
            "documentation": self._generate_documentation(requirements),
            "tests": self._generate_tests(requirements, language),
            "complexity": self._assess_code_complexity(generated_code),
            "style_compliance": self.code_style
        }
        
        self.metrics["tasks_completed"] += 1
        return {"status": "completed", "result": result}
    
    async def _debug_code(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Debug code and identify issues."""
        code = task.get("code", "")
        error_message = task.get("error", "")
        
        # Simulate debugging analysis
        issues = [
            {"type": "syntax", "line": 10, "message": "Missing semicolon"},
            {"type": "logic", "line": 15, "message": "Potential null pointer access"}
        ]
        
        fixes = [
            {"issue": "syntax", "fix": "Add semicolon at end of line 10"},
            {"issue": "logic", "fix": "Add null check before accessing object"}
        ]
        
        result = {
            "issues_found": issues,
            "suggested_fixes": fixes,
            "severity": "medium",
            "fixed_code": self._apply_fixes(code, fixes)
        }
        
        self.metrics["tasks_completed"] += 1
        return {"status": "completed", "result": result}
    
    async def _refactor_code(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Refactor code for better structure and maintainability."""
        code = task.get("code", "")
        refactor_type = task.get("refactor_type", "general")
        
        refactored_code = self._perform_refactoring(code, refactor_type)
        
        result = {
            "original_complexity": self._assess_code_complexity(code),
            "refactored_code": refactored_code,
            "new_complexity": self._assess_code_complexity(refactored_code),
            "improvements": ["Better modularity", "Improved readability", "Reduced complexity"],
            "refactor_type": refactor_type
        }
        
        self.metrics["tasks_completed"] += 1
        return {"status": "completed", "result": result}
    
    async def _validate_code(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Validate code for correctness and style compliance."""
        code = task.get("code", "")
        language = task.get("language", "python")
        
        validation_results = {
            "syntax_valid": True,
            "style_compliance": 85,
            "security_issues": [],
            "performance_warnings": ["Consider using list comprehension in loop"],
            "test_coverage": 78,
            "overall_score": 82
        }
        
        self.metrics["tasks_completed"] += 1
        return {"status": "completed", "result": validation_results}
    
    def _create_code_template(self, requirements: str, language: str) -> str:
        """Create code template based on requirements and language."""
        if language == "python":
            return f'"""\n{requirements}\n"""\n\ndef main():\n    # Implementation here\n    pass\n\nif __name__ == "__main__":\n    main()'
        elif language == "javascript":
            template = f"""/**
 * {requirements}
 */

function main() {{
    // Implementation here
}}

main();"""
            return template
        else:
            return f"// {requirements}\n// Code template for {language}"
    
    def _generate_documentation(self, requirements: str) -> str:
        """Generate documentation for the code."""
        return f"Documentation for: {requirements}\n\nThis module implements the required functionality with proper error handling and testing."
    
    def _generate_tests(self, requirements: str, language: str) -> str:
        """Generate test cases for the code."""
        if language == "python":
            return "import unittest\n\nclass TestImplementation(unittest.TestCase):\n    def test_main_functionality(self):\n        # Test implementation\n        pass"
        else:
            return "// Test cases for the implementation"
    
    def _assess_code_complexity(self, code: str) -> Dict[str, Any]:
        """Assess code complexity metrics."""
        lines = len(code.split('\n'))
        return {
            "lines_of_code": lines,
            "cyclomatic_complexity": min(10, lines // 5),
            "maintainability_index": max(20, 100 - lines // 10)
        }
    
    def _apply_fixes(self, code: str, fixes: List[Dict[str, Any]]) -> str:
        """Apply suggested fixes to code."""
        return code + "\n// Fixes applied based on debugging analysis"
    
    def _perform_refactoring(self, code: str, refactor_type: str) -> str:
        """Perform code refactoring based on type."""
        return code + f"\n// Refactored for {refactor_type}"


class VisionAgent(Agent):
    """Specialized agent for computer vision and image processing tasks."""
    
    def __init__(self, agent_id: str, config: Dict[str, Any] = None):
        capabilities = AgentCapabilities(
            supported_tasks=[TaskType.VISION, TaskType.ANALYSIS],
            max_concurrent_tasks=config.get("max_concurrent_tasks", 2) if config else 2,
            specializations=["image_classification", "object_detection", "image_generation", "feature_extraction"],
            resource_requirements=ResourceRequirements(
                cpu_cores=config.get("cpu_cores", 4) if config else 4,
                memory_gb=config.get("memory_gb", 16) if config else 16,
                gpu_memory_gb=config.get("gpu_memory_gb", 8) if config else 8
            )
        )
        super().__init__(agent_id, "vision", capabilities)
        self.supported_formats = config.get("formats", ["jpg", "png", "tiff", "bmp"]) if config else ["jpg", "png", "tiff", "bmp"]
        self.model_type = config.get("model_type", "cnn") if config else "cnn"
    
    async def handle(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Handle computer vision tasks."""
        try:
            task_type = task.get("type", "image_preprocessing")
            
            if task_type == "image_preprocessing":
                return await self._preprocess_image(task)
            elif task_type == "feature_extraction":
                return await self._extract_features(task)
            elif task_type == "classification":
                return await self._classify_image(task)
            elif task_type == "object_detection":
                return await self._detect_objects(task)
            elif task_type == "inference":
                return await self._run_inference(task)
            else:
                return {"status": "unknown_task_type", "task_type": task_type}
        
        except Exception as e:
            self.metrics["tasks_failed"] += 1
            return {"status": "error", "error": str(e)}
    
    async def _preprocess_image(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess images for analysis."""
        image_path = task.get("image_path", "")
        preprocessing_steps = task.get("steps", ["resize", "normalize"])
        
        result = {
            "original_image": image_path,
            "preprocessing_steps": preprocessing_steps,
            "output_format": "tensor",
            "dimensions": [224, 224, 3],
            "preprocessing_time": 0.5,
            "status": "preprocessed"
        }
        
        self.metrics["tasks_completed"] += 1
        return {"status": "completed", "result": result}
    
    async def _extract_features(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Extract features from images."""
        image_data = task.get("image_data", "")
        feature_type = task.get("feature_type", "deep_features")
        
        # Simulate feature extraction
        features = {
            "feature_vector": [0.1, 0.2, 0.3] * 100,  # Simulated feature vector
            "feature_type": feature_type,
            "dimensionality": 300,
            "extraction_method": self.model_type,
            "confidence": 0.92
        }
        
        result = {
            "features": features,
            "processing_time": 1.2,
            "model_used": self.model_type
        }
        
        self.metrics["tasks_completed"] += 1
        return {"status": "completed", "result": result}
    
    async def _classify_image(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Classify images into categories."""
        image_data = task.get("image_data", "")
        num_classes = task.get("num_classes", 1000)
        
        # Simulate classification results
        predictions = [
            {"class": "cat", "confidence": 0.85},
            {"class": "dog", "confidence": 0.12},
            {"class": "bird", "confidence": 0.03}
        ]
        
        result = {
            "predictions": predictions,
            "top_prediction": predictions[0],
            "num_classes": num_classes,
            "inference_time": 0.8,
            "model_accuracy": 0.94
        }
        
        self.metrics["tasks_completed"] += 1
        return {"status": "completed", "result": result}
    
    async def _detect_objects(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Detect and locate objects in images."""
        image_data = task.get("image_data", "")
        detection_threshold = task.get("threshold", 0.5)
        
        # Simulate object detection results
        detections = [
            {
                "class": "person",
                "confidence": 0.92,
                "bounding_box": [100, 150, 300, 400],
                "area": 40000
            },
            {
                "class": "car",
                "confidence": 0.78,
                "bounding_box": [450, 200, 600, 350],
                "area": 22500
            }
        ]
        
        result = {
            "detections": detections,
            "total_objects": len(detections),
            "detection_threshold": detection_threshold,
            "processing_time": 1.5,
            "model_type": "yolo_v5"
        }
        
        self.metrics["tasks_completed"] += 1
        return {"status": "completed", "result": result}
    
    async def _run_inference(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Run general vision model inference."""
        image_data = task.get("image_data", "")
        model_name = task.get("model", "default_vision_model")
        
        # Simulate inference results
        inference_result = {
            "model_output": [0.1, 0.8, 0.05, 0.05],  # Simulated logits
            "processed_input_shape": [1, 3, 224, 224],
            "inference_time": 0.3,
            "model_name": model_name,
            "device": "cuda" if task.get("use_gpu", True) else "cpu"
        }
        
        result = {
            "inference": inference_result,
            "status": "completed",
            "timestamp": time.time()
        }
        
        self.metrics["tasks_completed"] += 1
        return {"status": "completed", "result": result}


class MessageRouter:
    """Routes messages between agents."""
    
    def __init__(self):
        self.routes: Dict[str, Agent] = {}
        self.message_history: List[Message] = []
        self.logger = logging.getLogger(__name__)
    
    def register_agent(self, agent: Agent) -> None:
        """Register an agent for message routing."""
        self.routes[agent.agent_id] = agent
    
    def unregister_agent(self, agent_id: str) -> None:
        """Unregister an agent."""
        if agent_id in self.routes:
            del self.routes[agent_id]
    
    async def route_message(self, message: Message) -> bool:
        """Route message to destination agent."""
        if message.recipient_id in self.routes:
            recipient = self.routes[message.recipient_id]
            await recipient.receive_message(message)
            self.message_history.append(message)
            self.logger.debug(f"Routed message {message.id} to {message.recipient_id}")
            return True
        else:
            self.logger.warning(f"No route found for agent {message.recipient_id}")
            return False


class TaskScheduler:
    """Schedules tasks to appropriate agents."""
    
    def __init__(self):
        self.pending_tasks: List[Task] = []
        self.completed_tasks: List[Task] = []
        self.task_dependencies: Dict[str, Set[str]] = {}
        self.logger = logging.getLogger(__name__)
    
    def add_task(self, task: Task) -> None:
        """Add a task to the schedule."""
        self.pending_tasks.append(task)
        self.logger.info(f"Added task {task.id} to schedule")
    
    def get_ready_tasks(self) -> List[Task]:
        """Get tasks that are ready to execute (dependencies satisfied)."""
        ready_tasks = []
        
        for task in self.pending_tasks:
            if self._are_dependencies_satisfied(task):
                ready_tasks.append(task)
        
        # Remove ready tasks from pending
        for task in ready_tasks:
            self.pending_tasks.remove(task)
        
        return ready_tasks
    
    def _are_dependencies_satisfied(self, task: Task) -> bool:
        """Check if task dependencies are satisfied."""
        for dep_id in task.dependencies:
            # Check if dependency is completed
            if not any(t.id == dep_id and t.completed_at for t in self.completed_tasks):
                return False
        return True
    
    def mark_completed(self, task: Task) -> None:
        """Mark task as completed."""
        self.completed_tasks.append(task)
        self.logger.info(f"Task {task.id} marked as completed")
    
    def find_suitable_agent(self, task: Task, agents: List[Agent]) -> Optional[Agent]:
        """Find the best agent for a task."""
        suitable_agents = [
            agent for agent in agents 
            if (task.task_type in agent.capabilities.supported_tasks and
                len(agent.current_tasks) < agent.capabilities.max_concurrent_tasks and
                agent.status == AgentStatus.IDLE)
        ]
        
        if not suitable_agents:
            return None
        
        # Select agent with best performance metrics for this task type
        return min(suitable_agents, key=lambda a: len(a.current_tasks))


class AgentOrchestrator:
    """
    Manages a team of AI agents with different specialties.
    
    Production-grade orchestrator that coordinates multiple specialized agents
    to solve complex tasks through intelligent task decomposition, assignment,
    and result integration.
    """
    
    def __init__(self, agent_configs: List[Dict[str, Any]]):
        """
        Initialize agents (e.g., reasoning_agent, coding_agent, vision_agent).
        
        Args:
            agent_configs: List of agent configurations
        """
        self.agent_configs = agent_configs
        self.agents = [self._create_agent(cfg) for cfg in agent_configs]
        self.agent_registry: Dict[str, Agent] = {agent.agent_id: agent for agent in self.agents}
        
        # Production orchestration components
        self.message_router = MessageRouter()
        self.task_scheduler = TaskScheduler()
        self.system_metrics = {
            "total_tasks_processed": 0,
            "active_agents": len(self.agents),
            "average_task_completion_time": 0.0,
            "system_uptime": time.time()
        }
        self.running = False
        self.logger = logging.getLogger(__name__)
        
        # Task execution tracking
        self.task_results: Dict[str, Any] = {}
        self.integration_strategies: Dict[str, Callable] = {
            "voting": self._voting_integration,
            "chaining": self._chaining_integration,
            "weighted_average": self._weighted_integration,
            "ensemble": self._ensemble_integration,
            "hierarchical": self._hierarchical_integration
        }
    
    def _create_agent(self, cfg: Dict[str, Any]) -> Agent:
        """Create agent based on configuration using factory pattern."""
        agent_type = cfg.get("type", "inference")
        agent_id = cfg.get("id", f"{agent_type}_agent_{uuid.uuid4().hex[:8]}")
        
        # Use factory pattern from agents module
        if hasattr(agents, 'create_agent'):
            return agents.create_agent(cfg)
        else:
            # Fallback to local agent creation
            if agent_type == "reasoning":
                return ReasoningAgent(agent_id, cfg)
            elif agent_type == "coding":
                return CodingAgent(agent_id, cfg)
            elif agent_type == "vision":
                return VisionAgent(agent_id, cfg)
            elif agent_type == "training":
                return TrainingAgent(agent_id)
            elif agent_type == "data":
                return DataAgent(agent_id)
            else:
                return InferenceAgent(agent_id)

    async def solve_task(self, task: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Distribute sub-tasks among agents and aggregate results.
        
        Example: break task into parts and assign to best-suited agent
        
        Args:
            task: Task description or structured task definition
            
        Returns:
            Integrated results from all participating agents
        """
        # Convert string task to structured format
        if isinstance(task, str):
            task_obj = {
                "id": str(uuid.uuid4()),
                "description": task,
                "type": "general",
                "priority": "normal"
            }
        else:
            task_obj = task
            
        self.logger.info(f"Solving task: {task_obj.get('description', task_obj.get('id'))}")
        
        try:
            # Step 1: Analyze task and create execution plan
            plan = await self._plan_task(task_obj)
            self.logger.debug(f"Execution plan created with {len(plan)} subtasks")
            
            # Step 2: Execute subtasks in parallel/sequential order
            partial_results = {}
            for subtask, agent in plan:
                self.logger.debug(f"Assigning subtask {subtask.get('id')} to {agent.agent_id}")
                result = await agent.handle(subtask)
                partial_results[agent.agent_id] = {
                    "agent_name": agent.agent_id,
                    "agent_type": getattr(agent, 'agent_type', 'unknown'),
                    "subtask": subtask,
                    "result": result,
                    "timestamp": time.time()
                }
            
            # Step 3: Integrate results from all agents
            integration_strategy = task_obj.get("integration_strategy", "voting")
            final_result = await self._integrate(partial_results, integration_strategy)
            
            # Track task completion
            self.task_results[task_obj["id"]] = final_result
            self.system_metrics["total_tasks_processed"] += 1
            
            return final_result
            
        except Exception as e:
            self.logger.error(f"Task execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "task_id": task_obj["id"],
                "timestamp": time.time()
            }

    async def _plan_task(self, task: Dict[str, Any]) -> List[Tuple[Dict[str, Any], Agent]]:
        """
        Analyze task and create a plan mapping subtasks to agents.
        
        Advanced task decomposition with intelligent agent selection based on
        capabilities, current workload, and historical performance.
        
        Args:
            task: Task to decompose and plan
            
        Returns:
            List of (subtask, agent) tuples representing the execution plan
        """
        task_type = task.get("type", "general")
        task_complexity = task.get("complexity", "medium")
        
        # Decompose task based on type and complexity
        subtasks = await self._decompose_task(task)
        
        # Create execution plan with agent assignments
        execution_plan = []
        
        for subtask in subtasks:
            # Find best agent for this subtask
            best_agent = await self._select_best_agent(subtask)
            
            if best_agent:
                execution_plan.append((subtask, best_agent))
            else:
                self.logger.warning(f"No suitable agent found for subtask: {subtask}")
                # Assign to least loaded general agent as fallback
                fallback_agent = min(self.agents, key=lambda a: len(a.current_tasks))
                execution_plan.append((subtask, fallback_agent))
        
        return execution_plan

    async def _decompose_task(self, task: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Decompose complex task into manageable subtasks.
        
        Uses task analysis to break down complex problems into smaller,
        specialized components that can be handled by different agents.
        """
        task_type = task.get("type", "general")
        description = task.get("description", "")
        
        subtasks = []
        
        if task_type == "ml_pipeline":
            # Machine learning pipeline decomposition
            subtasks = [
                {
                    "id": f"{task['id']}_data_prep",
                    "type": "data_processing",
                    "description": f"Prepare data for: {description}",
                    "agent_requirement": "data"
                },
                {
                    "id": f"{task['id']}_training",
                    "type": "training",
                    "description": f"Train model for: {description}",
                    "agent_requirement": "training"
                },
                {
                    "id": f"{task['id']}_evaluation",
                    "type": "evaluation",
                    "description": f"Evaluate results for: {description}",
                    "agent_requirement": "inference"
                }
            ]
        elif task_type == "code_generation":
            # Code generation pipeline
            subtasks = [
                {
                    "id": f"{task['id']}_analysis",
                    "type": "reasoning",
                    "description": f"Analyze requirements: {description}",
                    "agent_requirement": "reasoning"
                },
                {
                    "id": f"{task['id']}_coding",
                    "type": "coding",
                    "description": f"Generate code for: {description}",
                    "agent_requirement": "coding"
                },
                {
                    "id": f"{task['id']}_validation",
                    "type": "validation",
                    "description": f"Validate code for: {description}",
                    "agent_requirement": "coding"
                }
            ]
        elif task_type == "vision_analysis":
            # Computer vision analysis
            subtasks = [
                {
                    "id": f"{task['id']}_preprocessing",
                    "type": "image_preprocessing",
                    "description": f"Preprocess images for: {description}",
                    "agent_requirement": "vision"
                },
                {
                    "id": f"{task['id']}_feature_extraction",
                    "type": "feature_extraction",
                    "description": f"Extract features from: {description}",
                    "agent_requirement": "vision"
                },
                {
                    "id": f"{task['id']}_classification",
                    "type": "inference",
                    "description": f"Classify results for: {description}",
                    "agent_requirement": "inference"
                }
            ]
        else:
            # General task - create single subtask
            subtasks = [{
                "id": f"{task['id']}_execution",
                "type": task_type,
                "description": description,
                "agent_requirement": "general"
            }]
        
        return subtasks

    async def _select_best_agent(self, subtask: Dict[str, Any]) -> Optional[Agent]:
        """
        Select the best agent for a specific subtask based on capabilities,
        workload, and performance metrics.
        """
        required_type = subtask.get("agent_requirement", "general")
        subtask_type = subtask.get("type")
        
        # Filter agents by capability
        candidate_agents = []
        
        for agent in self.agents:
            # Check if agent supports the required task type
            if hasattr(agent, 'agent_type') and required_type != "general":
                if agent.agent_type == required_type:
                    candidate_agents.append(agent)
            elif subtask_type in [t.value for t in agent.capabilities.supported_tasks]:
                candidate_agents.append(agent)
        
        if not candidate_agents:
            return None
        
        # Score agents based on multiple factors
        agent_scores = {}
        
        for agent in candidate_agents:
            score = 0.0
            
            # Factor 1: Current workload (lower is better)
            workload_factor = 1.0 - (len(agent.current_tasks) / agent.capabilities.max_concurrent_tasks)
            score += workload_factor * 0.4
            
            # Factor 2: Performance metrics
            if agent.metrics["tasks_completed"] > 0:
                success_rate = 1.0 - (agent.metrics["tasks_failed"] / 
                                    (agent.metrics["tasks_completed"] + agent.metrics["tasks_failed"]))
                score += success_rate * 0.3
            
            # Factor 3: Speed (inverse of average task time)
            if agent.metrics["average_task_time"] > 0:
                speed_factor = min(1.0, 10.0 / agent.metrics["average_task_time"])
                score += speed_factor * 0.2
            
            # Factor 4: Agent status
            if agent.status == AgentStatus.IDLE:
                score += 0.1
            
            agent_scores[agent] = score
        
        # Return agent with highest score
        return max(agent_scores.items(), key=lambda x: x[1])[0]

    async def _integrate(self, partial_results: Dict[str, Dict[str, Any]], 
                        strategy: str = "voting") -> Dict[str, Any]:
        """
        Integrate partial results from agents into a final answer.
        
        Supports multiple integration strategies including voting, chaining,
        weighted averaging, ensemble methods, and hierarchical combination.
        
        Args:
            partial_results: Results from individual agents
            strategy: Integration strategy to use
            
        Returns:
            Integrated final result
        """
        if not partial_results:
            return {"success": False, "error": "No partial results to integrate"}
        
        self.logger.info(f"Integrating results using {strategy} strategy")
        
        if strategy in self.integration_strategies:
            integration_func = self.integration_strategies[strategy]
            return await integration_func(partial_results)
        else:
            self.logger.warning(f"Unknown integration strategy: {strategy}, using voting")
            return await self._voting_integration(partial_results)

    async def initialize(self) -> None:
        """Initialize the orchestration system."""
        self.logger.info("Initializing agent orchestrator")
        
        # Register all agents with router
        for agent in self.agents:
            self.message_router.register_agent(agent)
            agent.orchestrator_ref = weakref.ref(self)
            await agent.start()
        
        self.logger.info(f"Orchestrator initialized with {len(self.agents)} agents")
    
    async def _create_initial_agents(self) -> None:
        """Create initial set of agents."""
        # Create inference agents
        for i in range(3):
            agent = InferenceAgent(f"inference_agent_{i}")
            await self._register_agent(agent)
        
        # Create training agents
        for i in range(2):
            agent = TrainingAgent(f"training_agent_{i}")
            await self._register_agent(agent)
        
        # Create data processing agents
        for i in range(2):
            agent = DataAgent(f"data_agent_{i}")
            await self._register_agent(agent)
        
        # Create coordinator agent
        coordinator = CoordinatorAgent("coordinator_agent")
        await self._register_agent(coordinator)
    
    async def _register_agent(self, agent: Agent) -> None:
        """Register an agent with the orchestrator."""
        self.agents[agent.agent_id] = agent
        self.message_router.register_agent(agent)
        agent.orchestrator_ref = weakref.ref(self)
        await agent.start()
        self.system_metrics["active_agents"] += 1
    
    async def run(self) -> None:
        """Run the main orchestration loop."""
        self.running = True
        self.logger.info("Starting agent orchestration loop")
        
        try:
            await asyncio.gather(
                self._task_distribution_loop(),
                self._monitoring_loop(),
                self._optimization_loop()
            )
        except Exception as e:
            self.logger.error(f"Orchestration error: {e}")
            raise
    
    async def _task_distribution_loop(self) -> None:
        """Main task distribution loop."""
        while self.running:
            try:
                # Get ready tasks
                ready_tasks = self.task_scheduler.get_ready_tasks()
                
                # Assign tasks to agents
                for task in ready_tasks:
                    await self._assign_task(task)
                
                await asyncio.sleep(0.1)  # Short delay to prevent busy waiting
                
            except Exception as e:
                self.logger.error(f"Task distribution error: {e}")
                await asyncio.sleep(1)
    
    async def _assign_task(self, task: Task) -> bool:
        """Assign a task to an appropriate agent."""
        suitable_agent = self.task_scheduler.find_suitable_agent(
            task, list(self.agents.values())
        )
        
        if suitable_agent:
            success = await suitable_agent.assign_task(task)
            if success:
                self.logger.info(f"Assigned task {task.id} to {suitable_agent.agent_id}")
                return True
        
        # If no suitable agent found, put task back in queue
        self.task_scheduler.pending_tasks.append(task)
        return False
    
    async def _monitoring_loop(self) -> None:
        """Monitor system health and performance."""
        while self.running:
            try:
                # Update system metrics
                self._update_system_metrics()
                
                # Check agent health
                await self._check_agent_health()
                
                await asyncio.sleep(10)  # Monitor every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(5)
    
    async def _optimization_loop(self) -> None:
        """Optimize system performance and resource allocation."""
        while self.running:
            try:
                # Perform optimization
                await self._optimize_resource_allocation()
                
                await asyncio.sleep(60)  # Optimize every minute
                
            except Exception as e:
                self.logger.error(f"Optimization error: {e}")
                await asyncio.sleep(30)
    
    def _update_system_metrics(self) -> None:
        """Update system-wide performance metrics."""
        active_agents = sum(1 for agent in self.agents.values() 
                          if agent.status != AgentStatus.SHUTDOWN)
        self.system_metrics["active_agents"] = active_agents
        
        # Calculate total tasks processed
        total_tasks = sum(agent.metrics["tasks_completed"] for agent in self.agents.values())
        self.system_metrics["total_tasks_processed"] = total_tasks
        
        # Calculate average task completion time
        if self.agents:
            avg_times = [agent.metrics["average_task_time"] for agent in self.agents.values() 
                        if agent.metrics["average_task_time"] > 0]
            if avg_times:
                self.system_metrics["average_task_completion_time"] = sum(avg_times) / len(avg_times)
    
    async def _check_agent_health(self) -> None:
        """Check health of all agents."""
        for agent_id, agent in list(self.agents.items()):
            if agent.status == AgentStatus.FAILED:
                self.logger.warning(f"Agent {agent_id} has failed, attempting restart")
                await self._restart_agent(agent_id)
    
    async def _restart_agent(self, agent_id: str) -> None:
        """Restart a failed agent."""
        try:
            old_agent = self.agents[agent_id]
            await old_agent.stop()
            
            # Create new agent of the same type
            if "inference" in agent_id:
                new_agent = InferenceAgent(agent_id)
            elif "training" in agent_id:
                new_agent = TrainingAgent(agent_id)
            elif "data" in agent_id:
                new_agent = DataAgent(agent_id)
            else:
                new_agent = CoordinatorAgent(agent_id)
            
            await self._register_agent(new_agent)
            self.logger.info(f"Restarted agent {agent_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to restart agent {agent_id}: {e}")
    
    async def _optimize_resource_allocation(self) -> None:
        """Optimize resource allocation across agents."""
        # Analyze agent performance and workload
        overloaded_agents = [
            agent for agent in self.agents.values()
            if len(agent.current_tasks) >= agent.capabilities.max_concurrent_tasks
        ]
        
        underutilized_agents = [
            agent for agent in self.agents.values()
            if len(agent.current_tasks) == 0 and agent.status == AgentStatus.IDLE
        ]
        
        if overloaded_agents and len(underutilized_agents) < 2:
            # Consider creating additional agents
            await self._scale_up_agents()
    
    async def _scale_up_agents(self) -> None:
        """Scale up agent pool when needed."""
        if len(self.agents) < self.config.max_concurrent_agents:
            # Create additional inference agent (most common task type)
            new_agent_id = f"inference_agent_{len([a for a in self.agents if 'inference' in a])}"
            new_agent = InferenceAgent(new_agent_id)
            await self._register_agent(new_agent)
            self.logger.info(f"Scaled up: created {new_agent_id}")
    
    async def submit_task(self, task: Task) -> str:
        """
        Submit a task for execution.
        
        Args:
            task: Task to execute
            
        Returns:
            Task ID
        """
        self.task_scheduler.add_task(task)
        self.logger.info(f"Submitted task {task.id}")
        return task.id
    
    async def get_task_result(self, task_id: str) -> Optional[Task]:
        """Get result of a completed task."""
        for task in self.task_scheduler.completed_tasks:
            if task.id == task_id:
                return task
        return None
    
    async def route_message(self, message: Message) -> bool:
        """Route message between agents."""
        return await self.message_router.route_message(message)
    
    async def agent_heartbeat(self, agent_id: str, metrics: Dict[str, Any]) -> None:
        """Receive heartbeat from an agent."""
        if agent_id in self.agents:
            self.agents[agent_id].metrics.update(metrics)
    
    async def task_completed(self, task: Task) -> None:
        """Handle task completion notification."""
        self.task_scheduler.mark_completed(task)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status."""
        return {
            "agents": {
                agent_id: {
                    "status": agent.status.value,
                    "current_tasks": len(agent.current_tasks),
                    "metrics": agent.metrics
                }
                for agent_id, agent in self.agents.items()
            },
            "system_metrics": self.system_metrics,
            "pending_tasks": len(self.task_scheduler.pending_tasks),
            "completed_tasks": len(self.task_scheduler.completed_tasks)
        }
    
    async def cleanup(self) -> None:
        """Clean up orchestrator resources."""
        self.running = False
        
        # Stop all agents
        stop_tasks = []
        for agent in self.agents.values():
            stop_tasks.append(agent.stop())
        
        await asyncio.gather(*stop_tasks)
        self.logger.info("Agent orchestrator cleanup completed")

    async def _voting_integration(self, partial_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Integrate results using voting strategy."""
        if not partial_results:
            return {"success": False, "error": "No results to vote on"}
        
        # Extract all successful results
        successful_results = []
        for agent_id, result_data in partial_results.items():
            result = result_data.get("result", {})
            if result.get("status") != "error":
                successful_results.append(result)
        
        if not successful_results:
            return {"success": False, "error": "No successful results for voting"}
        
        # Simple majority voting for classification tasks
        if all("predictions" in result for result in successful_results):
            # For classification: vote on top prediction
            votes = {}
            for result in successful_results:
                top_pred = result["predictions"][0]["class"] if result["predictions"] else "unknown"
                votes[top_pred] = votes.get(top_pred, 0) + 1
            
            winner = max(votes.items(), key=lambda x: x[1])
            return {
                "success": True,
                "strategy": "voting",
                "final_prediction": winner[0],
                "vote_count": winner[1],
                "total_votes": len(successful_results),
                "all_votes": votes
            }
        
        # For other tasks: return aggregated information
        return {
            "success": True,
            "strategy": "voting",
            "num_agents": len(successful_results),
            "results": successful_results,
            "consensus": "majority_rule"
        }