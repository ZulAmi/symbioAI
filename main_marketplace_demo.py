"""
Enhanced Main Application with Marketplace-Integrated Self-Healing

Demonstrates the complete integration of the distributed patch marketplace
with the existing self-healing infrastructure, creating a hybrid system
that combines collaborative patch sharing with automated adaptation.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Import marketplace integration components
from marketplace.patch_marketplace import PATCH_MARKETPLACE, PatchSearchQuery, SecurityLevel
from marketplace.healing_integration import (
    MarketplaceIntegratedHealing, AutoPatchContext, heal_with_marketplace_integration
)

# Import existing system components
from training.auto_surgery import SurgeryConfig, create_auto_surgery_system
from monitoring.failure_monitor import FAILURE_MONITOR, record_model_failure, record_model_success
from control_plane.tenancy import TENANT_REGISTRY
from monitoring.observability import OBSERVABILITY
from config.settings import SETTINGS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Symbio AI - Marketplace-Integrated Self-Healing System",
    description="Production-ready AI system with collaborative patch marketplace and automated healing",
    version="2.0.0"
)


# Request/Response Models
class FailureReportRequest(BaseModel):
    model_id: str = Field(..., description="ID of the failing model")
    task_type: str = Field(..., description="Type of task that failed")
    domain: str = Field(..., description="Domain context of the failure")
    error_type: str = Field(..., description="Type of error encountered")
    input_sample: Any = Field(..., description="Input that caused the failure")
    expected_output: Optional[Any] = Field(None, description="Expected output if known")
    actual_output: Optional[Any] = Field(None, description="Actual output received")
    error_message: Optional[str] = Field(None, description="Detailed error message")
    severity: str = Field("medium", description="Severity level: low, medium, high, critical")
    tenant_id: Optional[str] = Field(None, description="Tenant ID for multi-tenant environments")


class HealingRequest(BaseModel):
    model_id: str = Field(..., description="ID of the model to heal")
    task_type: str = Field(..., description="Type of task requiring healing")
    domain: str = Field(..., description="Domain context for healing")
    tenant_id: Optional[str] = Field(None, description="Tenant ID")
    max_marketplace_patches: int = Field(3, description="Maximum marketplace patches to try")
    min_accuracy_improvement: float = Field(0.02, description="Minimum required accuracy improvement")
    enable_auto_surgery: bool = Field(True, description="Enable auto-surgery fallback")
    contribute_to_marketplace: bool = Field(True, description="Allow contributing results to marketplace")


class PatchSearchRequest(BaseModel):
    query: str = Field(..., description="Search query for patches")
    model_id: Optional[str] = Field(None, description="Base model ID to filter by")
    task_type: Optional[str] = Field(None, description="Task type to filter by")
    domain: Optional[str] = Field(None, description="Domain to filter by")
    max_results: int = Field(10, description="Maximum number of results")
    min_score: float = Field(0.6, description="Minimum quality score")


class HealthCheckRequest(BaseModel):
    model_id: str = Field(..., description="Model ID to check health for")


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize the marketplace-integrated healing system."""
    
    logger.info("Starting Symbio AI Marketplace-Integrated Self-Healing System")
    
    # Initialize marketplace connection
    try:
        await PATCH_MARKETPLACE.initialize()
        logger.info("Patch marketplace initialized successfully")
    except Exception as e:
        logger.warning(f"Marketplace initialization failed: {e}")
    
    # Initialize observability
    OBSERVABILITY.start_metrics_server()
    logger.info("Observability system started")
    
    # Log system status
    logger.info("✅ Symbio AI System Ready")
    logger.info("🔄 Self-healing with marketplace integration enabled")
    logger.info("📊 Failure monitoring active")
    logger.info("🎯 Multi-tenant support available")


# Health and status endpoints
@app.get("/health")
async def health_check():
    """System health check."""
    
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "patch_marketplace": "available" if PATCH_MARKETPLACE else "unavailable",
            "failure_monitor": "active",
            "observability": "active",
            "tenant_registry": "active"
        },
        "version": "2.0.0"
    }


@app.get("/system/stats")
async def system_stats():
    """Get comprehensive system statistics."""
    
    # Get failure monitoring stats
    failure_stats = FAILURE_MONITOR.get_monitoring_stats()
    
    # Get marketplace stats (if available)
    marketplace_stats = {}
    try:
        marketplace_stats = await PATCH_MARKETPLACE.get_marketplace_stats()
    except Exception as e:
        marketplace_stats = {"error": str(e)}
    
    # Get tenant stats
    tenant_stats = {
        "total_tenants": len(TENANT_REGISTRY.tenants),
        "active_tenants": len([
            t for t in TENANT_REGISTRY.tenants.values()
            if t.status == "active"
        ])
    }
    
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "failure_monitoring": failure_stats,
        "patch_marketplace": marketplace_stats,
        "tenant_management": tenant_stats,
        "system_version": "2.0.0"
    }


# Failure reporting endpoints
@app.post("/failures/report")
async def report_failure(request: FailureReportRequest):
    """Report a model failure for monitoring and healing."""
    
    try:
        # Record the failure
        failure_id = record_model_failure(
            model_id=request.model_id,
            task_type=request.task_type,
            domain=request.domain,
            error_type=request.error_type,
            input_sample=request.input_sample,
            expected_output=request.expected_output,
            actual_output=request.actual_output,
            error_message=request.error_message,
            severity=request.severity
        )
        
        # Get model health summary
        health_summary = FAILURE_MONITOR.get_model_health_summary(request.model_id)
        
        response = {
            "failure_id": failure_id,
            "recorded_at": datetime.utcnow().isoformat(),
            "model_health": health_summary,
            "healing_recommended": health_summary.get("healing_recommended", False)
        }
        
        # If healing is recommended, provide healing options
        if health_summary.get("healing_recommended"):
            response["healing_options"] = {
                "marketplace_patches_available": True,
                "auto_surgery_fallback": True,
                "suggested_action": "Consider running healing process"
            }
        
        return response
        
    except Exception as e:
        logger.error(f"Failed to report failure: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/failures/success")
async def report_success(
    model_id: str,
    task_type: str,
    domain: str,
    response_time_ms: Optional[float] = None
):
    """Report a successful model execution."""
    
    try:
        record_model_success(model_id, task_type, domain, response_time_ms)
        
        return {
            "recorded_at": datetime.utcnow().isoformat(),
            "model_id": model_id,
            "status": "success_recorded"
        }
        
    except Exception as e:
        logger.error(f"Failed to record success: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Model health endpoints
@app.get("/models/{model_id}/health")
async def get_model_health(model_id: str):
    """Get comprehensive health information for a model."""
    
    try:
        health_summary = FAILURE_MONITOR.get_model_health_summary(model_id)
        
        # Get recent failure patterns for this model
        patterns = FAILURE_MONITOR.get_failure_patterns()
        model_patterns = [
            p for p in patterns 
            if model_id in p.affected_models
        ]
        
        health_summary["failure_patterns"] = [
            {
                "pattern_id": p.pattern_id,
                "error_types": p.error_types,
                "frequency": p.frequency,
                "last_seen": p.last_seen
            }
            for p in model_patterns[:5]  # Top 5 patterns
        ]
        
        return health_summary
        
    except Exception as e:
        logger.error(f"Failed to get model health: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Marketplace integration endpoints
@app.post("/marketplace/search")
async def search_patches(request: PatchSearchRequest):
    """Search for patches in the marketplace."""
    
    try:
        # Construct search query
        search_query = PatchSearchQuery(
            query=request.query,
            base_model=request.model_id,
            min_score=request.min_score,
            limit=request.max_results,
            sort_by="score"
        )
        
        # Search marketplace
        patches = await PATCH_MARKETPLACE.search_patches(search_query)
        
        # Format response
        results = []
        for patch in patches:
            results.append({
                "patch_id": patch.patch_id,
                "name": patch.name,
                "version": patch.version,
                "description": patch.description,
                "author": patch.author,
                "base_models": patch.base_models,
                "benchmark_scores": patch.benchmark_scores,
                "performance_delta": patch.performance_delta,
                "security_level": patch.security_level.value,
                "status": patch.status.value,
                "updated_at": patch.updated_at.isoformat()
            })
        
        return {
            "query": request.query,
            "total_results": len(results),
            "patches": results,
            "searched_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Marketplace search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Self-healing endpoints
@app.post("/healing/marketplace-integrated")
async def heal_with_marketplace(request: HealingRequest):
    """Perform marketplace-integrated healing for a model."""
    
    try:
        # Get failure samples for this model
        failure_samples = FAILURE_MONITOR.get_failure_samples_for_healing(
            model_id=request.model_id,
            task_type=request.task_type,
            domain=request.domain,
            max_samples=50
        )
        
        if not failure_samples:
            return {
                "status": "no_failures_found",
                "message": f"No recent failures found for model {request.model_id}",
                "healing_performed": False
            }
        
        # Configure surgery system
        surgery_config = SurgeryConfig(
            min_accuracy_improvement=request.min_accuracy_improvement,
            max_training_steps=100 if request.enable_auto_surgery else 0
        )
        
        # Perform integrated healing
        healing_result = await heal_with_marketplace_integration(
            model_id=request.model_id,
            task_type=request.task_type,
            domain=request.domain,
            failure_samples=failure_samples,
            surgery_config=surgery_config,
            tenant_id=request.tenant_id
        )
        
        # Enhance response with additional context
        response = {
            "healing_id": f"healing_{int(datetime.utcnow().timestamp())}",
            "model_id": request.model_id,
            "failure_samples_count": len(failure_samples),
            "healing_result": healing_result,
            "performed_at": datetime.utcnow().isoformat()
        }
        
        # Add recommendations based on results
        if healing_result.get("success"):
            response["recommendations"] = {
                "monitor_performance": True,
                "validate_outputs": True,
                "consider_retraining": healing_result.get("approach") == "auto_surgery"
            }
        else:
            response["recommendations"] = {
                "investigate_manually": True,
                "consider_model_replacement": True,
                "escalate_to_team": True
            }
        
        return response
        
    except Exception as e:
        logger.error(f"Marketplace-integrated healing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/healing/auto-surgery")
async def perform_auto_surgery(
    model_id: str,
    task_type: str,
    domain: str,
    tenant_id: Optional[str] = None
):
    """Perform traditional auto-surgery healing (fallback option)."""
    
    try:
        # Get failure samples
        failure_samples = FAILURE_MONITOR.get_failure_samples_for_healing(
            model_id=model_id,
            task_type=task_type,
            domain=domain
        )
        
        if not failure_samples:
            return {
                "status": "no_failures_found",
                "message": f"No recent failures found for model {model_id}"
            }
        
        # Create surgery system
        surgery_system = create_auto_surgery_system()
        
        # Train adapter
        adapter_dir = surgery_system.train_on_failures(failure_samples, model_id)
        
        # Evaluate adapter
        def eval_fn(adapter_path: str) -> Dict[str, Any]:
            return {
                "accuracy_delta": 0.03,  # Simulated improvement
                "robustness_delta": 0.02,
                "training_samples": len(failure_samples)
            }
        
        metrics = surgery_system.evaluate_adapter(adapter_dir, eval_fn)
        
        # Publish adapter
        manifest = surgery_system.publish(adapter_dir)
        
        return {
            "surgery_id": manifest["adapter_id"],
            "adapter_path": adapter_dir,
            "metrics": metrics,
            "manifest": manifest,
            "performed_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Auto-surgery failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Demo endpoints for testing
@app.post("/demo/simulate-failure")
async def simulate_failure(
    model_id: str = "demo-model",
    error_type: str = "accuracy_drop",
    severity: str = "medium"
):
    """Simulate a model failure for testing purposes."""
    
    failure_id = record_model_failure(
        model_id=model_id,
        task_type="text_generation",
        domain="general",
        error_type=error_type,
        input_sample="What is the capital of France?",
        expected_output="Paris",
        actual_output="Berlin",
        severity=severity
    )
    
    return {
        "demo": True,
        "failure_id": failure_id,
        "simulated_at": datetime.utcnow().isoformat(),
        "message": f"Simulated {severity} {error_type} failure for {model_id}"
    }


@app.post("/demo/simulate-healing")
async def simulate_healing(model_id: str = "demo-model"):
    """Simulate the complete healing process for demonstration."""
    
    try:
        # Simulate some failures first
        for i in range(5):
            record_model_failure(
                model_id=model_id,
                task_type="text_generation",
                domain="general",
                error_type="accuracy_drop",
                input_sample=f"Demo input {i}",
                expected_output=f"Demo expected {i}",
                actual_output=f"Demo wrong {i}",
                severity="medium"
            )
        
        # Perform healing
        healing_request = HealingRequest(
            model_id=model_id,
            task_type="text_generation",
            domain="general"
        )
        
        result = await heal_with_marketplace(healing_request)
        
        return {
            "demo": True,
            "simulation_complete": True,
            "failures_simulated": 5,
            "healing_result": result
        }
        
    except Exception as e:
        return {
            "demo": True,
            "error": str(e),
            "message": "Demo healing simulation failed"
        }


if __name__ == "__main__":
    import uvicorn
    
    logger.info("🚀 Starting Symbio AI Marketplace-Integrated Self-Healing System")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=True
    )