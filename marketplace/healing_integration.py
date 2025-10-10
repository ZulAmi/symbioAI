"""
Patch Marketplace Integration for Self-Healing Pipeline

Integrates the distributed patch marketplace with the existing self-healing
infrastructure, enabling automatic patch discovery, evaluation, and deployment
based on failure patterns and performance requirements.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass
from datetime import datetime, timedelta

from marketplace.patch_marketplace import (
    PATCH_MARKETPLACE, PatchManifest, PatchType, PatchSearchQuery,
    SecurityLevel, PatchStatus
)
from training.auto_surgery import AutoModelSurgery, SurgeryConfig
from monitoring.failure_monitor import FailureMonitor
from control_plane.tenancy import TENANT_REGISTRY
from monitoring.observability import OBSERVABILITY


@dataclass
class AutoPatchContext:
    """Context for automatic patch discovery and application."""
    
    model_id: str
    task_type: str
    domain: str
    failure_patterns: List[str]
    performance_requirements: Dict[str, float]
    tenant_id: Optional[str] = None
    security_constraints: Optional[SecurityLevel] = None
    max_patches: int = 3
    evaluation_required: bool = True


class MarketplaceIntegratedHealing:
    """
    Enhanced self-healing system that integrates with the patch marketplace
    for collaborative failure resolution and performance optimization.
    """
    
    def __init__(self, surgery_config: SurgeryConfig):
        self.surgery_config = surgery_config
        self.logger = logging.getLogger(__name__)
        
        # Integration components
        self.marketplace = PATCH_MARKETPLACE
        self.auto_surgery = AutoModelSurgery(surgery_config)
        
        # Patch evaluation and deployment
        self.patch_evaluator = PatchEvaluator()
        self.patch_deployer = PatchDeployer()
        
        # Success tracking
        self.patch_success_history: Dict[str, List[Dict[str, Any]]] = {}
        self.community_contributions: List[Dict[str, Any]] = []
    
    async def heal_with_marketplace(
        self,
        context: AutoPatchContext,
        failure_samples: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Attempt healing using marketplace patches before falling back to auto-surgery.
        """
        
        healing_result = {
            "approach": None,
            "patches_tried": [],
            "surgery_performed": False,
            "success": False,
            "performance_improvement": {},
            "community_contribution": None
        }
        
        try:
            # 1. Search for relevant patches in marketplace
            marketplace_patches = await self._discover_relevant_patches(context)
            
            if marketplace_patches:
                # 2. Try marketplace patches first
                patch_result = await self._try_marketplace_patches(
                    context, marketplace_patches, failure_samples
                )
                
                if patch_result["success"]:
                    healing_result.update(patch_result)
                    healing_result["approach"] = "marketplace_patch"
                    
                    # Contribute success data back to marketplace
                    await self._contribute_success_data(context, patch_result)
                    return healing_result
            
            # 3. Fall back to auto-surgery if no suitable patches found
            self.logger.info(f"No suitable marketplace patches found for {context.model_id}, falling back to auto-surgery")
            
            surgery_result = await self._perform_auto_surgery(context, failure_samples)
            healing_result.update(surgery_result)
            healing_result["approach"] = "auto_surgery"
            
            # 4. If auto-surgery succeeds, consider publishing to marketplace
            if surgery_result["success"]:
                contribution = await self._consider_marketplace_contribution(
                    context, surgery_result
                )
                healing_result["community_contribution"] = contribution
            
            return healing_result
            
        except Exception as e:
            self.logger.error(f"Healing failed for {context.model_id}: {e}")
            healing_result["error"] = str(e)
            return healing_result
    
    async def _discover_relevant_patches(
        self,
        context: AutoPatchContext
    ) -> List[PatchManifest]:
        """Discover relevant patches from the marketplace."""
        
        # Construct search query based on failure context
        search_query = PatchSearchQuery(
            query=f"{context.task_type} {context.domain}",
            patch_types=[PatchType.LORA_ADAPTER, PatchType.FINE_TUNE],
            base_model=context.model_id,
            min_score=0.6,
            max_age_days=180,
            security_level=context.security_constraints,
            has_evaluation=context.evaluation_required,
            sort_by="score",
            limit=context.max_patches * 2  # Get more candidates for filtering
        )
        
        # Search marketplace
        candidates = await self.marketplace.search_patches(search_query)
        
        # Filter by failure patterns and requirements
        relevant_patches = []
        for patch in candidates:
            if self._is_patch_relevant(patch, context):
                relevant_patches.append(patch)
        
        # Limit to max_patches
        return relevant_patches[:context.max_patches]
    
    def _is_patch_relevant(
        self,
        patch: PatchManifest,
        context: AutoPatchContext
    ) -> bool:
        """Check if a patch is relevant to the current failure context."""
        
        # Check base model compatibility
        if context.model_id not in patch.base_models and patch.base_models:
            return False
        
        # Check if patch addresses similar failure patterns
        patch_description = patch.description.lower()
        for pattern in context.failure_patterns:
            if pattern.lower() in patch_description:
                return True
        
        # Check performance requirements
        for req_metric, req_value in context.performance_requirements.items():
            if req_metric in patch.performance_delta:
                if patch.performance_delta[req_metric] >= req_value * 0.8:  # 80% of requirement
                    return True
        
        # Check security constraints
        if context.security_constraints:
            if patch.security_level.value < context.security_constraints.value:
                return False
        
        return True
    
    async def _try_marketplace_patches(
        self,
        context: AutoPatchContext,
        patches: List[PatchManifest],
        failure_samples: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Try applying marketplace patches in order of relevance."""
        
        result = {
            "success": False,
            "patches_tried": [],
            "best_patch": None,
            "performance_improvement": {},
            "evaluation_results": {}
        }
        
        for patch in patches:
            try:
                # Install patch
                installation_path = await self.marketplace.install_patch(
                    patch.patch_id,
                    tenant_id=context.tenant_id,
                    verify_signature=True
                )
                
                # Evaluate patch performance
                evaluation_result = await self.patch_evaluator.evaluate_patch_on_failures(
                    patch, failure_samples, context
                )
                
                patch_attempt = {
                    "patch_id": patch.patch_id,
                    "installation_path": installation_path,
                    "evaluation": evaluation_result
                }
                result["patches_tried"].append(patch_attempt)
                
                # Check if patch meets requirements
                if self._patch_meets_requirements(evaluation_result, context):
                    result["success"] = True
                    result["best_patch"] = patch.patch_id
                    result["performance_improvement"] = evaluation_result.get("performance_delta", {})
                    result["evaluation_results"] = evaluation_result
                    
                    # Emit success telemetry
                    OBSERVABILITY.emit_counter(
                        "healing.marketplace_patch_success",
                        1,
                        patch_id=patch.patch_id,
                        model_id=context.model_id,
                        task_type=context.task_type
                    )
                    
                    break
                    
            except Exception as e:
                self.logger.warning(f"Failed to apply patch {patch.patch_id}: {e}")
                result["patches_tried"].append({
                    "patch_id": patch.patch_id,
                    "error": str(e)
                })
        
        return result
    
    def _patch_meets_requirements(
        self,
        evaluation_result: Dict[str, Any],
        context: AutoPatchContext
    ) -> bool:
        """Check if patch evaluation meets performance requirements."""
        
        performance_delta = evaluation_result.get("performance_delta", {})
        
        for metric, required_improvement in context.performance_requirements.items():
            actual_improvement = performance_delta.get(metric, 0)
            if actual_improvement < required_improvement:
                return False
        
        # Check overall success rate
        success_rate = evaluation_result.get("success_rate", 0)
        if success_rate < 0.8:  # Require 80% success rate
            return False
        
        return True
    
    async def _perform_auto_surgery(
        self,
        context: AutoPatchContext,
        failure_samples: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Perform traditional auto-surgery as fallback."""
        
        try:
            # Train adapter on failures
            adapter_dir = self.auto_surgery.train_on_failures(failure_samples)
            
            # Evaluate adapter
            def eval_adapter(adapter_path: str) -> Dict[str, Any]:
                return self.patch_evaluator.evaluate_surgery_adapter(
                    adapter_path, failure_samples, context
                )
            
            metrics = self.auto_surgery.evaluate_adapter(adapter_dir, eval_adapter)
            
            # Publish adapter
            manifest = self.auto_surgery.publish(adapter_dir)
            
            return {
                "success": True,
                "surgery_performed": True,
                "adapter_path": adapter_dir,
                "metrics": metrics,
                "manifest": manifest
            }
            
        except Exception as e:
            self.logger.error(f"Auto-surgery failed: {e}")
            return {
                "success": False,
                "surgery_performed": True,
                "error": str(e)
            }
    
    async def _consider_marketplace_contribution(
        self,
        context: AutoPatchContext,
        surgery_result: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Consider contributing successful auto-surgery results to marketplace."""
        
        # Check if results are worth contributing
        metrics = surgery_result.get("metrics", {})
        if not self._is_contribution_worthy(metrics, context):
            return None
        
        try:
            # Create patch manifest from surgery results
            manifest = await self._create_patch_manifest_from_surgery(
                context, surgery_result
            )
            
            # Prepare artifacts for upload
            artifacts = {
                "adapter.safetensors": surgery_result["adapter_path"] + "/adapter_model.safetensors",
                "adapter_config.json": surgery_result["adapter_path"] + "/adapter_config.json",
                "training_args.json": surgery_result["adapter_path"] + "/training_args.json"
            }
            
            # Publish to marketplace
            patch_id = await self.marketplace.publish_patch(
                manifest, artifacts, auto_upload=True
            )
            
            contribution = {
                "patch_id": patch_id,
                "type": "auto_surgery_contribution",
                "manifest": manifest.to_dict(),
                "contributed_at": datetime.utcnow().isoformat()
            }
            
            self.community_contributions.append(contribution)
            
            # Emit telemetry
            OBSERVABILITY.emit_counter(
                "healing.community_contribution",
                1,
                patch_type="auto_surgery",
                model_id=context.model_id,
                task_type=context.task_type
            )
            
            self.logger.info(f"Contributed auto-surgery results to marketplace: {patch_id}")
            return contribution
            
        except Exception as e:
            self.logger.error(f"Failed to contribute to marketplace: {e}")
            return {"error": str(e)}
    
    def _is_contribution_worthy(
        self,
        metrics: Dict[str, Any],
        context: AutoPatchContext
    ) -> bool:
        """Check if surgery results are worthy of marketplace contribution."""
        
        # Check minimum quality thresholds
        accuracy_delta = metrics.get("accuracy_delta", 0)
        robustness_delta = metrics.get("robustness_delta", 0)
        
        if accuracy_delta < 0.02:  # Less than 2% improvement
            return False
        
        if robustness_delta < 0.01:  # Less than 1% robustness improvement
            return False
        
        # Check tenant policy for contributions
        if context.tenant_id:
            tenant = TENANT_REGISTRY.get_tenant(context.tenant_id)
            if tenant and "marketplace_contributions" not in tenant.enabled_policies:
                return False
        
        return True
    
    async def _create_patch_manifest_from_surgery(
        self,
        context: AutoPatchContext,
        surgery_result: Dict[str, Any]
    ) -> PatchManifest:
        """Create a patch manifest from auto-surgery results."""
        
        metrics = surgery_result.get("metrics", {})
        
        manifest = PatchManifest(
            patch_id=f"auto-surgery-{context.model_id}-{int(datetime.utcnow().timestamp())}",
            name=f"Auto-Surgery Adapter for {context.model_id}",
            version="1.0.0",
            patch_type=PatchType.LORA_ADAPTER,
            status=PatchStatus.PUBLISHED,
            security_level=SecurityLevel.COMMUNITY_VERIFIED,
            description=f"Automatically generated LoRA adapter addressing {', '.join(context.failure_patterns)} in {context.domain} tasks",
            author="symbio-ai-auto-surgery",
            organization="symbio-ai",
            base_models=[context.model_id],
            framework_versions={"transformers": ">=4.20.0", "peft": ">=0.4.0"},
            benchmark_scores={
                "accuracy_improvement": metrics.get("accuracy_delta", 0),
                "robustness_improvement": metrics.get("robustness_delta", 0)
            },
            performance_delta={
                context.task_type: metrics.get("accuracy_delta", 0),
                "robustness": metrics.get("robustness_delta", 0)
            },
            license="apache-2.0"
        )
        
        return manifest
    
    async def _contribute_success_data(
        self,
        context: AutoPatchContext,
        patch_result: Dict[str, Any]
    ) -> None:
        """Contribute success data back to the marketplace."""
        
        if not patch_result.get("best_patch"):
            return
        
        patch_id = patch_result["best_patch"]
        evaluation_results = patch_result.get("evaluation_results", {})
        
        # Contribute evaluation
        evaluator_id = f"symbio-ai-{context.tenant_id or 'system'}"
        await self.marketplace.contribute_evaluation(
            patch_id, evaluation_results, evaluator_id
        )
        
        # Track success history
        if patch_id not in self.patch_success_history:
            self.patch_success_history[patch_id] = []
        
        self.patch_success_history[patch_id].append({
            "context": context.__dict__,
            "results": evaluation_results,
            "timestamp": datetime.utcnow().isoformat()
        })
    
    def get_healing_stats(self) -> Dict[str, Any]:
        """Get comprehensive healing statistics."""
        
        return {
            "patch_success_history": {
                patch_id: len(history) 
                for patch_id, history in self.patch_success_history.items()
            },
            "community_contributions": len(self.community_contributions),
            "marketplace_integration": True,
            "auto_surgery_fallback": True,
            "recent_contributions": [
                contrib for contrib in self.community_contributions
                if datetime.fromisoformat(contrib["contributed_at"]) > 
                   datetime.utcnow() - timedelta(days=7)
            ]
        }


class PatchEvaluator:
    """Evaluates patches against failure samples and performance requirements."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    async def evaluate_patch_on_failures(
        self,
        patch: PatchManifest,
        failure_samples: List[Dict[str, Any]],
        context: AutoPatchContext
    ) -> Dict[str, Any]:
        """Evaluate how well a patch addresses failure samples."""
        
        # Simulate patch evaluation (in production, would load and test actual patch)
        success_count = 0
        total_samples = len(failure_samples)
        performance_improvements = {}
        
        for sample in failure_samples:
            # Simulate patch application and evaluation
            success = self._simulate_patch_evaluation(patch, sample, context)
            if success:
                success_count += 1
        
        success_rate = success_count / total_samples if total_samples > 0 else 0
        
        # Estimate performance delta based on patch metadata and success rate
        for metric, baseline_delta in patch.performance_delta.items():
            adjusted_delta = baseline_delta * success_rate
            performance_improvements[metric] = adjusted_delta
        
        return {
            "success_rate": success_rate,
            "samples_passed": success_count,
            "samples_total": total_samples,
            "performance_delta": performance_improvements,
            "patch_compatibility": success_rate > 0.7,
            "evaluation_timestamp": datetime.utcnow().isoformat()
        }
    
    def evaluate_surgery_adapter(
        self,
        adapter_path: str,
        failure_samples: List[Dict[str, Any]],
        context: AutoPatchContext
    ) -> Dict[str, Any]:
        """Evaluate auto-surgery adapter results."""
        
        # Simulate adapter evaluation
        base_accuracy = 0.7  # Baseline accuracy
        improved_accuracy = min(0.95, base_accuracy + 0.05)  # 5% improvement, capped at 95%
        
        return {
            "accuracy_delta": improved_accuracy - base_accuracy,
            "robustness_delta": 0.02,  # 2% robustness improvement
            "samples_improved": len(failure_samples),
            "adapter_size_mb": 2.5,
            "training_time_minutes": 15
        }
    
    def _simulate_patch_evaluation(
        self,
        patch: PatchManifest,
        sample: Dict[str, Any],
        context: AutoPatchContext
    ) -> bool:
        """Simulate evaluation of a patch on a failure sample."""
        
        # Simulate success based on patch quality and compatibility
        base_success_prob = 0.6
        
        # Adjust based on patch status
        if patch.status == PatchStatus.VERIFIED:
            base_success_prob += 0.2
        elif patch.status == PatchStatus.DEPRECATED:
            base_success_prob -= 0.3
        
        # Adjust based on community ratings
        if patch.community_ratings:
            avg_rating = sum(patch.community_ratings) / len(patch.community_ratings)
            rating_bonus = (avg_rating - 3.0) / 5.0 * 0.2  # Scale 1-5 rating to -0.4 to +0.4
            base_success_prob += rating_bonus
        
        # Adjust based on recency
        age_days = (datetime.utcnow() - patch.updated_at).days
        if age_days > 90:
            base_success_prob -= 0.1
        
        # Simple deterministic simulation based on patch and sample hash
        patch_hash = hash(patch.patch_id)
        sample_hash = hash(str(sample))
        combined_hash = (patch_hash + sample_hash) % 100
        
        return (combined_hash / 100.0) < base_success_prob


class PatchDeployer:
    """Handles deployment and integration of marketplace patches."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.deployed_patches: Dict[str, Dict[str, Any]] = {}
    
    async def deploy_patch(
        self,
        patch_id: str,
        installation_path: str,
        context: AutoPatchContext
    ) -> Dict[str, Any]:
        """Deploy a patch for production use."""
        
        deployment_info = {
            "patch_id": patch_id,
            "installation_path": installation_path,
            "model_id": context.model_id,
            "task_type": context.task_type,
            "domain": context.domain,
            "deployed_at": datetime.utcnow().isoformat(),
            "status": "active"
        }
        
        self.deployed_patches[patch_id] = deployment_info
        
        # Emit telemetry
        OBSERVABILITY.emit_counter(
            "healing.patch_deployed",
            1,
            patch_id=patch_id,
            model_id=context.model_id
        )
        
        self.logger.info(f"Deployed patch {patch_id} for model {context.model_id}")
        return deployment_info
    
    def get_deployed_patches(self) -> Dict[str, Dict[str, Any]]:
        """Get information about deployed patches."""
        return dict(self.deployed_patches)


# Integration function for existing self-healing pipeline
async def heal_with_marketplace_integration(
    model_id: str,
    task_type: str,
    domain: str,
    failure_samples: List[Dict[str, Any]],
    surgery_config: SurgeryConfig,
    tenant_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Enhanced healing function that integrates marketplace and auto-surgery.
    """
    
    # Create healing context
    context = AutoPatchContext(
        model_id=model_id,
        task_type=task_type,
        domain=domain,
        failure_patterns=_extract_failure_patterns(failure_samples),
        performance_requirements={"accuracy": 0.03, "robustness": 0.02},
        tenant_id=tenant_id,
        security_constraints=SecurityLevel.COMMUNITY_VERIFIED,
        evaluation_required=True
    )
    
    # Initialize integrated healing system
    healing_system = MarketplaceIntegratedHealing(surgery_config)
    
    # Perform healing with marketplace integration
    result = await healing_system.heal_with_marketplace(context, failure_samples)
    
    return result


def _extract_failure_patterns(failure_samples: List[Dict[str, Any]]) -> List[str]:
    """Extract failure patterns from samples."""
    
    patterns = []
    
    # Analyze common failure modes
    error_types = set()
    for sample in failure_samples:
        if "error_type" in sample:
            error_types.add(sample["error_type"])
    
    patterns.extend(list(error_types))
    
    # Add domain-specific patterns
    if len(failure_samples) > 10:
        patterns.append("high_failure_rate")
    
    return patterns or ["general_performance"]