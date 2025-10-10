"""
Distributed Patch Marketplace for Symbio AI

Extends the self-healing pipeline with collaborative patch sharing, discovery,
and evaluation. Integrates with Hugging Face Hub for patch storage and
metadata management while maintaining local governance and security.
"""

from __future__ import annotations

import json
import time
import hashlib
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum
import asyncio

try:
    from huggingface_hub import HfApi, ModelCard, DatasetCard, hf_hub_download, create_repo
    from huggingface_hub.utils import HfHubHTTPError
    HF_AVAILABLE = True
except ImportError:
    HfApi = ModelCard = DatasetCard = hf_hub_download = create_repo = None
    HfHubHTTPError = Exception
    HF_AVAILABLE = False

from registry.adapter_registry import ADAPTER_REGISTRY, AdapterMetadata
from control_plane.tenancy import TENANT_REGISTRY
from control_plane.policy_engine import DEFAULT_POLICY_ENGINE, PolicyContext, PolicyDecision
from monitoring.observability import OBSERVABILITY


class PatchType(Enum):
    """Types of patches available in the marketplace."""
    LORA_ADAPTER = "lora_adapter"
    FINE_TUNE = "fine_tune"
    KNOWLEDGE_DISTILL = "knowledge_distill"
    PROMPT_TEMPLATE = "prompt_template"
    ROUTING_RULE = "routing_rule"
    EVALUATION_METRIC = "evaluation_metric"
    PREPROCESSING = "preprocessing"


class PatchStatus(Enum):
    """Patch lifecycle status."""
    DRAFT = "draft"
    PUBLISHED = "published"
    VERIFIED = "verified"
    DEPRECATED = "deprecated"
    REVOKED = "revoked"


class SecurityLevel(Enum):
    """Security levels for patch evaluation."""
    PUBLIC = "public"
    COMMUNITY_VERIFIED = "community_verified"
    ENTERPRISE_APPROVED = "enterprise_approved"
    INTERNAL_ONLY = "internal_only"


@dataclass
class PatchManifest:
    """Patch manifest with metadata and verification."""
    
    patch_id: str
    name: str
    version: str
    patch_type: PatchType
    status: PatchStatus
    security_level: SecurityLevel
    
    # Metadata
    description: str
    author: str
    organization: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    # Dependencies and compatibility
    base_models: List[str] = field(default_factory=list)
    framework_versions: Dict[str, str] = field(default_factory=dict)
    python_requirements: List[str] = field(default_factory=list)
    
    # Performance metrics
    benchmark_scores: Dict[str, float] = field(default_factory=dict)
    evaluation_results: Dict[str, Any] = field(default_factory=dict)
    performance_delta: Dict[str, float] = field(default_factory=dict)
    
    # Distribution
    huggingface_repo: Optional[str] = None
    artifact_urls: Dict[str, str] = field(default_factory=dict)
    file_checksums: Dict[str, str] = field(default_factory=dict)
    
    # Usage and community
    downloads: int = 0
    stars: int = 0
    usage_count: int = 0
    community_ratings: List[float] = field(default_factory=list)
    
    # Security and verification
    signature: Optional[str] = None
    verification_hash: Optional[str] = None
    trusted_by: Set[str] = field(default_factory=set)
    
    # License and governance
    license: str = "apache-2.0"
    governance_policy: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "patch_id": self.patch_id,
            "name": self.name,
            "version": self.version,
            "patch_type": self.patch_type.value,
            "status": self.status.value,
            "security_level": self.security_level.value,
            "description": self.description,
            "author": self.author,
            "organization": self.organization,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "base_models": self.base_models,
            "framework_versions": self.framework_versions,
            "python_requirements": self.python_requirements,
            "benchmark_scores": self.benchmark_scores,
            "evaluation_results": self.evaluation_results,
            "performance_delta": self.performance_delta,
            "huggingface_repo": self.huggingface_repo,
            "artifact_urls": self.artifact_urls,
            "file_checksums": self.file_checksums,
            "downloads": self.downloads,
            "stars": self.stars,
            "usage_count": self.usage_count,
            "community_ratings": self.community_ratings,
            "signature": self.signature,
            "verification_hash": self.verification_hash,
            "trusted_by": list(self.trusted_by),
            "license": self.license,
            "governance_policy": self.governance_policy,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PatchManifest":
        """Create from dictionary."""
        return cls(
            patch_id=data["patch_id"],
            name=data["name"],
            version=data["version"],
            patch_type=PatchType(data["patch_type"]),
            status=PatchStatus(data["status"]),
            security_level=SecurityLevel(data["security_level"]),
            description=data["description"],
            author=data["author"],
            organization=data.get("organization"),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            base_models=data.get("base_models", []),
            framework_versions=data.get("framework_versions", {}),
            python_requirements=data.get("python_requirements", []),
            benchmark_scores=data.get("benchmark_scores", {}),
            evaluation_results=data.get("evaluation_results", {}),
            performance_delta=data.get("performance_delta", {}),
            huggingface_repo=data.get("huggingface_repo"),
            artifact_urls=data.get("artifact_urls", {}),
            file_checksums=data.get("file_checksums", {}),
            downloads=data.get("downloads", 0),
            stars=data.get("stars", 0),
            usage_count=data.get("usage_count", 0),
            community_ratings=data.get("community_ratings", []),
            signature=data.get("signature"),
            verification_hash=data.get("verification_hash"),
            trusted_by=set(data.get("trusted_by", [])),
            license=data.get("license", "apache-2.0"),
            governance_policy=data.get("governance_policy"),
        )


@dataclass
class PatchSearchQuery:
    """Query parameters for patch search."""
    
    query: Optional[str] = None
    patch_types: Optional[List[PatchType]] = None
    base_model: Optional[str] = None
    min_score: Optional[float] = None
    max_age_days: Optional[int] = None
    security_level: Optional[SecurityLevel] = None
    author: Optional[str] = None
    organization: Optional[str] = None
    has_evaluation: bool = False
    min_downloads: int = 0
    sort_by: str = "relevance"  # relevance, downloads, score, recent
    limit: int = 20


class PatchMarketplace:
    """Distributed patch marketplace with HuggingFace Hub integration."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Local cache and registry
        self.local_patches: Dict[str, PatchManifest] = {}
        self.patch_cache_dir = Path(self.config.get("cache_dir", "./cache/patches"))
        self.patch_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # HuggingFace integration
        self.hf_api = None
        self.hf_org = self.config.get("huggingface_org", "symbio-ai-patches")
        self.hf_token = self.config.get("huggingface_token")
        
        # Security and governance
        self.trusted_authors: Set[str] = set(self.config.get("trusted_authors", []))
        self.trusted_orgs: Set[str] = set(self.config.get("trusted_orgs", []))
        self.auto_verify_threshold = self.config.get("auto_verify_threshold", 0.85)
        
        # Performance tracking
        self.usage_stats: Dict[str, Dict[str, Any]] = {}
        
        if HF_AVAILABLE and self.hf_token:
            self.hf_api = HfApi(token=self.hf_token)
            self.logger.info("HuggingFace Hub integration enabled")
        else:
            self.logger.warning("HuggingFace Hub integration disabled (missing token or library)")
    
    async def initialize(self) -> None:
        """Initialize the marketplace."""
        # Load local patch cache
        await self._load_local_cache()
        
        # Sync with remote registries
        if self.hf_api:
            await self._sync_with_hub()
        
        self.logger.info("Patch marketplace initialized")
    
    async def publish_patch(
        self,
        manifest: PatchManifest,
        artifact_paths: Dict[str, str],
        auto_upload: bool = True
    ) -> str:
        """Publish a patch to the marketplace."""
        
        # Validate manifest
        await self._validate_manifest(manifest)
        
        # Generate checksums
        for name, path in artifact_paths.items():
            if Path(path).exists():
                manifest.file_checksums[name] = self._compute_checksum(path)
        
        # Upload to HuggingFace Hub if enabled
        if auto_upload and self.hf_api:
            repo_id = await self._upload_to_hub(manifest, artifact_paths)
            manifest.huggingface_repo = repo_id
        
        # Sign manifest
        manifest.signature = self._sign_manifest(manifest)
        manifest.verification_hash = self._compute_verification_hash(manifest)
        
        # Store locally
        self.local_patches[manifest.patch_id] = manifest
        await self._save_manifest(manifest)
        
        # Register with adapter registry
        await self._register_with_adapter_registry(manifest)
        
        # Emit telemetry
        OBSERVABILITY.emit_counter(
            "marketplace.patch_published",
            1,
            patch_type=manifest.patch_type.value,
            author=manifest.author,
            security_level=manifest.security_level.value
        )
        
        self.logger.info(f"Published patch: {manifest.patch_id}")
        return manifest.patch_id
    
    async def search_patches(self, query: PatchSearchQuery) -> List[PatchManifest]:
        """Search for patches in the marketplace."""
        
        # Search local cache
        local_results = await self._search_local(query)
        
        # Search HuggingFace Hub
        remote_results = []
        if self.hf_api:
            remote_results = await self._search_hub(query)
        
        # Combine and deduplicate results
        all_results = {p.patch_id: p for p in local_results + remote_results}
        results = list(all_results.values())
        
        # Apply ranking and filtering
        results = self._rank_results(results, query)
        
        # Emit telemetry
        OBSERVABILITY.emit_counter(
            "marketplace.patch_search",
            1,
            query_type=query.sort_by,
            results_count=len(results)
        )
        
        return results[:query.limit]
    
    async def install_patch(
        self,
        patch_id: str,
        tenant_id: Optional[str] = None,
        verify_signature: bool = True
    ) -> str:
        """Install a patch from the marketplace."""
        
        # Get patch manifest
        manifest = await self._get_patch_manifest(patch_id)
        if not manifest:
            raise ValueError(f"Patch {patch_id} not found")
        
        # Security and policy checks
        await self._check_install_policy(manifest, tenant_id)
        
        # Verify signature if required
        if verify_signature and not await self._verify_patch_signature(manifest):
            raise SecurityError(f"Patch signature verification failed: {patch_id}")
        
        # Download artifacts
        artifact_dir = await self._download_patch_artifacts(manifest)
        
        # Install patch
        installation_path = await self._install_patch_artifacts(manifest, artifact_dir)
        
        # Update usage statistics
        manifest.usage_count += 1
        self._update_usage_stats(manifest, tenant_id)
        
        # Register with adapter registry
        await self._register_installed_patch(manifest, installation_path)
        
        # Emit telemetry
        OBSERVABILITY.emit_counter(
            "marketplace.patch_installed",
            1,
            patch_id=patch_id,
            patch_type=manifest.patch_type.value,
            tenant_id=tenant_id or "system"
        )
        
        self.logger.info(f"Installed patch: {patch_id} -> {installation_path}")
        return installation_path
    
    async def evaluate_patch(
        self,
        patch_id: str,
        evaluation_suite: Optional[str] = None
    ) -> Dict[str, Any]:
        """Evaluate a patch using automated benchmarks."""
        
        manifest = await self._get_patch_manifest(patch_id)
        if not manifest:
            raise ValueError(f"Patch {patch_id} not found")
        
        # Run evaluation
        results = await self._run_patch_evaluation(manifest, evaluation_suite)
        
        # Update manifest with results
        manifest.evaluation_results.update(results)
        manifest.updated_at = datetime.utcnow()
        
        # Auto-verify if score is high enough
        if self._should_auto_verify(results):
            manifest.status = PatchStatus.VERIFIED
            manifest.trusted_by.add("auto_evaluator")
        
        # Save updated manifest
        await self._save_manifest(manifest)
        
        # Emit telemetry
        OBSERVABILITY.emit_counter(
            "marketplace.patch_evaluated",
            1,
            patch_id=patch_id,
            auto_verified=manifest.status == PatchStatus.VERIFIED
        )
        
        return results
    
    async def get_recommendations(
        self,
        context: Dict[str, Any],
        tenant_id: Optional[str] = None
    ) -> List[PatchManifest]:
        """Get patch recommendations based on context."""
        
        # Extract context information
        model_type = context.get("model_type")
        task_type = context.get("task_type")
        domain = context.get("domain")
        performance_issues = context.get("performance_issues", [])
        
        # Build recommendation query
        query = PatchSearchQuery(
            query=f"{task_type} {domain}",
            base_model=model_type,
            min_score=0.7,
            max_age_days=90,
            has_evaluation=True,
            sort_by="score",
            limit=10
        )
        
        # Get candidates
        candidates = await self.search_patches(query)
        
        # Apply recommendation scoring
        scored_candidates = []
        for patch in candidates:
            score = self._calculate_recommendation_score(patch, context, tenant_id)
            scored_candidates.append((score, patch))
        
        # Sort by score and return top recommendations
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        recommendations = [patch for score, patch in scored_candidates[:5]]
        
        # Emit telemetry
        OBSERVABILITY.emit_counter(
            "marketplace.recommendations_generated",
            1,
            tenant_id=tenant_id or "system",
            count=len(recommendations)
        )
        
        return recommendations
    
    async def contribute_evaluation(
        self,
        patch_id: str,
        evaluation_results: Dict[str, Any],
        evaluator_id: str
    ) -> None:
        """Contribute evaluation results for a patch."""
        
        manifest = await self._get_patch_manifest(patch_id)
        if not manifest:
            raise ValueError(f"Patch {patch_id} not found")
        
        # Add evaluation results
        eval_key = f"evaluation_{evaluator_id}_{int(time.time())}"
        manifest.evaluation_results[eval_key] = {
            "evaluator": evaluator_id,
            "timestamp": datetime.utcnow().isoformat(),
            "results": evaluation_results
        }
        
        # Update aggregate scores
        self._update_aggregate_scores(manifest)
        
        # Save manifest
        await self._save_manifest(manifest)
        
        # Emit telemetry
        OBSERVABILITY.emit_counter(
            "marketplace.evaluation_contributed",
            1,
            patch_id=patch_id,
            evaluator=evaluator_id
        )
        
        self.logger.info(f"Evaluation contributed for patch {patch_id} by {evaluator_id}")
    
    def get_marketplace_stats(self) -> Dict[str, Any]:
        """Get marketplace statistics."""
        
        total_patches = len(self.local_patches)
        by_type = {}
        by_status = {}
        by_security = {}
        
        for patch in self.local_patches.values():
            # Count by type
            patch_type = patch.patch_type.value
            by_type[patch_type] = by_type.get(patch_type, 0) + 1
            
            # Count by status
            status = patch.status.value
            by_status[status] = by_status.get(status, 0) + 1
            
            # Count by security level
            security = patch.security_level.value
            by_security[security] = by_security.get(security, 0) + 1
        
        return {
            "total_patches": total_patches,
            "by_type": by_type,
            "by_status": by_status,
            "by_security_level": by_security,
            "huggingface_integration": self.hf_api is not None,
            "trusted_authors": len(self.trusted_authors),
            "trusted_organizations": len(self.trusted_orgs),
            "cache_directory": str(self.patch_cache_dir),
            "usage_stats": dict(self.usage_stats)
        }
    
    # Private methods
    
    async def _load_local_cache(self) -> None:
        """Load patches from local cache."""
        manifest_files = self.patch_cache_dir.glob("*.json")
        
        for manifest_file in manifest_files:
            try:
                with open(manifest_file, 'r') as f:
                    data = json.load(f)
                    manifest = PatchManifest.from_dict(data)
                    self.local_patches[manifest.patch_id] = manifest
            except Exception as e:
                self.logger.warning(f"Failed to load manifest {manifest_file}: {e}")
    
    async def _sync_with_hub(self) -> None:
        """Sync with HuggingFace Hub."""
        if not self.hf_api:
            return
        
        try:
            # List repositories in organization
            repos = self.hf_api.list_repos(author=self.hf_org, repo_type="model")
            
            for repo in repos:
                if repo.id.endswith("-patch"):
                    # Download and cache manifest
                    await self._cache_hub_patch(repo.id)
                    
        except Exception as e:
            self.logger.warning(f"Failed to sync with HuggingFace Hub: {e}")
    
    async def _validate_manifest(self, manifest: PatchManifest) -> None:
        """Validate patch manifest."""
        
        # Check required fields
        if not manifest.name or not manifest.version:
            raise ValueError("Patch name and version are required")
        
        # Check security level permissions
        tenant = TENANT_REGISTRY.get_tenant("system")  # Use system context for validation
        if tenant and manifest.security_level == SecurityLevel.ENTERPRISE_APPROVED:
            if "enterprise_publishing" not in tenant.enabled_policies:
                raise PermissionError("Enterprise publishing not enabled")
        
        # Validate dependencies
        for requirement in manifest.python_requirements:
            if not self._is_safe_requirement(requirement):
                raise SecurityError(f"Unsafe Python requirement: {requirement}")
    
    async def _upload_to_hub(
        self,
        manifest: PatchManifest,
        artifact_paths: Dict[str, str]
    ) -> str:
        """Upload patch to HuggingFace Hub."""
        
        if not self.hf_api:
            raise RuntimeError("HuggingFace API not available")
        
        repo_id = f"{self.hf_org}/{manifest.patch_id}-patch"
        
        try:
            # Create repository
            self.hf_api.create_repo(repo_id=repo_id, exist_ok=True)
            
            # Upload manifest as README
            readme_content = self._generate_model_card(manifest)
            self.hf_api.upload_file(
                path_or_fileobj=readme_content.encode(),
                path_in_repo="README.md",
                repo_id=repo_id
            )
            
            # Upload manifest JSON
            manifest_json = json.dumps(manifest.to_dict(), indent=2)
            self.hf_api.upload_file(
                path_or_fileobj=manifest_json.encode(),
                path_in_repo="patch_manifest.json",
                repo_id=repo_id
            )
            
            # Upload artifacts
            for name, path in artifact_paths.items():
                if Path(path).exists():
                    self.hf_api.upload_file(
                        path_or_fileobj=path,
                        path_in_repo=name,
                        repo_id=repo_id
                    )
            
            return repo_id
            
        except Exception as e:
            self.logger.error(f"Failed to upload to HuggingFace Hub: {e}")
            raise
    
    def _compute_checksum(self, file_path: str) -> str:
        """Compute SHA256 checksum of a file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    
    def _sign_manifest(self, manifest: PatchManifest) -> str:
        """Sign patch manifest (simplified implementation)."""
        # In production, use proper cryptographic signing
        manifest_str = json.dumps(manifest.to_dict(), sort_keys=True)
        return hashlib.sha256(manifest_str.encode()).hexdigest()
    
    def _compute_verification_hash(self, manifest: PatchManifest) -> str:
        """Compute verification hash for manifest."""
        verification_data = {
            "patch_id": manifest.patch_id,
            "version": manifest.version,
            "author": manifest.author,
            "checksums": manifest.file_checksums
        }
        verification_str = json.dumps(verification_data, sort_keys=True)
        return hashlib.sha256(verification_str.encode()).hexdigest()
    
    async def _register_with_adapter_registry(self, manifest: PatchManifest) -> None:
        """Register patch with the adapter registry."""
        
        adapter_metadata = AdapterMetadata(
            adapter_id=manifest.patch_id,
            name=manifest.name,
            version=manifest.version,
            capabilities={manifest.patch_type.value},
            owner=manifest.author,
            lineage=f"marketplace:{manifest.huggingface_repo or 'local'}",
            config={
                "patch_type": manifest.patch_type.value,
                "security_level": manifest.security_level.value,
                "manifest_path": str(self.patch_cache_dir / f"{manifest.patch_id}.json")
            }
        )
        
        ADAPTER_REGISTRY.register_adapter(adapter_metadata)
    
    def _calculate_recommendation_score(
        self,
        patch: PatchManifest,
        context: Dict[str, Any],
        tenant_id: Optional[str]
    ) -> float:
        """Calculate recommendation score for a patch."""
        
        score = 0.0
        
        # Base compatibility score
        if context.get("model_type") in patch.base_models:
            score += 0.3
        
        # Performance improvement score
        task_type = context.get("task_type", "")
        if task_type in patch.performance_delta:
            delta = patch.performance_delta[task_type]
            score += min(0.4, delta * 2)  # Cap at 0.4
        
        # Community validation score
        if patch.community_ratings:
            avg_rating = sum(patch.community_ratings) / len(patch.community_ratings)
            score += (avg_rating / 5.0) * 0.2
        
        # Recency score
        age_days = (datetime.utcnow() - patch.updated_at).days
        recency_score = max(0, 1 - (age_days / 90))  # Decay over 90 days
        score += recency_score * 0.1
        
        # Security level adjustment
        if patch.security_level in [SecurityLevel.COMMUNITY_VERIFIED, SecurityLevel.ENTERPRISE_APPROVED]:
            score += 0.1
        
        return min(1.0, score)
    
    async def _get_patch_manifest(self, patch_id: str) -> Optional[PatchManifest]:
        """Get patch manifest by ID."""
        
        # Check local cache first
        if patch_id in self.local_patches:
            return self.local_patches[patch_id]
        
        # Try to fetch from HuggingFace Hub
        if self.hf_api:
            try:
                repo_id = f"{self.hf_org}/{patch_id}-patch"
                manifest_path = hf_hub_download(
                    repo_id=repo_id,
                    filename="patch_manifest.json",
                    cache_dir=str(self.patch_cache_dir)
                )
                
                with open(manifest_path, 'r') as f:
                    data = json.load(f)
                    manifest = PatchManifest.from_dict(data)
                    self.local_patches[patch_id] = manifest
                    return manifest
                    
            except Exception as e:
                self.logger.debug(f"Failed to fetch patch {patch_id} from Hub: {e}")
        
        return None
    
    async def _save_manifest(self, manifest: PatchManifest) -> None:
        """Save manifest to local cache."""
        manifest_path = self.patch_cache_dir / f"{manifest.patch_id}.json"
        
        with open(manifest_path, 'w') as f:
            json.dump(manifest.to_dict(), f, indent=2)
    
    def _generate_model_card(self, manifest: PatchManifest) -> str:
        """Generate HuggingFace model card for patch."""
        
        return f"""---
license: {manifest.license}
library_name: symbio-ai
tags:
- patch
- {manifest.patch_type.value}
- symbio-ai-marketplace
---

# {manifest.name}

{manifest.description}

## Patch Information

- **Type**: {manifest.patch_type.value}
- **Version**: {manifest.version}
- **Author**: {manifest.author}
- **Security Level**: {manifest.security_level.value}
- **Status**: {manifest.status.value}

## Base Models

{chr(10).join(f"- {model}" for model in manifest.base_models)}

## Performance Metrics

{chr(10).join(f"- {metric}: {score:.3f}" for metric, score in manifest.benchmark_scores.items())}

## Installation

```python
from marketplace.patch_marketplace import PatchMarketplace

marketplace = PatchMarketplace()
await marketplace.install_patch("{manifest.patch_id}")
```

## Usage Statistics

- Downloads: {manifest.downloads}
- Usage Count: {manifest.usage_count}
- Community Rating: {sum(manifest.community_ratings) / len(manifest.community_ratings) if manifest.community_ratings else 'N/A'}

Generated by Symbio AI Patch Marketplace
"""

    def _update_usage_stats(self, manifest: PatchManifest, tenant_id: Optional[str]) -> None:
        """Update usage statistics."""
        
        if manifest.patch_id not in self.usage_stats:
            self.usage_stats[manifest.patch_id] = {
                "total_installs": 0,
                "tenant_installs": {},
                "last_used": None
            }
        
        stats = self.usage_stats[manifest.patch_id]
        stats["total_installs"] += 1
        stats["last_used"] = datetime.utcnow().isoformat()
        
        if tenant_id:
            tenant_key = tenant_id
            stats["tenant_installs"][tenant_key] = stats["tenant_installs"].get(tenant_key, 0) + 1


class SecurityError(Exception):
    """Security-related error in patch marketplace."""
    pass


# Global marketplace instance
PATCH_MARKETPLACE = PatchMarketplace()