"""
API Gateway and service mesh configuration for Symbio AI.

Provides REST API endpoints, GraphQL interface, WebSocket connections,
load balancing, service discovery, and API versioning capabilities.
"""

import asyncio
import logging
import json
from typing import Dict, Any, List, Optional, Union, Callable, Tuple
from dataclasses import dataclass, field, asdict
from abc import ABC, abstractmethod
from enum import Enum
from datetime import datetime, timedelta
import time
from pathlib import Path
import hashlib
import jwt
from contextlib import asynccontextmanager
import aiohttp
from aiohttp import web, WSMsgType
from aiohttp.web import middleware
import aioredis
from asyncio import Queue
import ssl
import weakref

from control_plane.policy_engine import DEFAULT_POLICY_ENGINE, PolicyContext
from control_plane.tenancy import TENANT_REGISTRY, TenantSettings
from monitoring.observability import OBSERVABILITY


class APIVersion(Enum):
    """API version enumeration."""
    V1 = "v1"
    V2 = "v2"
    BETA = "beta"
    ALPHA = "alpha"


class ServiceType(Enum):
    """Service type classification."""
    CORE = "core"
    AI_MODEL = "ai_model"
    DATA_PROCESSING = "data_processing"
    TRAINING = "training"
    INFERENCE = "inference"
    MONITORING = "monitoring"
    AUTHENTICATION = "authentication"


class LoadBalancingStrategy(Enum):
    """Load balancing strategies."""
    ROUND_ROBIN = "round_robin"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_CONNECTIONS = "least_connections"
    RANDOM = "random"
    CONSISTENT_HASH = "consistent_hash"
    HEALTH_BASED = "health_based"


@dataclass
class ServiceEndpoint:
    """Service endpoint configuration."""
    host: str
    port: int
    path: str = "/"
    protocol: str = "http"
    weight: int = 100
    health_check_path: str = "/health"
    timeout: float = 30.0
    max_retries: int = 3
    circuit_breaker_enabled: bool = True
    
    @property
    def url(self) -> str:
        return f"{self.protocol}://{self.host}:{self.port}{self.path}"
    
    @property
    def health_url(self) -> str:
        return f"{self.protocol}://{self.host}:{self.port}{self.health_check_path}"


@dataclass
class APIRoute:
    """API route definition."""
    path: str
    method: str
    handler: str
    version: APIVersion = APIVersion.V1
    auth_required: bool = True
    rate_limit: int = 100
    timeout: float = 30.0
    cache_ttl: int = 0
    description: str = ""
    tags: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    response_schema: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ServiceRegistration:
    """Service registration information."""
    name: str
    service_type: ServiceType
    endpoints: List[ServiceEndpoint]
    version: str = "1.0.0"
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    health_check_interval: int = 30
    registration_time: str = field(default_factory=lambda: datetime.now().isoformat())
    last_heartbeat: str = field(default_factory=lambda: datetime.now().isoformat())


class CircuitBreaker:
    """Circuit breaker for fault tolerance."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        timeout: float = 60.0,
        expected_exception: type = Exception
    ):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.expected_exception = expected_exception
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.logger = logging.getLogger(__name__)
    
    async def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        if self.state == "OPEN":
            if time.time() - self.last_failure_time < self.timeout:
                raise Exception("Circuit breaker is OPEN")
            else:
                self.state = "HALF_OPEN"
        
        try:
            result = await func(*args, **kwargs)
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failure_count = 0
            return result
            
        except self.expected_exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
                self.logger.warning(f"Circuit breaker opened due to {self.failure_count} failures")
            
            raise e


class RateLimiter:
    """Token bucket rate limiter."""
    
    def __init__(self, rate: int, burst: int = None):
        self.rate = rate  # tokens per second
        self.burst = burst or rate
        self.tokens = self.burst
        self.last_update = time.time()
        self._lock = asyncio.Lock()
    
    async def acquire(self, tokens: int = 1) -> bool:
        """Acquire tokens from the bucket."""
        async with self._lock:
            now = time.time()
            # Add tokens based on elapsed time
            elapsed = now - self.last_update
            self.tokens = min(self.burst, self.tokens + elapsed * self.rate)
            self.last_update = now
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False


class LoadBalancer:
    """Service load balancer with multiple strategies."""
    
    def __init__(self, strategy: LoadBalancingStrategy = LoadBalancingStrategy.ROUND_ROBIN):
        self.strategy = strategy
        self.current_index = 0
        self.connections = {}
        self.health_status = {}
        self.logger = logging.getLogger(__name__)
    
    async def select_endpoint(self, endpoints: List[ServiceEndpoint]) -> Optional[ServiceEndpoint]:
        """Select endpoint based on load balancing strategy."""
        healthy_endpoints = [
            ep for ep in endpoints 
            if self.health_status.get(ep.url, True)
        ]
        
        if not healthy_endpoints:
            self.logger.warning("No healthy endpoints available")
            return None
        
        if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return self._round_robin_select(healthy_endpoints)
        elif self.strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
            return self._weighted_round_robin_select(healthy_endpoints)
        elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            return self._least_connections_select(healthy_endpoints)
        elif self.strategy == LoadBalancingStrategy.RANDOM:
            return self._random_select(healthy_endpoints)
        elif self.strategy == LoadBalancingStrategy.HEALTH_BASED:
            return self._health_based_select(healthy_endpoints)
        else:
            return self._round_robin_select(healthy_endpoints)
    
    def _round_robin_select(self, endpoints: List[ServiceEndpoint]) -> ServiceEndpoint:
        """Round-robin endpoint selection."""
        endpoint = endpoints[self.current_index % len(endpoints)]
        self.current_index += 1
        return endpoint
    
    def _weighted_round_robin_select(self, endpoints: List[ServiceEndpoint]) -> ServiceEndpoint:
        """Weighted round-robin selection."""
        total_weight = sum(ep.weight for ep in endpoints)
        weighted_endpoints = []
        
        for ep in endpoints:
            count = int((ep.weight / total_weight) * 100)
            weighted_endpoints.extend([ep] * count)
        
        if weighted_endpoints:
            endpoint = weighted_endpoints[self.current_index % len(weighted_endpoints)]
            self.current_index += 1
            return endpoint
        
        return endpoints[0]
    
    def _least_connections_select(self, endpoints: List[ServiceEndpoint]) -> ServiceEndpoint:
        """Least connections selection."""
        min_connections = float('inf')
        selected_endpoint = endpoints[0]
        
        for ep in endpoints:
            connections = self.connections.get(ep.url, 0)
            if connections < min_connections:
                min_connections = connections
                selected_endpoint = ep
        
        return selected_endpoint
    
    def _random_select(self, endpoints: List[ServiceEndpoint]) -> ServiceEndpoint:
        """Random endpoint selection."""
        import random
        return random.choice(endpoints)
    
    def _health_based_select(self, endpoints: List[ServiceEndpoint]) -> ServiceEndpoint:
        """Health score-based selection."""
        # In a real implementation, this would use actual health metrics
        return min(endpoints, key=lambda ep: self.connections.get(ep.url, 0))
    
    async def update_health_status(self, endpoint_url: str, is_healthy: bool):
        """Update endpoint health status."""
        self.health_status[endpoint_url] = is_healthy
    
    def increment_connections(self, endpoint_url: str):
        """Increment connection count for endpoint."""
        self.connections[endpoint_url] = self.connections.get(endpoint_url, 0) + 1
    
    def decrement_connections(self, endpoint_url: str):
        """Decrement connection count for endpoint."""
        if endpoint_url in self.connections:
            self.connections[endpoint_url] = max(0, self.connections[endpoint_url] - 1)


class ServiceDiscovery:
    """Service discovery and registration."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self.redis = None
        self.services = {}
        self.health_checkers = {}
        self.logger = logging.getLogger(__name__)
    
    async def initialize(self):
        """Initialize Redis connection."""
        try:
            self.redis = await aioredis.from_url(self.redis_url)
            self.logger.info("Service discovery initialized with Redis")
        except Exception as e:
            self.logger.error(f"Failed to connect to Redis: {e}")
            # Fallback to in-memory storage
            self.redis = None
    
    async def register_service(self, registration: ServiceRegistration) -> bool:
        """Register a service."""
        try:
            service_key = f"service:{registration.name}"
            service_data = asdict(registration)
            
            if self.redis:
                await self.redis.setex(
                    service_key, 
                    registration.health_check_interval * 3,  # TTL
                    json.dumps(service_data)
                )
            else:
                self.services[registration.name] = registration
            
            # Start health checking
            await self._start_health_check(registration)
            
            self.logger.info(f"Service registered: {registration.name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register service {registration.name}: {e}")
            return False
    
    async def discover_service(self, service_name: str) -> Optional[ServiceRegistration]:
        """Discover a service by name."""
        try:
            service_key = f"service:{service_name}"
            
            if self.redis:
                service_data = await self.redis.get(service_key)
                if service_data:
                    data = json.loads(service_data)
                    return ServiceRegistration(**data)
            else:
                return self.services.get(service_name)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to discover service {service_name}: {e}")
            return None
    
    async def list_services(self, service_type: ServiceType = None) -> List[ServiceRegistration]:
        """List all registered services."""
        try:
            services = []
            
            if self.redis:
                keys = await self.redis.keys("service:*")
                for key in keys:
                    service_data = await self.redis.get(key)
                    if service_data:
                        data = json.loads(service_data)
                        registration = ServiceRegistration(**data)
                        if not service_type or registration.service_type == service_type:
                            services.append(registration)
            else:
                for registration in self.services.values():
                    if not service_type or registration.service_type == service_type:
                        services.append(registration)
            
            return services
            
        except Exception as e:
            self.logger.error(f"Failed to list services: {e}")
            return []
    
    async def _start_health_check(self, registration: ServiceRegistration):
        """Start health check for registered service."""
        async def health_check():
            while True:
                try:
                    healthy = await self._check_service_health(registration)
                    if healthy:
                        # Update heartbeat
                        await self._update_heartbeat(registration.name)
                    else:
                        # Remove unhealthy service
                        await self._deregister_service(registration.name)
                        break
                    
                    await asyncio.sleep(registration.health_check_interval)
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    self.logger.error(f"Health check error for {registration.name}: {e}")
                    await asyncio.sleep(registration.health_check_interval)
        
        # Store the task for potential cancellation
        task = asyncio.create_task(health_check())
        self.health_checkers[registration.name] = task
    
    async def _check_service_health(self, registration: ServiceRegistration) -> bool:
        """Check health of a service."""
        try:
            async with aiohttp.ClientSession() as session:
                for endpoint in registration.endpoints:
                    try:
                        async with session.get(
                            endpoint.health_url,
                            timeout=aiohttp.ClientTimeout(total=5)
                        ) as response:
                            if response.status == 200:
                                return True
                    except:
                        continue
                return False
        except:
            return False
    
    async def _update_heartbeat(self, service_name: str):
        """Update service heartbeat."""
        try:
            if self.redis:
                service_key = f"service:{service_name}"
                service_data = await self.redis.get(service_key)
                if service_data:
                    data = json.loads(service_data)
                    data['last_heartbeat'] = datetime.now().isoformat()
                    await self.redis.setex(
                        service_key,
                        data.get('health_check_interval', 30) * 3,
                        json.dumps(data)
                    )
            else:
                if service_name in self.services:
                    self.services[service_name].last_heartbeat = datetime.now().isoformat()
                    
        except Exception as e:
            self.logger.error(f"Failed to update heartbeat for {service_name}: {e}")
    
    async def _deregister_service(self, service_name: str):
        """Deregister a service."""
        try:
            if self.redis:
                await self.redis.delete(f"service:{service_name}")
            else:
                self.services.pop(service_name, None)
            
            # Cancel health check task
            if service_name in self.health_checkers:
                self.health_checkers[service_name].cancel()
                del self.health_checkers[service_name]
            
            self.logger.info(f"Service deregistered: {service_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to deregister service {service_name}: {e}")
    
    async def cleanup(self):
        """Cleanup resources."""
        # Cancel all health check tasks
        for task in self.health_checkers.values():
            task.cancel()
        
        if self.redis:
            await self.redis.close()


class APIGateway:
    """API Gateway with routing, authentication, and load balancing."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.routes = {}
        self.service_discovery = ServiceDiscovery(config.get('redis_url', 'redis://localhost:6379'))
        self.load_balancer = LoadBalancer(
            LoadBalancingStrategy(config.get('load_balancing_strategy', 'round_robin'))
        )
        self.rate_limiters = {}
        self.circuit_breakers = {}
        self.request_cache = {}
        self.policy_engine = DEFAULT_POLICY_ENGINE
        self.tenant_registry = TENANT_REGISTRY
        self.observability = OBSERVABILITY
        self.websocket_connections = weakref.WeakSet()
        self.app = None
        self.logger = logging.getLogger(__name__)
    
    async def initialize(self):
        """Initialize API Gateway."""
        await self.service_discovery.initialize()
        self.app = web.Application(middlewares=[
            self._cors_middleware,
            self._auth_middleware,
            self._policy_middleware,
            self._rate_limit_middleware,
            self._logging_middleware,
            self._error_middleware
        ])
        
        # Setup routes
        self._setup_routes()
        
        self.logger.info("API Gateway initialized")

    def _ensure_tenant(self, identity_payload: Dict[str, Any]) -> TenantSettings:
        tenant_id = identity_payload.get("tenant_id") or identity_payload.get("org_id") or "public"
        tenant = self.tenant_registry.get_tenant(tenant_id)
        if tenant:
            return tenant

        tenant = TenantSettings(
            tenant_id=tenant_id,
            name=identity_payload.get("tenant_name", tenant_id.title()),
            contact_email=identity_payload.get("email", "ops@symbio.ai"),
            metadata={"source": "jwt"}
        )
        self.tenant_registry.register_tenant(tenant)
        self.logger.info("Registered new tenant from token: %s", tenant_id)
        return tenant
    
    def _setup_routes(self):
        """Setup API routes."""
        # Health check
        self.app.router.add_get('/health', self._health_check)
        self.app.router.add_get('/ready', self._readiness_check)
        
        # API versioned routes
        for version in APIVersion:
            version_prefix = f'/{version.value}'
            
            # Core API routes
            self.app.router.add_route('*', f'{version_prefix}/models{{path:.*}}', self._route_to_service)
            self.app.router.add_route('*', f'{version_prefix}/training{{path:.*}}', self._route_to_service)
            self.app.router.add_route('*', f'{version_prefix}/inference{{path:.*}}', self._route_to_service)
            self.app.router.add_route('*', f'{version_prefix}/data{{path:.*}}', self._route_to_service)
            self.app.router.add_route('*', f'{version_prefix}/agents{{path:.*}}', self._route_to_service)
            
            # WebSocket endpoints
            self.app.router.add_get(f'{version_prefix}/ws/training', self._websocket_handler)
            self.app.router.add_get(f'{version_prefix}/ws/inference', self._websocket_handler)
            self.app.router.add_get(f'{version_prefix}/ws/monitoring', self._websocket_handler)
        
        # Admin routes
        self.app.router.add_get('/admin/services', self._list_services)
        self.app.router.add_get('/admin/metrics', self._get_metrics)
        self.app.router.add_post('/admin/services/register', self._register_service)
    
    @middleware
    async def _cors_middleware(self, request, handler):
        """CORS middleware."""
        response = await handler(request)
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        return response
    
    @middleware
    async def _auth_middleware(self, request, handler):
        """Authentication middleware."""
        # Skip auth for health checks and public endpoints
        if request.path in ['/health', '/ready'] or request.path.startswith('/admin'):
            return await handler(request)
        
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return web.json_response({'error': 'Missing or invalid authorization header'}, status=401)
        
        token = auth_header[7:]  # Remove 'Bearer ' prefix
        try:
            # In production, validate JWT token
            payload = jwt.decode(token, self.config.get('jwt_secret', 'secret'), algorithms=['HS256'])
            request['user'] = payload
            request['tenant'] = self._ensure_tenant(payload)
        except jwt.InvalidTokenError:
            return web.json_response({'error': 'Invalid token'}, status=401)
        
        return await handler(request)

    @middleware
    async def _policy_middleware(self, request, handler):
        """Policy evaluation and quota enforcement."""
        tenant = request.get('tenant')
        if not tenant:
            return await handler(request)

        metadata = {
            'method': request.method,
            'route': request.path,
        }

        quota_ok = self.tenant_registry.check_quota(tenant.tenant_id)
        if not quota_ok:
            metadata['quota_exceeded'] = self.config.get('quota_retry_after_seconds', 60)

        context = PolicyContext(
            tenant=tenant,
            request_payload={},
            adapter_id=tenant.default_adapter_id,
            route=request.path,
            metadata=metadata
        )
        decision = self.policy_engine.evaluate(context)
        if not decision.allowed:
            self.observability.emit_counter(
                'gateway.policy_denials',
                1,
                tenant_id=tenant.tenant_id,
                route=request.path,
                reason=decision.reason,
            )
            status_code = 429 if metadata.get('quota_exceeded') else 403
            payload = {'error': 'Policy denied', 'reason': decision.reason}
            if decision.remediation:
                payload['remediation'] = decision.remediation
            return web.json_response(payload, status=status_code)

        return await handler(request)
    
    @middleware
    async def _rate_limit_middleware(self, request, handler):
        """Rate limiting middleware."""
        client_ip = request.remote
        endpoint = f"{request.method}:{request.path}"
        
        tenant = request.get('tenant')
        rate_limit_key = f"{tenant.tenant_id if tenant else client_ip}:{endpoint}"
        if rate_limit_key not in self.rate_limiters:
            self.rate_limiters[rate_limit_key] = RateLimiter(
                rate=self.config.get('default_rate_limit', 100)
            )
        
        rate_limiter = self.rate_limiters[rate_limit_key]
        if not await rate_limiter.acquire():
            self.observability.emit_counter(
                'gateway.rate_limit_exceeded',
                1,
                tenant_id=tenant.tenant_id if tenant else 'anon',
                endpoint=endpoint
            )
            return web.json_response({'error': 'Rate limit exceeded'}, status=429)
        
        return await handler(request)
    
    @middleware
    async def _logging_middleware(self, request, handler):
        """Request logging middleware."""
        start_time = time.time()
        
        try:
            response = await handler(request)
            duration = time.time() - start_time
            duration_ms = duration * 1000
            
            self.logger.info(
                f"{request.method} {request.path} {response.status} "
                f"{duration:.3f}s {request.remote}"
            )

            self.observability.emit_counter(
                'gateway.requests_per_minute',
                1,
                method=request.method,
                route=request.path,
                status=response.status
            )
            self.observability.emit_gauge(
                'gateway.request_latency_ms',
                duration_ms,
                method=request.method,
                route=request.path
            )
            
            return response
        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(
                f"{request.method} {request.path} ERROR "
                f"{duration:.3f}s {request.remote} - {str(e)}"
            )
            self.observability.emit_counter(
                'gateway.errors',
                1,
                method=request.method,
                route=request.path,
                error=str(e)
            )
            raise
    
    @middleware
    async def _error_middleware(self, request, handler):
        """Error handling middleware."""
        try:
            return await handler(request)
        except web.HTTPException:
            raise
        except Exception as e:
            self.logger.error(f"Unhandled error in {request.path}: {e}")
            return web.json_response(
                {'error': 'Internal server error'},
                status=500
            )
    
    async def _health_check(self, request):
        """Health check endpoint."""
        return web.json_response({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'version': self.config.get('version', '1.0.0')
        })
    
    async def _readiness_check(self, request):
        """Readiness check endpoint."""
        # Check if critical services are available
        services = await self.service_discovery.list_services()
        ready = len(services) > 0
        
        return web.json_response({
            'status': 'ready' if ready else 'not_ready',
            'services_count': len(services),
            'timestamp': datetime.now().isoformat()
        }, status=200 if ready else 503)
    
    async def _route_to_service(self, request):
        """Route request to appropriate microservice."""
        path_parts = request.path.strip('/').split('/')
        tenant = request.get('tenant')
        
        if len(path_parts) < 2:
            return web.json_response({'error': 'Invalid path'}, status=400)
        
        version = path_parts[0]
        service_name = path_parts[1]
        
        # Map service names to service types
        service_type_map = {
            'models': ServiceType.AI_MODEL,
            'training': ServiceType.TRAINING,
            'inference': ServiceType.INFERENCE,
            'data': ServiceType.DATA_PROCESSING,
            'agents': ServiceType.CORE
        }
        
        service_type = service_type_map.get(service_name)
        if not service_type:
            return web.json_response({'error': f'Unknown service: {service_name}'}, status=404)
        
        # Find service
        services = await self.service_discovery.list_services(service_type)
        if not services:
            self.observability.emit_counter(
                'gateway.service_unavailable',
                1,
                tenant_id=tenant.tenant_id if tenant else 'anon',
                service=service_name
            )
            return web.json_response({'error': f'Service {service_name} not available'}, status=503)
        
        # Select endpoint
        service = services[0]  # For simplicity, use first service
        endpoint = await self.load_balancer.select_endpoint(service.endpoints)
        
        if not endpoint:
            self.observability.emit_counter(
                'gateway.service_unavailable',
                1,
                tenant_id=tenant.tenant_id if tenant else 'anon',
                service=service_name,
                reason='no_healthy_endpoints'
            )
            return web.json_response({'error': 'No healthy endpoints available'}, status=503)
        
        self.observability.emit_counter(
            'gateway.route_requests',
            1,
            tenant_id=tenant.tenant_id if tenant else 'anon',
            service=service_name,
            endpoint=endpoint.url
        )

        # Get or create circuit breaker for this endpoint
        cb_key = endpoint.url
        if cb_key not in self.circuit_breakers:
            self.circuit_breakers[cb_key] = CircuitBreaker()
        
        circuit_breaker = self.circuit_breakers[cb_key]
        
        try:
            # Forward request
            response = await circuit_breaker.call(
                self._forward_request,
                request,
                endpoint
            )
            return response
            
        except Exception as e:
            self.logger.error(f"Failed to forward request to {endpoint.url}: {e}")
            self.observability.emit_counter(
                'gateway.route_failures',
                1,
                tenant_id=tenant.tenant_id if tenant else 'anon',
                service=service_name,
                endpoint=endpoint.url
            )
            return web.json_response({'error': 'Service temporarily unavailable'}, status=503)
    
    async def _forward_request(self, request, endpoint: ServiceEndpoint):
        """Forward HTTP request to service endpoint."""
        # Prepare request data
        url = f"{endpoint.url}{request.path_qs}"
        headers = dict(request.headers)
        tenant = request.get('tenant')
        
        # Remove hop-by-hop headers
        hop_by_hop = ['connection', 'keep-alive', 'proxy-authenticate', 
                      'proxy-authorization', 'te', 'trailers', 'transfer-encoding', 'upgrade']
        for header in hop_by_hop:
            headers.pop(header, None)
        
        # Read request body
        body = None
        if request.can_read_body:
            body = await request.read()
        request_size = len(body) if body else 0
        response_size = 0
        
        self.load_balancer.increment_connections(endpoint.url)
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.request(
                    request.method,
                    url,
                    headers=headers,
                    data=body,
                    timeout=aiohttp.ClientTimeout(total=endpoint.timeout)
                ) as response:
                    # Read response
                    response_body = await response.read()
                    response_size = len(response_body)
                    response_headers = dict(response.headers)
                    
                    # Remove hop-by-hop headers
                    for header in hop_by_hop:
                        response_headers.pop(header, None)
                    
                    self.observability.emit_counter(
                        'gateway.backend_success',
                        1,
                        tenant_id=tenant.tenant_id if tenant else 'anon',
                        endpoint=endpoint.url,
                        status=response.status
                    )
                    return web.Response(
                        body=response_body,
                        status=response.status,
                        headers=response_headers
                    )
        finally:
            self.load_balancer.decrement_connections(endpoint.url)
            if tenant:
                self.tenant_registry.update_usage(
                    tenant.tenant_id,
                    requests=1,
                    tokens=request_size + response_size
                )
    
    async def _websocket_handler(self, request):
        """WebSocket connection handler."""
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        
        self.websocket_connections.add(ws)
        
        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                        await self._handle_websocket_message(ws, data)
                    except json.JSONDecodeError:
                        await ws.send_str(json.dumps({'error': 'Invalid JSON'}))
                elif msg.type == WSMsgType.ERROR:
                    self.logger.error(f'WebSocket error: {ws.exception()}')
        except Exception as e:
            self.logger.error(f'WebSocket handler error: {e}')
        finally:
            self.websocket_connections.discard(ws)
        
        return ws
    
    async def _handle_websocket_message(self, ws, data: Dict[str, Any]):
        """Handle WebSocket message."""
        message_type = data.get('type')
        
        if message_type == 'subscribe':
            # Handle subscription to events
            topic = data.get('topic')
            await ws.send_str(json.dumps({
                'type': 'subscribed',
                'topic': topic,
                'timestamp': datetime.now().isoformat()
            }))
        elif message_type == 'ping':
            # Handle ping
            await ws.send_str(json.dumps({
                'type': 'pong',
                'timestamp': datetime.now().isoformat()
            }))
        else:
            await ws.send_str(json.dumps({'error': f'Unknown message type: {message_type}'}))
    
    async def _list_services(self, request):
        """List registered services (admin endpoint)."""
        services = await self.service_discovery.list_services()
        return web.json_response({
            'services': [asdict(service) for service in services],
            'total_count': len(services)
        })
    
    async def _get_metrics(self, request):
        """Get API Gateway metrics (admin endpoint)."""
        return web.json_response({
            'rate_limiters': len(self.rate_limiters),
            'circuit_breakers': len(self.circuit_breakers),
            'websocket_connections': len(self.websocket_connections),
            'cache_entries': len(self.request_cache),
            'telemetry': self.observability.export_snapshot(),
            'dashboard': self.observability.load_dashboard_config()
        })
    
    async def _register_service(self, request):
        """Register a new service (admin endpoint)."""
        try:
            data = await request.json()
            registration = ServiceRegistration(**data)
            success = await self.service_discovery.register_service(registration)
            
            if success:
                return web.json_response({'message': 'Service registered successfully'})
            else:
                return web.json_response({'error': 'Failed to register service'}, status=500)
                
        except Exception as e:
            return web.json_response({'error': str(e)}, status=400)
    
    async def broadcast_websocket_message(self, message: Dict[str, Any]):
        """Broadcast message to all WebSocket connections."""
        if not self.websocket_connections:
            return
        
        message_str = json.dumps(message)
        disconnected = set()
        
        for ws in self.websocket_connections:
            try:
                await ws.send_str(message_str)
            except Exception:
                disconnected.add(ws)
        
        # Remove disconnected connections
        for ws in disconnected:
            self.websocket_connections.discard(ws)
    
    async def run(self, host: str = '0.0.0.0', port: int = 8080):
        """Run API Gateway server."""
        await self.initialize()
        
        # Setup SSL if configured
        ssl_context = None
        if self.config.get('ssl_enabled'):
            ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
            ssl_context.load_cert_chain(
                self.config['ssl_cert_path'],
                self.config['ssl_key_path']
            )
        
        runner = web.AppRunner(self.app)
        await runner.setup()
        
        site = web.TCPSite(runner, host, port, ssl_context=ssl_context)
        await site.start()
        
        self.logger.info(f"API Gateway running on {'https' if ssl_context else 'http'}://{host}:{port}")
        
        try:
            await asyncio.Event().wait()  # Run forever
        except KeyboardInterrupt:
            self.logger.info("Shutting down API Gateway")
        finally:
            await self.cleanup()
    
    async def cleanup(self):
        """Cleanup resources."""
        await self.service_discovery.cleanup()
        
        # Close WebSocket connections
        for ws in list(self.websocket_connections):
            await ws.close()


class ServiceMesh:
    """
    Production-grade service mesh for Symbio AI.
    
    Provides service discovery, load balancing, circuit breaking,
    API gateway functionality, and inter-service communication.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.api_gateway = APIGateway(config.get('gateway', {}))
        self.service_discovery = ServiceDiscovery(config.get('redis_url', 'redis://localhost:6379'))
        self.load_balancer = LoadBalancer()
        self.logger = logging.getLogger(__name__)
    
    async def initialize(self):
        """Initialize service mesh."""
        await self.service_discovery.initialize()
        await self.api_gateway.initialize()
        self.logger.info("Service mesh initialized")
    
    async def register_core_services(self):
        """Register core Symbio AI services."""
        core_services = [
            ServiceRegistration(
                name="symbio-training",
                service_type=ServiceType.TRAINING,
                endpoints=[ServiceEndpoint(host="training-service", port=8080)],
                tags=["ai", "training", "core"]
            ),
            ServiceRegistration(
                name="symbio-inference",
                service_type=ServiceType.INFERENCE,
                endpoints=[ServiceEndpoint(host="inference-service", port=8080)],
                tags=["ai", "inference", "core"]
            ),
            ServiceRegistration(
                name="symbio-models",
                service_type=ServiceType.AI_MODEL,
                endpoints=[ServiceEndpoint(host="models-service", port=8080)],
                tags=["ai", "models", "core"]
            ),
            ServiceRegistration(
                name="symbio-data",
                service_type=ServiceType.DATA_PROCESSING,
                endpoints=[ServiceEndpoint(host="data-service", port=8080)],
                tags=["data", "processing", "core"]
            ),
            ServiceRegistration(
                name="symbio-monitoring",
                service_type=ServiceType.MONITORING,
                endpoints=[ServiceEndpoint(host="monitoring-service", port=8080)],
                tags=["monitoring", "metrics", "core"]
            )
        ]
        
        for service in core_services:
            await self.service_discovery.register_service(service)
    
    async def run(self, host: str = '0.0.0.0', port: int = 8080):
        """Run service mesh."""
        await self.initialize()
        await self.register_core_services()
        
        # Run API Gateway
        await self.api_gateway.run(host, port)
    
    async def cleanup(self):
        """Cleanup service mesh resources."""
        await self.api_gateway.cleanup()
        await self.service_discovery.cleanup()
    
    def get_service_mesh_config(self) -> Dict[str, Any]:
        """Get service mesh configuration for export."""
        return {
            "service_mesh_version": "1.0.0",
            "configuration": self.config,
            "registered_services_count": len(self.service_discovery.services),
            "load_balancing_strategy": self.load_balancer.strategy.value,
            "features": [
                "service_discovery",
                "load_balancing",
                "circuit_breaking", 
                "api_gateway",
                "websocket_support",
                "rate_limiting",
                "health_checking",
                "ssl_termination"
            ]
        }