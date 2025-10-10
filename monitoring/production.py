"""
Production-grade logging and monitoring system for Symbio AI.

Provides structured logging, metrics collection, health monitoring,
and alerting capabilities for enterprise deployment.
"""

import asyncio
import logging
import json
import time
import psutil
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, field, asdict
from abc import ABC, abstractmethod
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict, deque
import weakref
import socket
import uuid


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MetricType(Enum):
    """Types of metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class Alert:
    """System alert definition."""
    id: str
    timestamp: str
    severity: AlertSeverity
    component: str
    message: str
    details: Dict[str, Any]
    resolved: bool = False
    resolved_at: Optional[str] = None


@dataclass
class Metric:
    """System metric definition."""
    name: str
    value: Union[int, float]
    metric_type: MetricType
    timestamp: str
    tags: Dict[str, str] = field(default_factory=dict)
    unit: Optional[str] = None


@dataclass
class HealthCheck:
    """Health check result."""
    component: str
    status: str  # healthy, degraded, unhealthy
    timestamp: str
    latency_ms: float
    details: Dict[str, Any] = field(default_factory=dict)


class MetricsCollector:
    """Collects and aggregates system metrics."""
    
    def __init__(self):
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.counters: Dict[str, int] = defaultdict(int)
        self.gauges: Dict[str, float] = defaultdict(float)
        self.histograms: Dict[str, List[float]] = defaultdict(list)
        self.timers: Dict[str, List[float]] = defaultdict(list)
        self.lock = threading.RLock()
        self.logger = logging.getLogger(__name__)
    
    def increment_counter(self, name: str, value: int = 1, tags: Dict[str, str] = None) -> None:
        """Increment a counter metric."""
        with self.lock:
            self.counters[name] += value
            self._record_metric(name, value, MetricType.COUNTER, tags or {})
    
    def set_gauge(self, name: str, value: float, tags: Dict[str, str] = None) -> None:
        """Set a gauge metric value."""
        with self.lock:
            self.gauges[name] = value
            self._record_metric(name, value, MetricType.GAUGE, tags or {})
    
    def record_histogram(self, name: str, value: float, tags: Dict[str, str] = None) -> None:
        """Record a histogram value."""
        with self.lock:
            self.histograms[name].append(value)
            # Keep only last 1000 values per histogram
            if len(self.histograms[name]) > 1000:
                self.histograms[name] = self.histograms[name][-1000:]
            self._record_metric(name, value, MetricType.HISTOGRAM, tags or {})
    
    def record_timer(self, name: str, duration_ms: float, tags: Dict[str, str] = None) -> None:
        """Record a timer value."""
        with self.lock:
            self.timers[name].append(duration_ms)
            # Keep only last 1000 values per timer
            if len(self.timers[name]) > 1000:
                self.timers[name] = self.timers[name][-1000:]
            self._record_metric(name, duration_ms, MetricType.TIMER, tags or {})
    
    def _record_metric(self, name: str, value: float, metric_type: MetricType, tags: Dict[str, str]) -> None:
        """Record a metric for time series storage."""
        metric = Metric(
            name=name,
            value=value,
            metric_type=metric_type,
            timestamp=datetime.now().isoformat(),
            tags=tags
        )
        self.metrics[name].append(metric)
    
    def get_counter(self, name: str) -> int:
        """Get current counter value."""
        return self.counters.get(name, 0)
    
    def get_gauge(self, name: str) -> float:
        """Get current gauge value."""
        return self.gauges.get(name, 0.0)
    
    def get_histogram_stats(self, name: str) -> Dict[str, float]:
        """Get histogram statistics."""
        values = self.histograms.get(name, [])
        if not values:
            return {}
        
        sorted_values = sorted(values)
        count = len(sorted_values)
        
        return {
            "count": count,
            "min": min(sorted_values),
            "max": max(sorted_values),
            "mean": sum(sorted_values) / count,
            "p50": sorted_values[int(count * 0.5)],
            "p95": sorted_values[int(count * 0.95)],
            "p99": sorted_values[int(count * 0.99)]
        }
    
    def get_timer_stats(self, name: str) -> Dict[str, float]:
        """Get timer statistics."""
        return self.get_histogram_stats(name)  # Same logic as histogram
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all current metrics."""
        with self.lock:
            return {
                "counters": dict(self.counters),
                "gauges": dict(self.gauges),
                "histogram_stats": {name: self.get_histogram_stats(name) 
                                   for name in self.histograms.keys()},
                "timer_stats": {name: self.get_timer_stats(name) 
                               for name in self.timers.keys()}
            }


class SystemMonitor:
    """Monitors system resources and performance."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
        self.monitoring = False
        self.monitor_task: Optional[asyncio.Task] = None
        self.logger = logging.getLogger(__name__)
    
    async def start_monitoring(self, interval: float = 30.0) -> None:
        """Start system monitoring."""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_task = asyncio.create_task(self._monitor_loop(interval))
        self.logger.info("System monitoring started")
    
    async def stop_monitoring(self) -> None:
        """Stop system monitoring."""
        self.monitoring = False
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        self.logger.info("System monitoring stopped")
    
    async def _monitor_loop(self, interval: float) -> None:
        """Main monitoring loop."""
        while self.monitoring:
            try:
                await self._collect_system_metrics()
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(interval)
    
    async def _collect_system_metrics(self) -> None:
        """Collect system metrics."""
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        self.metrics.set_gauge("system.cpu.usage_percent", cpu_percent)
        
        cpu_count = psutil.cpu_count()
        self.metrics.set_gauge("system.cpu.count", cpu_count)
        
        # Memory metrics
        memory = psutil.virtual_memory()
        self.metrics.set_gauge("system.memory.total_bytes", memory.total)
        self.metrics.set_gauge("system.memory.available_bytes", memory.available)
        self.metrics.set_gauge("system.memory.used_bytes", memory.used)
        self.metrics.set_gauge("system.memory.usage_percent", memory.percent)
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        self.metrics.set_gauge("system.disk.total_bytes", disk.total)
        self.metrics.set_gauge("system.disk.used_bytes", disk.used)
        self.metrics.set_gauge("system.disk.free_bytes", disk.free)
        self.metrics.set_gauge("system.disk.usage_percent", (disk.used / disk.total) * 100)
        
        # Network metrics
        network = psutil.net_io_counters()
        self.metrics.increment_counter("system.network.bytes_sent", network.bytes_sent)
        self.metrics.increment_counter("system.network.bytes_recv", network.bytes_recv)
        
        # Process metrics
        process = psutil.Process()
        self.metrics.set_gauge("process.cpu_percent", process.cpu_percent())
        self.metrics.set_gauge("process.memory_mb", process.memory_info().rss / 1024 / 1024)
        self.metrics.set_gauge("process.num_threads", process.num_threads())
        
        self.logger.debug("System metrics collected")


class AlertManager:
    """Manages system alerts and notifications."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
        self.alerts: Dict[str, Alert] = {}
        self.alert_rules: List[Callable] = []
        self.alert_handlers: List[Callable] = []
        self.checking = False
        self.check_task: Optional[asyncio.Task] = None
        self.logger = logging.getLogger(__name__)
    
    def add_alert_rule(self, rule_func: Callable[[MetricsCollector], Optional[Alert]]) -> None:
        """Add an alert rule function."""
        self.alert_rules.append(rule_func)
        self.logger.info(f"Added alert rule: {rule_func.__name__}")
    
    def add_alert_handler(self, handler_func: Callable[[Alert], None]) -> None:
        """Add an alert handler function."""
        self.alert_handlers.append(handler_func)
        self.logger.info(f"Added alert handler: {handler_func.__name__}")
    
    async def start_checking(self, interval: float = 60.0) -> None:
        """Start alert checking."""
        if self.checking:
            return
        
        self.checking = True
        self.check_task = asyncio.create_task(self._check_loop(interval))
        self.logger.info("Alert checking started")
    
    async def stop_checking(self) -> None:
        """Stop alert checking."""
        self.checking = False
        if self.check_task:
            self.check_task.cancel()
            try:
                await self.check_task
            except asyncio.CancelledError:
                pass
        self.logger.info("Alert checking stopped")
    
    async def _check_loop(self, interval: float) -> None:
        """Main alert checking loop."""
        while self.checking:
            try:
                await self._check_alerts()
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in alert checking loop: {e}")
                await asyncio.sleep(interval)
    
    async def _check_alerts(self) -> None:
        """Check all alert rules."""
        for rule_func in self.alert_rules:
            try:
                alert = rule_func(self.metrics)
                if alert:
                    await self._handle_alert(alert)
            except Exception as e:
                self.logger.error(f"Error in alert rule {rule_func.__name__}: {e}")
    
    async def _handle_alert(self, alert: Alert) -> None:
        """Handle a new alert."""
        # Check if this alert already exists and is unresolved
        existing_alert = self.alerts.get(alert.id)
        if existing_alert and not existing_alert.resolved:
            return  # Don't duplicate alerts
        
        self.alerts[alert.id] = alert
        
        # Notify all handlers
        for handler in self.alert_handlers:
            try:
                await asyncio.get_event_loop().run_in_executor(None, handler, alert)
            except Exception as e:
                self.logger.error(f"Error in alert handler: {e}")
        
        self.logger.warning(f"Alert triggered: {alert.severity.value} - {alert.message}")
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert."""
        if alert_id in self.alerts:
            self.alerts[alert_id].resolved = True
            self.alerts[alert_id].resolved_at = datetime.now().isoformat()
            self.logger.info(f"Alert resolved: {alert_id}")
            return True
        return False
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active (unresolved) alerts."""
        return [alert for alert in self.alerts.values() if not alert.resolved]
    
    def get_all_alerts(self) -> List[Alert]:
        """Get all alerts."""
        return list(self.alerts.values())


class HealthChecker:
    """Performs health checks on system components."""
    
    def __init__(self):
        self.health_checks: Dict[str, Callable] = {}
        self.health_status: Dict[str, HealthCheck] = {}
        self.checking = False
        self.check_task: Optional[asyncio.Task] = None
        self.logger = logging.getLogger(__name__)
    
    def register_check(self, name: str, check_func: Callable[[], Dict[str, Any]]) -> None:
        """Register a health check function."""
        self.health_checks[name] = check_func
        self.logger.info(f"Registered health check: {name}")
    
    async def start_checking(self, interval: float = 120.0) -> None:
        """Start health checking."""
        if self.checking:
            return
        
        self.checking = True
        self.check_task = asyncio.create_task(self._check_loop(interval))
        self.logger.info("Health checking started")
    
    async def stop_checking(self) -> None:
        """Stop health checking."""
        self.checking = False
        if self.check_task:
            self.check_task.cancel()
            try:
                await self.check_task
            except asyncio.CancelledError:
                pass
        self.logger.info("Health checking stopped")
    
    async def _check_loop(self, interval: float) -> None:
        """Main health checking loop."""
        while self.checking:
            try:
                await self._run_health_checks()
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in health checking loop: {e}")
                await asyncio.sleep(interval)
    
    async def _run_health_checks(self) -> None:
        """Run all health checks."""
        for name, check_func in self.health_checks.items():
            try:
                start_time = time.perf_counter()
                result = await asyncio.get_event_loop().run_in_executor(None, check_func)
                latency_ms = (time.perf_counter() - start_time) * 1000
                
                health_check = HealthCheck(
                    component=name,
                    status=result.get("status", "unknown"),
                    timestamp=datetime.now().isoformat(),
                    latency_ms=latency_ms,
                    details=result
                )
                
                self.health_status[name] = health_check
                
            except Exception as e:
                self.logger.error(f"Health check failed for {name}: {e}")
                self.health_status[name] = HealthCheck(
                    component=name,
                    status="unhealthy",
                    timestamp=datetime.now().isoformat(),
                    latency_ms=0.0,
                    details={"error": str(e)}
                )
    
    async def check_component(self, name: str) -> Optional[HealthCheck]:
        """Check health of a specific component."""
        if name not in self.health_checks:
            return None
        
        try:
            start_time = time.perf_counter()
            result = await asyncio.get_event_loop().run_in_executor(
                None, self.health_checks[name]
            )
            latency_ms = (time.perf_counter() - start_time) * 1000
            
            health_check = HealthCheck(
                component=name,
                status=result.get("status", "unknown"),
                timestamp=datetime.now().isoformat(),
                latency_ms=latency_ms,
                details=result
            )
            
            self.health_status[name] = health_check
            return health_check
            
        except Exception as e:
            health_check = HealthCheck(
                component=name,
                status="unhealthy",
                timestamp=datetime.now().isoformat(),
                latency_ms=0.0,
                details={"error": str(e)}
            )
            self.health_status[name] = health_check
            return health_check
    
    def get_overall_health(self) -> Dict[str, Any]:
        """Get overall system health status."""
        if not self.health_status:
            return {"status": "unknown", "components": {}}
        
        component_statuses = {}
        healthy_count = 0
        total_count = len(self.health_status)
        
        for name, health_check in self.health_status.items():
            component_statuses[name] = {
                "status": health_check.status,
                "timestamp": health_check.timestamp,
                "latency_ms": health_check.latency_ms
            }
            
            if health_check.status == "healthy":
                healthy_count += 1
        
        # Determine overall status
        if healthy_count == total_count:
            overall_status = "healthy"
        elif healthy_count >= total_count * 0.75:
            overall_status = "degraded"
        else:
            overall_status = "unhealthy"
        
        return {
            "status": overall_status,
            "healthy_components": healthy_count,
            "total_components": total_count,
            "health_ratio": healthy_count / total_count if total_count > 0 else 0.0,
            "components": component_statuses,
            "last_check": datetime.now().isoformat()
        }


class PerformanceProfiler:
    """Performance profiling and timing utilities."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
        self.active_timers: Dict[str, float] = {}
        self.logger = logging.getLogger(__name__)
    
    def start_timer(self, name: str) -> str:
        """Start a performance timer."""
        timer_id = f"{name}_{uuid.uuid4().hex[:8]}"
        self.active_timers[timer_id] = time.perf_counter()
        return timer_id
    
    def end_timer(self, timer_id: str, metric_name: Optional[str] = None) -> float:
        """End a performance timer and record the duration."""
        if timer_id not in self.active_timers:
            self.logger.warning(f"Timer {timer_id} not found")
            return 0.0
        
        start_time = self.active_timers.pop(timer_id)
        duration_ms = (time.perf_counter() - start_time) * 1000
        
        if metric_name:
            self.metrics.record_timer(metric_name, duration_ms)
        
        return duration_ms
    
    def time_context(self, name: str):
        """Context manager for timing operations."""
        return TimingContext(self, name)


class TimingContext:
    """Context manager for performance timing."""
    
    def __init__(self, profiler: PerformanceProfiler, name: str):
        self.profiler = profiler
        self.name = name
        self.timer_id: Optional[str] = None
        self.duration_ms: float = 0.0
    
    def __enter__(self):
        self.timer_id = self.profiler.start_timer(self.name)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.timer_id:
            self.duration_ms = self.profiler.end_timer(self.timer_id, self.name)


class ProductionLogger:
    """Production-grade structured logging system."""
    
    def __init__(self, config: Union[Dict[str, Any], str] = None):
        # Handle both dict config and string name
        if isinstance(config, str):
            # If string provided (logger name), create default config
            self.name = config
            self.config = {'level': 'INFO', 'file': 'logs/symbio_ai.log'}
        elif isinstance(config, dict):
            self.name = config.get('name', __name__)
            self.config = config
        else:
            # Default config
            self.name = __name__
            self.config = {'level': 'INFO', 'file': 'logs/symbio_ai.log'}
        
        self.setup_logging()
        self._logger = logging.getLogger(self.name)
        self.request_id_var = None
        self.correlation_id_var = None
    
    def setup_logging(self) -> None:
        """Setup structured logging configuration."""
        # Create formatter for structured logs
        formatter = StructuredFormatter(
            fmt='%(asctime)s %(name)s %(levelname)s %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Setup root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, self.config.get('level', 'INFO')))
        
        # Console handler (avoid duplicate attachment)
        console_handler_exists = any(
            isinstance(handler, logging.StreamHandler) and getattr(handler, "_symbio_structured", False)
            for handler in root_logger.handlers
        )
        if not console_handler_exists:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            console_handler._symbio_structured = True  # Mark to prevent duplicates
            root_logger.addHandler(console_handler)
        
        # File handler
        log_file = self.config.get('file', 'logs/symbio_ai.log')
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        resolved_log_path = str(log_path.resolve())
        file_handler_exists = any(
            isinstance(handler, logging.FileHandler)
            and getattr(handler, "_symbio_structured", False)
            and getattr(handler, "baseFilename", None) == resolved_log_path
            for handler in root_logger.handlers
        )

        if not file_handler_exists:
            file_handler = logging.FileHandler(resolved_log_path)
            file_handler.setFormatter(formatter)
            file_handler._symbio_structured = True
            root_logger.addHandler(file_handler)
        
        # Disable some noisy loggers
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        logging.getLogger('asyncio').setLevel(logging.WARNING)
    
    def get_logger(self, name: str = None) -> logging.Logger:
        """Get a logger with structured formatting."""
        return logging.getLogger(name or self.name)

    def __getattr__(self, item):
        """Proxy attribute access to the underlying standard logger."""
        return getattr(self._logger, item)


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured logging."""
    
    def format(self, record):
        """Format log record as structured JSON."""
        # Get the basic formatted message
        message = super().format(record)
        
        # Create structured log entry
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'thread': record.thread,
            'process': record.process
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Add any custom attributes
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                          'filename', 'module', 'exc_info', 'exc_text', 'stack_info',
                          'lineno', 'funcName', 'created', 'msecs', 'relativeCreated',
                          'thread', 'threadName', 'processName', 'process', 'message']:
                log_entry[key] = value
        
        return json.dumps(log_entry)


# Alert rule functions
def high_cpu_alert_rule(metrics: MetricsCollector) -> Optional[Alert]:
    """Alert rule for high CPU usage."""
    cpu_usage = metrics.get_gauge("system.cpu.usage_percent")
    if cpu_usage > 90:
        return Alert(
            id="high_cpu_usage",
            timestamp=datetime.now().isoformat(),
            severity=AlertSeverity.CRITICAL,
            component="system",
            message=f"High CPU usage detected: {cpu_usage:.1f}%",
            details={"cpu_usage_percent": cpu_usage, "threshold": 90}
        )
    return None


def high_memory_alert_rule(metrics: MetricsCollector) -> Optional[Alert]:
    """Alert rule for high memory usage."""
    memory_usage = metrics.get_gauge("system.memory.usage_percent")
    if memory_usage > 85:
        return Alert(
            id="high_memory_usage",
            timestamp=datetime.now().isoformat(),
            severity=AlertSeverity.WARNING if memory_usage < 95 else AlertSeverity.CRITICAL,
            component="system",
            message=f"High memory usage detected: {memory_usage:.1f}%",
            details={"memory_usage_percent": memory_usage, "threshold": 85}
        )
    return None


def disk_space_alert_rule(metrics: MetricsCollector) -> Optional[Alert]:
    """Alert rule for low disk space."""
    disk_usage = metrics.get_gauge("system.disk.usage_percent")
    if disk_usage > 80:
        return Alert(
            id="low_disk_space",
            timestamp=datetime.now().isoformat(),
            severity=AlertSeverity.WARNING if disk_usage < 90 else AlertSeverity.CRITICAL,
            component="system",
            message=f"Low disk space: {disk_usage:.1f}% used",
            details={"disk_usage_percent": disk_usage, "threshold": 80}
        )
    return None


# Alert handlers
def log_alert_handler(alert: Alert) -> None:
    """Log alert to system logger."""
    logger = logging.getLogger("alerts")
    log_level = {
        AlertSeverity.INFO: logging.INFO,
        AlertSeverity.WARNING: logging.WARNING,
        AlertSeverity.ERROR: logging.ERROR,
        AlertSeverity.CRITICAL: logging.CRITICAL
    }.get(alert.severity, logging.WARNING)
    
    logger.log(log_level, f"ALERT: {alert.message}", extra={
        "alert_id": alert.id,
        "component": alert.component,
        "severity": alert.severity.value,
        "details": alert.details
    })


def webhook_alert_handler(alert: Alert) -> None:
    """Send alert to webhook endpoint."""
    # In a real implementation, this would make an HTTP request
    # to a webhook URL (Slack, PagerDuty, etc.)
    logger = logging.getLogger("alerts.webhook")
    logger.info(f"Would send webhook for alert: {alert.id}")


class MonitoringSystem:
    """
    Production-grade monitoring system for Symbio AI.
    
    Integrates metrics collection, alerting, health checking,
    and performance profiling into a unified monitoring solution.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize components
        self.metrics = MetricsCollector()
        self.system_monitor = SystemMonitor(self.metrics)
        self.alert_manager = AlertManager(self.metrics)
        self.health_checker = HealthChecker()
        self.profiler = PerformanceProfiler(self.metrics)
        self.logger_setup = ProductionLogger(config.get('logging', {}))
        
        self.logger = logging.getLogger(__name__)
        self.running = False
        
        # Setup default alert rules
        self._setup_default_alert_rules()
        
        # Setup default health checks
        self._setup_default_health_checks()
    
    def _setup_default_alert_rules(self) -> None:
        """Setup default system alert rules."""
        self.alert_manager.add_alert_rule(high_cpu_alert_rule)
        self.alert_manager.add_alert_rule(high_memory_alert_rule)
        self.alert_manager.add_alert_rule(disk_space_alert_rule)
        
        # Add default alert handlers
        self.alert_manager.add_alert_handler(log_alert_handler)
        if self.config.get('webhook_url'):
            self.alert_manager.add_alert_handler(webhook_alert_handler)
    
    def _setup_default_health_checks(self) -> None:
        """Setup default system health checks."""
        def system_health_check():
            """Basic system health check."""
            try:
                # Check if system is responsive
                cpu_usage = psutil.cpu_percent(interval=0.1)
                memory = psutil.virtual_memory()
                
                if cpu_usage > 95 or memory.percent > 95:
                    return {"status": "unhealthy", "reason": "high_resource_usage"}
                elif cpu_usage > 80 or memory.percent > 80:
                    return {"status": "degraded", "reason": "moderate_resource_usage"}
                else:
                    return {"status": "healthy"}
                    
            except Exception as e:
                return {"status": "unhealthy", "error": str(e)}
        
        def network_health_check():
            """Network connectivity health check."""
            try:
                # Simple socket test
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(5)
                result = sock.connect_ex(('8.8.8.8', 53))
                sock.close()
                
                if result == 0:
                    return {"status": "healthy"}
                else:
                    return {"status": "unhealthy", "reason": "network_connectivity"}
                    
            except Exception as e:
                return {"status": "unhealthy", "error": str(e)}
        
        self.health_checker.register_check("system", system_health_check)
        self.health_checker.register_check("network", network_health_check)
    
    async def start(self) -> None:
        """Start the monitoring system."""
        if self.running:
            return
        
        self.logger.info("Starting monitoring system")
        self.running = True
        
        # Start all monitoring components
        await asyncio.gather(
            self.system_monitor.start_monitoring(interval=30.0),
            self.alert_manager.start_checking(interval=60.0),
            self.health_checker.start_checking(interval=120.0)
        )
        
        self.logger.info("Monitoring system started")
    
    async def stop(self) -> None:
        """Stop the monitoring system."""
        if not self.running:
            return
        
        self.logger.info("Stopping monitoring system")
        self.running = False
        
        # Stop all monitoring components
        await asyncio.gather(
            self.system_monitor.stop_monitoring(),
            self.alert_manager.stop_checking(),
            self.health_checker.stop_checking()
        )
        
        self.logger.info("Monitoring system stopped")
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive monitoring data for dashboard."""
        return {
            "metrics": self.metrics.get_all_metrics(),
            "health": self.health_checker.get_overall_health(),
            "active_alerts": [asdict(alert) for alert in self.alert_manager.get_active_alerts()],
            "system_info": {
                "hostname": socket.gethostname(),
                "timestamp": datetime.now().isoformat(),
                "uptime": time.time() - self.config.get('start_time', time.time())
            }
        }
    
    def export_metrics(self, format: str = "json") -> str:
        """Export metrics in specified format."""
        if format == "json":
            data = {
                "timestamp": datetime.now().isoformat(),
                "metrics": self.metrics.get_all_metrics(),
                "health": self.health_checker.get_overall_health()
            }
            return json.dumps(data, indent=2)
        elif format == "prometheus":
            # In a real implementation, this would format metrics for Prometheus
            return "# Prometheus metrics would be formatted here"
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    # Context managers and decorators for easy instrumentation
    def time_operation(self, operation_name: str):
        """Context manager for timing operations."""
        return self.profiler.time_context(operation_name)
    
    def increment_counter(self, name: str, value: int = 1, **tags) -> None:
        """Increment a counter metric."""
        self.metrics.increment_counter(name, value, tags)
    
    def set_gauge(self, name: str, value: float, **tags) -> None:
        """Set a gauge metric."""
        self.metrics.set_gauge(name, value, tags)
    
    def record_histogram(self, name: str, value: float, **tags) -> None:
        """Record a histogram value."""
        self.metrics.record_histogram(name, value, tags)