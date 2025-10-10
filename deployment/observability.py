#!/usr/bin/env python3
"""
Observability Module - Monitoring and Metrics Collection
Provides centralized observability for all Symbio AI components.
"""

import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict, deque
import threading


@dataclass
class Metric:
    """A single metric observation."""
    name: str
    value: float
    timestamp: float = field(default_factory=time.time)
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class Event:
    """A system event."""
    event_type: str
    message: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    severity: str = "info"  # info, warning, error, critical


class ObservabilitySystem:
    """
    Central observability system for metrics, events, and traces.
    Thread-safe singleton implementation.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self._initialized = True
        self._lock = threading.Lock()
        
        # Metrics storage (last 1000 per metric)
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Events storage (last 1000)
        self.events: deque = deque(maxlen=1000)
        
        # Counters
        self.counters: Dict[str, int] = defaultdict(int)
        
        # Timers (for tracking durations)
        self._active_timers: Dict[str, float] = {}
        
        # System stats
        self.start_time = time.time()
        self.total_operations = 0
        self.total_errors = 0
    
    def record_metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Record a metric value."""
        with self._lock:
            metric = Metric(name=name, value=value, tags=tags or {})
            self.metrics[name].append(metric)
            self.total_operations += 1
    
    def increment_counter(self, name: str, amount: int = 1):
        """Increment a counter."""
        with self._lock:
            self.counters[name] += amount
    
    def emit_counter(self, name: str, value: int = 1, tags: Optional[Dict[str, str]] = None, **kwargs):
        """Emit a counter metric (alias for increment_counter for compatibility)."""
        # Ignore extra kwargs for compatibility
        self.increment_counter(name, value)
    
    def emit_gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None, **kwargs):
        """Emit a gauge metric (alias for record_metric for compatibility)."""
        # Ignore extra kwargs for compatibility
        self.record_metric(name, value, tags)
    
    def emit_histogram(self, name: str, value: float, tags: Optional[Dict[str, str]] = None, **kwargs):
        """Emit a histogram metric (alias for record_metric for compatibility)."""
        # Ignore extra kwargs for compatibility
        self.record_metric(name, value, tags)
    
    def record_event(self, event_type: str, message: str, 
                    severity: str = "info", metadata: Optional[Dict] = None):
        """Record an event."""
        with self._lock:
            event = Event(
                event_type=event_type,
                message=message,
                severity=severity,
                metadata=metadata or {}
            )
            self.events.append(event)
            
            if severity in ("error", "critical"):
                self.total_errors += 1
    
    def start_timer(self, name: str):
        """Start a timer for measuring duration."""
        with self._lock:
            self._active_timers[name] = time.time()
    
    def stop_timer(self, name: str, tags: Optional[Dict[str, str]] = None):
        """Stop a timer and record the duration."""
        with self._lock:
            if name in self._active_timers:
                duration = time.time() - self._active_timers[name]
                self.record_metric(f"{name}_duration_ms", duration * 1000, tags)
                del self._active_timers[name]
                return duration
            return None
    
    def get_metric_stats(self, name: str) -> Dict[str, float]:
        """Get statistics for a metric."""
        with self._lock:
            if name not in self.metrics or not self.metrics[name]:
                return {}
            
            values = [m.value for m in self.metrics[name]]
            return {
                'count': len(values),
                'mean': sum(values) / len(values),
                'min': min(values),
                'max': max(values),
                'latest': values[-1]
            }
    
    def get_counter(self, name: str) -> int:
        """Get current counter value."""
        with self._lock:
            return self.counters.get(name, 0)
    
    def get_recent_events(self, limit: int = 100, 
                         severity: Optional[str] = None) -> List[Event]:
        """Get recent events, optionally filtered by severity."""
        with self._lock:
            events = list(self.events)
            if severity:
                events = [e for e in events if e.severity == severity]
            return events[-limit:]
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health metrics."""
        with self._lock:
            uptime = time.time() - self.start_time
            error_rate = self.total_errors / max(self.total_operations, 1)
            
            return {
                'status': 'healthy' if error_rate < 0.01 else 'degraded',
                'uptime_seconds': uptime,
                'total_operations': self.total_operations,
                'total_errors': self.total_errors,
                'error_rate': error_rate,
                'active_timers': len(self._active_timers),
                'metrics_tracked': len(self.metrics),
                'events_recorded': len(self.events)
            }
    
    def reset(self):
        """Reset all metrics and counters (for testing)."""
        with self._lock:
            self.metrics.clear()
            self.events.clear()
            self.counters.clear()
            self._active_timers.clear()
            self.total_operations = 0
            self.total_errors = 0
            self.start_time = time.time()
    
    def export_metrics(self, format: str = 'prometheus') -> str:
        """Export metrics in specified format."""
        with self._lock:
            if format == 'prometheus':
                lines = []
                
                # Export counters
                for name, value in self.counters.items():
                    lines.append(f"# TYPE {name} counter")
                    lines.append(f"{name} {value}")
                
                # Export gauges (latest metric values)
                for name, metrics in self.metrics.items():
                    if metrics:
                        latest = metrics[-1]
                        lines.append(f"# TYPE {name} gauge")
                        tag_str = ','.join(f'{k}="{v}"' for k, v in latest.tags.items())
                        tag_str = f"{{{tag_str}}}" if tag_str else ""
                        lines.append(f"{name}{tag_str} {latest.value}")
                
                return '\n'.join(lines)
            
            return ""


# Global singleton instance
OBSERVABILITY = ObservabilitySystem()


# Convenience functions
def record_metric(name: str, value: float, tags: Optional[Dict[str, str]] = None):
    """Record a metric (convenience function)."""
    OBSERVABILITY.record_metric(name, value, tags)


def increment_counter(name: str, amount: int = 1):
    """Increment a counter (convenience function)."""
    OBSERVABILITY.increment_counter(name, amount)


def record_event(event_type: str, message: str, severity: str = "info", 
                metadata: Optional[Dict] = None):
    """Record an event (convenience function)."""
    OBSERVABILITY.record_event(event_type, message, severity, metadata)


def start_timer(name: str):
    """Start a timer (convenience function)."""
    OBSERVABILITY.start_timer(name)


def stop_timer(name: str, tags: Optional[Dict[str, str]] = None) -> Optional[float]:
    """Stop a timer (convenience function)."""
    return OBSERVABILITY.stop_timer(name, tags)


class TimerContext:
    """Context manager for timing operations."""
    
    def __init__(self, name: str, tags: Optional[Dict[str, str]] = None):
        self.name = name
        self.tags = tags
    
    def __enter__(self):
        start_timer(self.name)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        stop_timer(self.name, self.tags)
        if exc_type is not None:
            record_event(
                'timer_error',
                f"Timer {self.name} encountered error: {exc_val}",
                severity='error'
            )
        return False


# Decorator for timing functions
def timed(metric_name: Optional[str] = None):
    """Decorator to time function execution."""
    def decorator(func):
        name = metric_name or f"{func.__module__}.{func.__name__}"
        
        def wrapper(*args, **kwargs):
            with TimerContext(name):
                return func(*args, **kwargs)
        
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper
    
    return decorator


if __name__ == "__main__":
    # Demo usage
    print("Observability System Demo")
    print("=" * 60)
    
    # Record some metrics
    record_metric("cpu_usage", 45.2, tags={"host": "server1"})
    record_metric("memory_usage", 78.5, tags={"host": "server1"})
    record_metric("request_latency_ms", 125.3)
    
    # Increment counters
    increment_counter("requests_total", 100)
    increment_counter("errors_total", 5)
    
    # Record events
    record_event("system_start", "System initialized successfully")
    record_event("warning", "High memory usage detected", severity="warning")
    
    # Time an operation
    with TimerContext("database_query"):
        time.sleep(0.1)  # Simulate work
    
    # Get stats
    print("\nMetric Stats:")
    print(OBSERVABILITY.get_metric_stats("cpu_usage"))
    
    print("\nSystem Health:")
    print(OBSERVABILITY.get_system_health())
    
    print("\nPrometheus Export:")
    print(OBSERVABILITY.export_metrics('prometheus'))
    
    print("\n✅ Observability system working!")
