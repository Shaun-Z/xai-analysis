"""
Hardware performance monitoring utilities for XAI analysis.
"""

import time
import psutil
import threading
from typing import Dict, List, Optional
from dataclasses import dataclass
import pandas as pd

try:
    import pynvml
    NVIDIA_AVAILABLE = True
except ImportError:
    NVIDIA_AVAILABLE = False
    pynvml = None


@dataclass
class HardwareMetrics:
    """Container for hardware performance metrics."""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_total_gb: float
    gpu_utilization: Optional[float] = None
    gpu_memory_used_mb: Optional[float] = None
    gpu_memory_total_mb: Optional[float] = None
    gpu_power_watts: Optional[float] = None
    gpu_temperature_c: Optional[float] = None


class HardwareMonitor:
    """Monitor hardware performance metrics during XAI computations."""
    
    def __init__(self, sampling_interval: float = 0.1):
        """
        Initialize hardware monitor.
        
        Args:
            sampling_interval: Time between measurements in seconds
        """
        self.sampling_interval = sampling_interval
        self.monitoring = False
        self.metrics: List[HardwareMetrics] = []
        self.monitor_thread = None
        
        # Initialize NVIDIA monitoring if available
        self.gpu_available = False
        if NVIDIA_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                self.gpu_available = True
            except Exception:
                self.gpu_available = False
    
    def _collect_metrics(self) -> HardwareMetrics:
        """Collect current hardware metrics."""
        # CPU and Memory metrics
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        
        metrics = HardwareMetrics(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_gb=memory.used / (1024**3),
            memory_total_gb=memory.total / (1024**3)
        )
        
        # GPU metrics if available
        if self.gpu_available:
            try:
                gpu_util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                power_watts = pynvml.nvmlDeviceGetPowerUsage(self.gpu_handle) / 1000.0
                temp_c = pynvml.nvmlDeviceGetTemperature(self.gpu_handle, pynvml.NVML_TEMPERATURE_GPU)
                
                metrics.gpu_utilization = gpu_util.gpu
                metrics.gpu_memory_used_mb = memory_info.used / (1024**2)
                metrics.gpu_memory_total_mb = memory_info.total / (1024**2)
                metrics.gpu_power_watts = power_watts
                metrics.gpu_temperature_c = temp_c
            except Exception as e:
                print(f"Warning: Could not collect GPU metrics: {e}")
        
        return metrics
    
    def _monitoring_loop(self):
        """Background monitoring loop."""
        while self.monitoring:
            try:
                metrics = self._collect_metrics()
                self.metrics.append(metrics)
                time.sleep(self.sampling_interval)
            except Exception as e:
                print(f"Error collecting metrics: {e}")
                time.sleep(self.sampling_interval)
    
    def start_monitoring(self):
        """Start background monitoring."""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.metrics = []
        self.monitor_thread = threading.Thread(target=self._monitoring_loop)
        self.monitor_thread.start()
    
    def stop_monitoring(self) -> List[HardwareMetrics]:
        """Stop monitoring and return collected metrics."""
        if not self.monitoring:
            return self.metrics
        
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        
        return self.metrics
    
    def get_summary_stats(self) -> Dict[str, float]:
        """Get summary statistics for collected metrics."""
        if not self.metrics:
            return {}
        
        df = self.to_dataframe()
        
        summary = {
            'duration_seconds': df['timestamp'].max() - df['timestamp'].min(),
            'avg_cpu_percent': df['cpu_percent'].mean(),
            'max_cpu_percent': df['cpu_percent'].max(),
            'avg_memory_percent': df['memory_percent'].mean(),
            'max_memory_percent': df['memory_percent'].max(),
            'avg_memory_used_gb': df['memory_used_gb'].mean(),
            'max_memory_used_gb': df['memory_used_gb'].max(),
        }
        
        if self.gpu_available and 'gpu_utilization' in df.columns:
            gpu_columns = [col for col in df.columns if col.startswith('gpu_') and df[col].notna().any()]
            for col in gpu_columns:
                summary[f'avg_{col}'] = df[col].mean()
                summary[f'max_{col}'] = df[col].max()
        
        return summary
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert metrics to pandas DataFrame."""
        if not self.metrics:
            return pd.DataFrame()
        
        data = []
        for metric in self.metrics:
            row = {
                'timestamp': metric.timestamp,
                'cpu_percent': metric.cpu_percent,
                'memory_percent': metric.memory_percent,
                'memory_used_gb': metric.memory_used_gb,
                'memory_total_gb': metric.memory_total_gb,
            }
            
            if metric.gpu_utilization is not None:
                row.update({
                    'gpu_utilization': metric.gpu_utilization,
                    'gpu_memory_used_mb': metric.gpu_memory_used_mb,
                    'gpu_memory_total_mb': metric.gpu_memory_total_mb,
                    'gpu_power_watts': metric.gpu_power_watts,
                    'gpu_temperature_c': metric.gpu_temperature_c,
                })
            
            data.append(row)
        
        return pd.DataFrame(data)


class BenchmarkTimer:
    """Context manager for timing operations with hardware monitoring."""
    
    def __init__(self, monitor: HardwareMonitor, operation_name: str):
        self.monitor = monitor
        self.operation_name = operation_name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        self.monitor.start_monitoring()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        self.monitor.stop_monitoring()
    
    @property
    def duration(self) -> float:
        """Get operation duration in seconds."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0.0
    
    def get_throughput(self, num_items: int) -> float:
        """Calculate throughput (items per second)."""
        if self.duration > 0:
            return num_items / self.duration
        return 0.0