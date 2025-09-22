#!/usr/bin/env python3
"""
Hardware Performance Benchmark for XAI Methods
Evaluates power consumption, throughput, and memory access patterns for each XAI method.
"""

import argparse
import importlib
import sys
import json
import time
import psutil
import threading
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Captum imports
from captum.attr import (
    IntegratedGradients,
    Saliency,
    GradientShap,
    DeepLift,
    DeepLiftShap,
    GuidedGradCam,
    LayerGradCam,
    Occlusion,
    LRP,
    InputXGradient,
    GuidedBackprop,
    Deconvolution
)

# Memory profiling
try:
    import tracemalloc
    import psutil
    HAS_MEMORY_PROFILING = True
except ImportError:
    HAS_MEMORY_PROFILING = False

# GPU monitoring
try:
    import pynvml
    HAS_NVIDIA_ML = True
    pynvml.nvmlInit()
except ImportError:
    HAS_NVIDIA_ML = False

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))


class HardwareMonitor:
    """Monitor hardware metrics during XAI method execution."""
    
    def __init__(self, gpu_index=0, sampling_interval=0.1):
        self.gpu_index = gpu_index
        self.sampling_interval = sampling_interval
        self.monitoring = False
        self.metrics = defaultdict(list)
        self.monitor_thread = None
        
        # Initialize GPU monitoring if available
        self.has_gpu = HAS_NVIDIA_ML and torch.cuda.is_available()
        if self.has_gpu:
            try:
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
                self.gpu_name = pynvml.nvmlDeviceGetName(self.gpu_handle).decode('utf-8')
            except Exception as e:
                print(f"Warning: Could not initialize GPU monitoring: {e}")
                self.has_gpu = False
    
    def _monitor_loop(self):
        """Continuous monitoring loop running in separate thread."""
        while self.monitoring:
            timestamp = time.time()
            
            # CPU metrics
            cpu_percent = psutil.cpu_percent()
            memory_info = psutil.virtual_memory()
            
            self.metrics['timestamp'].append(timestamp)
            self.metrics['cpu_percent'].append(cpu_percent)
            self.metrics['memory_used_gb'].append(memory_info.used / (1024**3))
            self.metrics['memory_percent'].append(memory_info.percent)
            
            # GPU metrics if available
            if self.has_gpu:
                try:
                    # GPU utilization
                    gpu_util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                    self.metrics['gpu_util_percent'].append(gpu_util.gpu)
                    self.metrics['gpu_memory_util_percent'].append(gpu_util.memory)
                    
                    # GPU memory
                    gpu_memory = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                    self.metrics['gpu_memory_used_gb'].append(gpu_memory.used / (1024**3))
                    self.metrics['gpu_memory_total_gb'].append(gpu_memory.total / (1024**3))
                    
                    # GPU power (if supported)
                    try:
                        power_draw = pynvml.nvmlDeviceGetPowerUsage(self.gpu_handle) / 1000.0  # Convert to Watts
                        self.metrics['gpu_power_watts'].append(power_draw)
                    except pynvml.NVMLError:
                        self.metrics['gpu_power_watts'].append(0)
                    
                    # GPU temperature
                    try:
                        temp = pynvml.nvmlDeviceGetTemperature(self.gpu_handle, pynvml.NVML_TEMPERATURE_GPU)
                        self.metrics['gpu_temp_celsius'].append(temp)
                    except pynvml.NVMLError:
                        self.metrics['gpu_temp_celsius'].append(0)
                        
                except Exception as e:
                    print(f"Warning: GPU monitoring error: {e}")
            
            time.sleep(self.sampling_interval)
    
    def start_monitoring(self):
        """Start hardware monitoring in background thread."""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.metrics.clear()
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop hardware monitoring and return collected metrics."""
        if not self.monitoring:
            return {}
        
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        
        # Calculate statistics
        stats = {}
        for metric, values in self.metrics.items():
            if values and metric != 'timestamp':
                stats[f"{metric}_mean"] = np.mean(values)
                stats[f"{metric}_max"] = np.max(values)
                stats[f"{metric}_min"] = np.min(values)
                stats[f"{metric}_std"] = np.std(values)
        
        return {
            'raw_metrics': dict(self.metrics),
            'statistics': stats,
            'duration': self.metrics['timestamp'][-1] - self.metrics['timestamp'][0] if self.metrics['timestamp'] else 0
        }


class XAIPerformanceBenchmark:
    """Benchmark XAI methods for hardware performance."""
    
    def __init__(self, model, device, data_loader):
        self.model = model
        self.device = device
        self.data_loader = data_loader
        self.model.eval()
        
        # Available XAI methods
        self.xai_methods = {
            'integrated_gradients': IntegratedGradients,
            'saliency': Saliency,
            'gradient_shap': GradientShap,
            'deeplift': DeepLift,
            'deeplift_shap': DeepLiftShap,
            'gradcam': self._create_gradcam,
            'guided_gradcam': self._create_guided_gradcam,
            'occlusion': Occlusion,
            'lrp': LRP,
            'input_x_gradient': InputXGradient,
            'guided_backprop': GuidedBackprop,
            'deconvolution': Deconvolution
        }
        
        self.results = {}
    
    def _create_gradcam(self, model):
        """Create GradCAM instance with appropriate target layer."""
        if hasattr(model, 'features'):
            target_layer = model.features[-1]
        else:
            conv_layers = [module for module in model.modules() if isinstance(module, nn.Conv2d)]
            if conv_layers:
                target_layer = conv_layers[-1]
            else:
                raise ValueError("Could not find convolutional layer for GradCAM")
        return LayerGradCam(model, target_layer)
    
    def _create_guided_gradcam(self, model):
        """Create Guided GradCAM instance with appropriate target layer."""
        if hasattr(model, 'features'):
            target_layer = model.features[-1]
        else:
            conv_layers = [module for module in model.modules() if isinstance(module, nn.Conv2d)]
            if conv_layers:
                target_layer = conv_layers[-1]
            else:
                raise ValueError("Could not find convolutional layer for Guided GradCAM")
        return GuidedGradCam(model, target_layer)
    
    def _get_baseline(self, input_tensor, baseline_type="zero"):
        """Generate baseline for attribution methods."""
        if baseline_type == "zero":
            return torch.zeros_like(input_tensor)
        elif baseline_type == "random":
            return torch.randn_like(input_tensor) * 0.1
        elif baseline_type == "mean":
            # Use dataset mean as baseline (assuming normalized data)
            return torch.zeros_like(input_tensor)
        else:
            raise ValueError(f"Unknown baseline type: {baseline_type}")
    
    def _compute_attribution(self, method, input_tensor, target, method_name):
        """Compute attribution with method-specific parameters."""
        baseline = self._get_baseline(input_tensor)
        
        try:
            if 'integrated_gradients' in method_name.lower():
                attribution = method.attribute(
                    input_tensor, 
                    baselines=baseline, 
                    target=target, 
                    n_steps=50
                )
            elif 'gradcam' in method_name.lower():
                attribution = method.attribute(input_tensor, target=target)
                if attribution.shape != input_tensor.shape:
                    attribution = torch.nn.functional.interpolate(
                        attribution, 
                        size=input_tensor.shape[-2:], 
                        mode='bilinear', 
                        align_corners=False
                    )
            elif 'occlusion' in method_name.lower():
                attribution = method.attribute(
                    input_tensor,
                    target=target,
                    sliding_window_shapes=(3, 8, 8),
                    strides=(3, 4, 4)
                )
            elif 'gradient_shap' in method_name.lower():
                rand_baseline = torch.randn_like(input_tensor) * 0.1
                attribution = method.attribute(
                    input_tensor,
                    baselines=rand_baseline,
                    target=target,
                    n_samples=50
                )
            elif 'deeplift_shap' in method_name.lower():
                # DeepLift SHAP requires baselines
                attribution = method.attribute(
                    input_tensor,
                    baselines=baseline,
                    target=target
                )
            elif 'lrp' in method_name.lower():
                # Skip LRP for models with BatchNorm - it requires special handling
                print(f"Skipping {method_name} - requires model-specific LRP rules for BatchNorm layers")
                return None
            else:
                # Standard methods
                attribution = method.attribute(input_tensor, target=target)
            
            return attribution
            
        except Exception as e:
            print(f"Error computing attribution for {method_name}: {e}")
            return None
    
    def benchmark_method(self, method_name, num_samples=50, warmup_runs=5):
        """Benchmark a specific XAI method."""
        print(f"\nBenchmarking {method_name}...")
        
        # Skip methods that might not be available
        if method_name not in self.xai_methods:
            print(f"Method {method_name} not available")
            return None
        
        try:
            # Create method instance
            if callable(self.xai_methods[method_name]):
                if method_name in ['gradcam', 'guided_gradcam']:
                    method = self.xai_methods[method_name](self.model)
                else:
                    method = self.xai_methods[method_name](self.model)
            else:
                method = self.xai_methods[method_name](self.model)
            
        except Exception as e:
            print(f"Failed to create method {method_name}: {e}")
            return None
        
        # Prepare data
        data_iter = iter(self.data_loader)
        samples = []
        for i in range(min(num_samples + warmup_runs, len(self.data_loader))):
            try:
                batch = next(data_iter)
                samples.append(batch)
            except StopIteration:
                break
        
        if len(samples) < warmup_runs + 1:
            print(f"Not enough samples for benchmarking {method_name}")
            return None
        
        # Warmup runs
        print(f"  Performing {warmup_runs} warmup runs...")
        for i in range(warmup_runs):
            inputs, labels = samples[i]
            inputs = inputs[:1].to(self.device)  # Single sample
            labels = labels[:1].to(self.device)
            
            with torch.no_grad():
                predictions = self.model(inputs)
                target = predictions.argmax(dim=1)
            
            try:
                _ = self._compute_attribution(method, inputs, target, method_name)
            except Exception as e:
                print(f"Warmup failed for {method_name}: {e}")
        
        # Actual benchmark
        monitor = HardwareMonitor()
        
        # Memory monitoring
        if HAS_MEMORY_PROFILING:
            tracemalloc.start()
        
        # Record initial GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            initial_gpu_memory = torch.cuda.memory_allocated() / (1024**3)
        else:
            initial_gpu_memory = 0
        
        # Start monitoring
        monitor.start_monitoring()
        
        # Benchmark timing
        times = []
        successful_runs = 0
        
        print(f"  Running {min(num_samples, len(samples) - warmup_runs)} benchmark iterations...")
        
        start_time = time.time()
        
        for i in range(warmup_runs, min(len(samples), warmup_runs + num_samples)):
            inputs, labels = samples[i]
            inputs = inputs[:1].to(self.device)  # Single sample
            labels = labels[:1].to(self.device)
            
            # Get prediction
            with torch.no_grad():
                predictions = self.model(inputs)
                target = predictions.argmax(dim=1)
            
            # Time the attribution computation
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            iter_start = time.perf_counter()
            
            try:
                attribution = self._compute_attribution(method, inputs, target, method_name)
                if attribution is not None:
                    successful_runs += 1
            except Exception as e:
                print(f"  Attribution failed on iteration {i-warmup_runs}: {e}")
                continue
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            iter_end = time.perf_counter()
            
            times.append(iter_end - iter_start)
        
        total_time = time.time() - start_time
        
        # Stop monitoring
        hardware_metrics = monitor.stop_monitoring()
        
        # Memory usage
        if torch.cuda.is_available():
            peak_gpu_memory = torch.cuda.max_memory_allocated() / (1024**3)
            gpu_memory_used = peak_gpu_memory - initial_gpu_memory
        else:
            peak_gpu_memory = 0
            gpu_memory_used = 0
        
        # CPU memory
        cpu_memory_peak = 0
        if HAS_MEMORY_PROFILING:
            current, peak = tracemalloc.get_traced_memory()
            cpu_memory_peak = peak / (1024**3)  # Convert to GB
            tracemalloc.stop()
        
        # Calculate performance metrics
        if not times:
            print(f"  No successful runs for {method_name}")
            return None
        
        results = {
            'method_name': method_name,
            'successful_runs': successful_runs,
            'total_runs': num_samples,
            'success_rate': successful_runs / num_samples,
            
            # Timing metrics
            'mean_time_per_explanation': np.mean(times),
            'std_time_per_explanation': np.std(times),
            'min_time_per_explanation': np.min(times),
            'max_time_per_explanation': np.max(times),
            'median_time_per_explanation': np.median(times),
            'total_time': total_time,
            'throughput_explanations_per_second': successful_runs / total_time,
            
            # Memory metrics
            'gpu_memory_used_gb': gpu_memory_used,
            'peak_gpu_memory_gb': peak_gpu_memory,
            'cpu_memory_peak_gb': cpu_memory_peak,
            
            # Hardware monitoring results
            'hardware_metrics': hardware_metrics,
            
            # Timing distribution
            'timing_percentiles': {
                '95th': np.percentile(times, 95),
                '99th': np.percentile(times, 99),
                '99.9th': np.percentile(times, 99.9)
            }
        }
        
        print(f"  Completed: {successful_runs}/{num_samples} successful runs")
        print(f"  Mean time per explanation: {results['mean_time_per_explanation']:.4f}s")
        print(f"  Throughput: {results['throughput_explanations_per_second']:.2f} explanations/sec")
        print(f"  GPU memory used: {results['gpu_memory_used_gb']:.3f} GB")
        
        return results
    
    def run_full_benchmark(self, methods=None, num_samples=50, warmup_runs=5):
        """Run benchmark on all or specified methods."""
        if methods is None:
            methods = list(self.xai_methods.keys())
        
        print("Starting comprehensive XAI hardware performance benchmark")
        print(f"Device: {self.device}")
        print(f"Number of samples per method: {num_samples}")
        print(f"Warmup runs: {warmup_runs}")
        print(f"Methods to benchmark: {methods}")
        
        results = {}
        
        for method_name in methods:
            try:
                result = self.benchmark_method(method_name, num_samples, warmup_runs)
                if result:
                    results[method_name] = result
                    
                # Clean up GPU memory between methods
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"Failed to benchmark {method_name}: {e}")
                continue
        
        self.results = results
        return results
    
    def save_results(self, filepath):
        """Save benchmark results to JSON file."""
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            return obj
        
        serializable_results = convert_numpy_types(self.results)
        
        with open(filepath, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'device': str(self.device),
                'model_name': self.model.__class__.__name__,
                'results': serializable_results
            }, f, indent=2)
        
        print(f"Results saved to {filepath}")
    
    def create_performance_report(self, save_path=None):
        """Create comprehensive performance report with visualizations."""
        if not self.results:
            print("No results to report")
            return
        
        # Create DataFrame for analysis
        data = []
        for method_name, result in self.results.items():
            data.append({
                'Method': method_name,
                'Mean Time (s)': result['mean_time_per_explanation'],
                'Throughput (exp/s)': result['throughput_explanations_per_second'],
                'GPU Memory (GB)': result['gpu_memory_used_gb'],
                'Success Rate': result['success_rate'],
                'GPU Power (W)': result['hardware_metrics']['statistics'].get('gpu_power_watts_mean', 0)
            })
        
        df = pd.DataFrame(data)
        
        # Create visualizations
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('XAI Methods Hardware Performance Benchmark', fontsize=16)
        
        # 1. Execution Time
        df_sorted = df.sort_values('Mean Time (s)')
        axes[0, 0].barh(df_sorted['Method'], df_sorted['Mean Time (s)'])
        axes[0, 0].set_xlabel('Mean Execution Time (seconds)')
        axes[0, 0].set_title('Execution Time per Method')
        
        # 2. Throughput
        df_sorted = df.sort_values('Throughput (exp/s)', ascending=False)
        axes[0, 1].barh(df_sorted['Method'], df_sorted['Throughput (exp/s)'])
        axes[0, 1].set_xlabel('Throughput (explanations/second)')
        axes[0, 1].set_title('Throughput per Method')
        
        # 3. GPU Memory Usage
        df_sorted = df.sort_values('GPU Memory (GB)')
        axes[0, 2].barh(df_sorted['Method'], df_sorted['GPU Memory (GB)'])
        axes[0, 2].set_xlabel('GPU Memory Usage (GB)')
        axes[0, 2].set_title('GPU Memory Usage per Method')
        
        # 4. Power Consumption
        if df['GPU Power (W)'].sum() > 0:
            df_sorted = df.sort_values('GPU Power (W)')
            axes[1, 0].barh(df_sorted['Method'], df_sorted['GPU Power (W)'])
            axes[1, 0].set_xlabel('Average GPU Power (Watts)')
            axes[1, 0].set_title('Power Consumption per Method')
        else:
            axes[1, 0].text(0.5, 0.5, 'Power data not available', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Power Consumption (N/A)')
        
        # 5. Success Rate
        axes[1, 1].barh(df['Method'], df['Success Rate'])
        axes[1, 1].set_xlabel('Success Rate')
        axes[1, 1].set_title('Success Rate per Method')
        axes[1, 1].set_xlim(0, 1)
        
        # 6. Performance vs Memory Tradeoff
        scatter = axes[1, 2].scatter(df['GPU Memory (GB)'], df['Throughput (exp/s)'], 
                                   c=df['Mean Time (s)'], cmap='viridis', s=100)
        axes[1, 2].set_xlabel('GPU Memory Usage (GB)')
        axes[1, 2].set_ylabel('Throughput (explanations/second)')
        axes[1, 2].set_title('Performance vs Memory Tradeoff')
        plt.colorbar(scatter, ax=axes[1, 2], label='Mean Time (s)')
        
        # Add method labels to scatter plot
        for i, method in enumerate(df['Method']):
            axes[1, 2].annotate(method, (df.iloc[i]['GPU Memory (GB)'], df.iloc[i]['Throughput (exp/s)']),
                               xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Performance report saved to {save_path}")
        
        plt.show()
        
        # Print summary table
        print("\n" + "="*80)
        print("XAI METHODS HARDWARE PERFORMANCE SUMMARY")
        print("="*80)
        print(df.round(4).to_string(index=False))
        
        # Efficiency rankings
        print("\n" + "="*80)
        print("EFFICIENCY RANKINGS")
        print("="*80)
        
        print("\nFastest Methods (by execution time):")
        fastest = df.nsmallest(5, 'Mean Time (s)')
        for i, (_, row) in enumerate(fastest.iterrows(), 1):
            print(f"{i}. {row['Method']}: {row['Mean Time (s)']:.4f}s")
        
        print("\nHighest Throughput Methods:")
        highest_throughput = df.nlargest(5, 'Throughput (exp/s)')
        for i, (_, row) in enumerate(highest_throughput.iterrows(), 1):
            print(f"{i}. {row['Method']}: {row['Throughput (exp/s)']:.2f} exp/s")
        
        print("\nMost Memory Efficient Methods:")
        most_efficient = df.nsmallest(5, 'GPU Memory (GB)')
        for i, (_, row) in enumerate(most_efficient.iterrows(), 1):
            print(f"{i}. {row['Method']}: {row['GPU Memory (GB)']:.3f} GB")


def load_model_and_data(model_name, checkpoint_path, dataset_name, device, batch_size=1):
    """Load model and dataset for benchmarking."""
    # Load model
    model_module = importlib.import_module(f'models.{model_name}')
    if hasattr(model_module, 'create_model'):
        model = model_module.create_model()
    elif hasattr(model_module, 'Net'):
        model = model_module.Net()
    else:
        raise ValueError(f"Could not find model class in models.{model_name}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    # Load dataset
    dataset_module = importlib.import_module(f'datasets.{dataset_name}')
    _, test_loader = dataset_module.get_dataloaders(batch_size=batch_size)
    
    return model, test_loader


def main():
    parser = argparse.ArgumentParser(description='XAI Hardware Performance Benchmark')
    
    parser.add_argument('--model', type=str, required=True,
                       help='Model name (Python module in models/ folder)')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset name (Python module in datasets/ folder)')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--methods', nargs='+', 
                       help='XAI methods to benchmark (default: all)')
    parser.add_argument('--num-samples', type=int, default=50,
                       help='Number of samples to benchmark per method')
    parser.add_argument('--warmup-runs', type=int, default=5,
                       help='Number of warmup runs before benchmarking')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--output-dir', type=str, default='results/hardware_benchmark',
                       help='Output directory for results')
    parser.add_argument('--gpu-index', type=int, default=0,
                       help='GPU index for monitoring')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model and data
    print("Loading model and dataset...")
    model, test_loader = load_model_and_data(
        args.model, args.checkpoint, args.dataset, device
    )
    
    # Create benchmark instance
    benchmark = XAIPerformanceBenchmark(model, device, test_loader)
    
    # Run benchmark
    benchmark.run_full_benchmark(
        methods=args.methods,
        num_samples=args.num_samples,
        warmup_runs=args.warmup_runs
    )
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"hardware_benchmark_{timestamp}.json"
    benchmark.save_results(results_file)
    
    # Create performance report
    report_file = output_dir / f"performance_report_{timestamp}.png"
    benchmark.create_performance_report(save_path=report_file)
    
    print("\nBenchmark completed successfully!")
    print(f"Results saved to: {results_file}")
    print(f"Report saved to: {report_file}")


if __name__ == "__main__":
    main()