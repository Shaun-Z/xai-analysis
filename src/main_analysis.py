"""
Main XAI analysis script that runs various Captum methods on different models
while monitoring hardware performance.
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple

import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.model_definitions import ModelFactory, count_parameters, get_model_size_mb
from xai.captum_methods import CaptumXAI, XAIResult, visualize_attributions
from monitoring.hardware_monitor import HardwareMonitor, BenchmarkTimer
from utils.data_utils import get_sample_images, visualize_batch, get_model_input_size


class XAIBenchmark:
    """Comprehensive XAI benchmarking with hardware monitoring."""
    
    def __init__(self, output_dir: str = "results", device: str = None):
        """
        Initialize XAI benchmark.
        
        Args:
            output_dir: Directory to save results
            device: Device to run on ('cpu', 'cuda', or None for auto)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Setup device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        print(f"Using device: {self.device}")
        
        # Initialize hardware monitor
        self.hardware_monitor = HardwareMonitor(sampling_interval=0.1)
        
        # Results storage
        self.results = []
        self.hardware_results = []
    
    def benchmark_model(self, 
                       model_name: str, 
                       num_samples: int = 5,
                       quick_mode: bool = False) -> Dict[str, Any]:
        """
        Benchmark XAI methods on a specific model.
        
        Args:
            model_name: Name of the model to benchmark
            num_samples: Number of sample images to analyze
            quick_mode: Whether to use quick mode for faster analysis
            
        Returns:
            Dictionary containing benchmark results
        """
        print(f"\n{'='*60}")
        print(f"Benchmarking {model_name}")
        print(f"{'='*60}")
        
        try:
            # Create model
            model = ModelFactory.create_model(model_name, num_classes=1000, pretrained=True)
            model = model.to(self.device)
            model.eval()
            
            # Get model info
            model_info = ModelFactory.get_model_info(model_name)
            num_params = count_parameters(model)
            model_size_mb = get_model_size_mb(model)
            
            print(f"Model: {model_info['description']}")
            print(f"Parameters: {num_params:,}")
            print(f"Size: {model_size_mb:.2f} MB")
            
            # Generate sample data
            input_size = get_model_input_size(model_name)
            sample_images = get_sample_images(
                num_samples=num_samples, 
                image_size=input_size, 
                device=self.device
            )
            
            # Initialize XAI analyzer
            xai_analyzer = CaptumXAI(model, self.device)
            
            # Results for this model
            model_results = {
                'model_name': model_name,
                'model_info': model_info,
                'num_parameters': num_params,
                'model_size_mb': model_size_mb,
                'device': self.device,
                'num_samples': num_samples,
                'xai_results': [],
                'hardware_stats': [],
                'total_time': 0
            }
            
            # Benchmark each sample
            total_start_time = time.time()
            
            for sample_idx in tqdm(range(num_samples), desc="Processing samples"):
                sample_image = sample_images[sample_idx:sample_idx+1]
                
                # Get prediction info
                output, predicted_class, confidence = xai_analyzer.get_prediction(sample_image)
                
                print(f"\nSample {sample_idx + 1}: Predicted class {predicted_class} "
                      f"(confidence: {confidence:.3f})")
                
                # Benchmark XAI methods
                with BenchmarkTimer(self.hardware_monitor, f"{model_name}_sample_{sample_idx}") as timer:
                    xai_results = xai_analyzer.explain_all_methods(
                        sample_image, 
                        target=predicted_class,
                        quick_mode=quick_mode
                    )
                
                # Collect hardware stats
                hardware_stats = self.hardware_monitor.get_summary_stats()
                hardware_stats['sample_idx'] = sample_idx
                hardware_stats['model_name'] = model_name
                hardware_stats['operation_duration'] = timer.duration
                
                # Save visualizations
                for xai_result in xai_results:
                    viz_path = (self.output_dir / 
                               f"{model_name}_sample_{sample_idx}_{xai_result.method_name.replace(' ', '_')}.png")
                    
                    try:
                        fig = visualize_attributions(xai_result, sample_image, str(viz_path))
                        plt.close(fig)
                    except Exception as e:
                        print(f"Warning: Could not save visualization for {xai_result.method_name}: {e}")
                
                # Store results
                sample_result = {
                    'sample_idx': sample_idx,
                    'predicted_class': predicted_class,
                    'confidence': confidence,
                    'xai_methods': []
                }
                
                for xai_result in xai_results:
                    sample_result['xai_methods'].append({
                        'method_name': xai_result.method_name,
                        'processing_time': xai_result.processing_time,
                        'target_class': xai_result.target_class,
                        'confidence': xai_result.confidence
                    })
                
                model_results['xai_results'].append(sample_result)
                model_results['hardware_stats'].append(hardware_stats)
                self.hardware_results.append(hardware_stats)
            
            model_results['total_time'] = time.time() - total_start_time
            
            # Calculate aggregated statistics
            all_processing_times = []
            method_times = {}
            
            for sample_result in model_results['xai_results']:
                for method_result in sample_result['xai_methods']:
                    method_name = method_result['method_name']
                    processing_time = method_result['processing_time']
                    
                    all_processing_times.append(processing_time)
                    
                    if method_name not in method_times:
                        method_times[method_name] = []
                    method_times[method_name].append(processing_time)
            
            # Add summary statistics
            model_results['summary_stats'] = {
                'avg_processing_time_per_method': sum(all_processing_times) / len(all_processing_times) if all_processing_times else 0,
                'total_processing_time': sum(all_processing_times),
                'method_avg_times': {method: sum(times) / len(times) for method, times in method_times.items()},
                'throughput_samples_per_second': num_samples / model_results['total_time'] if model_results['total_time'] > 0 else 0
            }
            
            print(f"\nCompleted {model_name} benchmark:")
            print(f"  Total time: {model_results['total_time']:.2f}s")
            print(f"  Average processing time per method: {model_results['summary_stats']['avg_processing_time_per_method']:.3f}s")
            print(f"  Throughput: {model_results['summary_stats']['throughput_samples_per_second']:.2f} samples/sec")
            
            return model_results
            
        except Exception as e:
            print(f"Error benchmarking {model_name}: {e}")
            import traceback
            traceback.print_exc()
            return {
                'model_name': model_name,
                'error': str(e),
                'total_time': 0
            }
    
    def run_full_benchmark(self, 
                          model_names: List[str] = None,
                          num_samples: int = 3,
                          quick_mode: bool = True) -> Dict[str, Any]:
        """
        Run full benchmark across multiple models.
        
        Args:
            model_names: List of model names to benchmark
            num_samples: Number of samples per model
            quick_mode: Whether to use quick mode
            
        Returns:
            Complete benchmark results
        """
        if model_names is None:
            model_names = ['simple_cnn', 'resnet18', 'mobilenet_v2']
        
        print(f"Starting XAI benchmark with {len(model_names)} models")
        print(f"Samples per model: {num_samples}")
        print(f"Quick mode: {quick_mode}")
        print(f"Device: {self.device}")
        
        benchmark_start_time = time.time()
        
        for model_name in model_names:
            model_result = self.benchmark_model(model_name, num_samples, quick_mode)
            self.results.append(model_result)
        
        total_benchmark_time = time.time() - benchmark_start_time
        
        # Compile final results
        final_results = {
            'benchmark_info': {
                'total_time': total_benchmark_time,
                'device': self.device,
                'num_models': len(model_names),
                'num_samples_per_model': num_samples,
                'quick_mode': quick_mode,
                'timestamp': time.time()
            },
            'model_results': self.results,
            'hardware_results': self.hardware_results
        }
        
        # Save results
        self.save_results(final_results)
        
        # Generate report
        self.generate_report(final_results)
        
        return final_results
    
    def save_results(self, results: Dict[str, Any]):
        """Save benchmark results to JSON file."""
        results_path = self.output_dir / "benchmark_results.json"
        
        # Convert any non-serializable objects
        serializable_results = self._make_serializable(results)
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Results saved to {results_path}")
    
    def _make_serializable(self, obj):
        """Convert objects to JSON-serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            return str(obj)
    
    def generate_report(self, results: Dict[str, Any]):
        """Generate a comprehensive analysis report."""
        print(f"\n{'='*80}")
        print("XAI BENCHMARK REPORT")
        print(f"{'='*80}")
        
        # Create performance comparison plots
        self._create_performance_plots(results)
        
        # Create hardware monitoring plots
        self._create_hardware_plots()
        
        # Print summary statistics
        self._print_summary_stats(results)
    
    def _create_performance_plots(self, results: Dict[str, Any]):
        """Create performance comparison plots."""
        # Extract data for plotting
        model_names = []
        avg_times = []
        throughputs = []
        model_sizes = []
        
        for model_result in results['model_results']:
            if 'error' not in model_result:
                model_names.append(model_result['model_name'])
                avg_times.append(model_result['summary_stats']['avg_processing_time_per_method'])
                throughputs.append(model_result['summary_stats']['throughput_samples_per_second'])
                model_sizes.append(model_result['model_size_mb'])
        
        if not model_names:
            return
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Average processing time per method
        axes[0, 0].bar(model_names, avg_times)
        axes[0, 0].set_title('Average Processing Time per XAI Method')
        axes[0, 0].set_ylabel('Time (seconds)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Throughput
        axes[0, 1].bar(model_names, throughputs, color='orange')
        axes[0, 1].set_title('Throughput (Samples per Second)')
        axes[0, 1].set_ylabel('Samples/sec')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Model size vs processing time
        axes[1, 0].scatter(model_sizes, avg_times, s=100, alpha=0.7)
        for i, model in enumerate(model_names):
            axes[1, 0].annotate(model, (model_sizes[i], avg_times[i]), xytext=(5, 5), 
                               textcoords='offset points', fontsize=8)
        axes[1, 0].set_xlabel('Model Size (MB)')
        axes[1, 0].set_ylabel('Avg Processing Time (s)')
        axes[1, 0].set_title('Model Size vs Processing Time')
        
        # Method comparison across models
        method_data = {}
        for model_result in results['model_results']:
            if 'error' not in model_result:
                model_name = model_result['model_name']
                for method, time_val in model_result['summary_stats']['method_avg_times'].items():
                    if method not in method_data:
                        method_data[method] = {}
                    method_data[method][model_name] = time_val
        
        if method_data:
            df_methods = pd.DataFrame(method_data).T
            df_methods.plot(kind='bar', ax=axes[1, 1])
            axes[1, 1].set_title('XAI Method Performance by Model')
            axes[1, 1].set_ylabel('Processing Time (s)')
            axes[1, 1].tick_params(axis='x', rotation=45)
            axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'performance_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Performance plots saved to {self.output_dir / 'performance_comparison.png'}")
    
    def _create_hardware_plots(self):
        """Create hardware monitoring plots."""
        if not self.hardware_results:
            return
        
        df_hardware = pd.DataFrame(self.hardware_results)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # CPU usage by model
        if 'avg_cpu_percent' in df_hardware.columns:
            df_hardware.boxplot(column='avg_cpu_percent', by='model_name', ax=axes[0, 0])
            axes[0, 0].set_title('CPU Usage by Model')
            axes[0, 0].set_ylabel('CPU %')
        
        # Memory usage by model
        if 'avg_memory_percent' in df_hardware.columns:
            df_hardware.boxplot(column='avg_memory_percent', by='model_name', ax=axes[0, 1])
            axes[0, 1].set_title('Memory Usage by Model')
            axes[0, 1].set_ylabel('Memory %')
        
        # GPU usage if available
        if 'avg_gpu_utilization' in df_hardware.columns:
            df_hardware.boxplot(column='avg_gpu_utilization', by='model_name', ax=axes[1, 0])
            axes[1, 0].set_title('GPU Utilization by Model')
            axes[1, 0].set_ylabel('GPU %')
        
        # Operation duration
        if 'operation_duration' in df_hardware.columns:
            df_hardware.boxplot(column='operation_duration', by='model_name', ax=axes[1, 1])
            axes[1, 1].set_title('Operation Duration by Model')
            axes[1, 1].set_ylabel('Duration (s)')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'hardware_monitoring.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Hardware plots saved to {self.output_dir / 'hardware_monitoring.png'}")
    
    def _print_summary_stats(self, results: Dict[str, Any]):
        """Print summary statistics."""
        print(f"\nBenchmark Summary:")
        print(f"  Total time: {results['benchmark_info']['total_time']:.2f}s")
        print(f"  Models tested: {results['benchmark_info']['num_models']}")
        print(f"  Samples per model: {results['benchmark_info']['num_samples_per_model']}")
        print(f"  Device: {results['benchmark_info']['device']}")
        
        print(f"\nModel Performance Rankings:")
        model_perf = []
        for model_result in results['model_results']:
            if 'error' not in model_result:
                model_perf.append({
                    'model': model_result['model_name'],
                    'avg_time': model_result['summary_stats']['avg_processing_time_per_method'],
                    'throughput': model_result['summary_stats']['throughput_samples_per_second'],
                    'size_mb': model_result['model_size_mb']
                })
        
        # Sort by throughput (higher is better)
        model_perf.sort(key=lambda x: x['throughput'], reverse=True)
        
        for i, model in enumerate(model_perf, 1):
            print(f"  {i}. {model['model']} - "
                  f"Throughput: {model['throughput']:.2f} samples/s, "
                  f"Avg time: {model['avg_time']:.3f}s, "
                  f"Size: {model['size_mb']:.1f}MB")


def main():
    """Main function to run XAI benchmark."""
    parser = argparse.ArgumentParser(description='XAI Analysis with Hardware Monitoring')
    parser.add_argument('--models', nargs='+', 
                       choices=ModelFactory.get_available_models(),
                       default=['simple_cnn', 'resnet18', 'mobilenet_v2'],
                       help='Models to benchmark')
    parser.add_argument('--samples', type=int, default=3,
                       help='Number of samples per model')
    parser.add_argument('--quick', action='store_true',
                       help='Use quick mode for faster analysis')
    parser.add_argument('--output', type=str, default='results',
                       help='Output directory')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default=None,
                       help='Device to use (auto-detect if not specified)')
    
    args = parser.parse_args()
    
    # Create benchmark instance
    benchmark = XAIBenchmark(output_dir=args.output, device=args.device)
    
    # Run benchmark
    results = benchmark.run_full_benchmark(
        model_names=args.models,
        num_samples=args.samples,
        quick_mode=args.quick
    )
    
    print(f"\nBenchmark completed! Results saved to {args.output}/")


if __name__ == "__main__":
    main()