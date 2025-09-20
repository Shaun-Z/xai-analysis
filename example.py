#!/usr/bin/env python3
"""
Simple example of XAI analysis with hardware monitoring.
"""

import sys
import os
sys.path.append('src')

from src.main_analysis import XAIBenchmark

def run_quick_example():
    """Run a quick example with minimal models and samples."""
    print("Running XAI Analysis Example")
    print("=" * 50)
    
    # Create benchmark with limited scope - demonstrate real dataset usage
    benchmark = XAIBenchmark(output_dir="example_results", dataset="cifar10")
    
    # Run benchmark on a small set of models
    results = benchmark.run_full_benchmark(
        model_names=['simple_cnn', 'mobilenet_v2'],  # Fast models
        num_samples=2,  # Few samples
        quick_mode=True  # Quick analysis
    )
    
    print("\nExample completed!")
    print("Check the 'example_results' directory for outputs.")
    
    return results

if __name__ == "__main__":
    run_quick_example()