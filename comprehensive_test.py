#!/usr/bin/env python3
"""
Comprehensive test showing more models and XAI methods.
"""

import sys
import os
sys.path.append('src')

from src.main_analysis import XAIBenchmark

def run_comprehensive_test():
    """Run comprehensive XAI analysis with more models."""
    print("Running Comprehensive XAI Analysis")
    print("=" * 60)
    
    # Create benchmark
    benchmark = XAIBenchmark(output_dir="comprehensive_results")
    
    # Run benchmark on more models
    results = benchmark.run_full_benchmark(
        model_names=['simple_cnn', 'resnet18', 'mobilenet_v2', 'efficientnet_b0'],
        num_samples=3,
        quick_mode=False  # Use full XAI analysis
    )
    
    print("\nComprehensive test completed!")
    print("Check the 'comprehensive_results' directory for detailed outputs.")
    
    return results

if __name__ == "__main__":
    run_comprehensive_test()