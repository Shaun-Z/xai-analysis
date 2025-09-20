#!/usr/bin/env python3
"""
Comprehensive test showing more models and XAI methods with real datasets.
"""

import sys
import os
sys.path.append('src')

from src.main_analysis import XAIBenchmark

def run_comprehensive_test():
    """Run comprehensive XAI analysis with more models."""
    print("Running Comprehensive XAI Analysis")
    print("=" * 60)
    
    # Test 1: CIFAR-10 dataset with CNN models
    print("\n1. Testing with CIFAR-10 dataset:")
    benchmark_cifar = XAIBenchmark(output_dir="comprehensive_cifar10_results", dataset="cifar10")
    
    cifar_results = benchmark_cifar.run_full_benchmark(
        model_names=['simple_cnn'],  # CNN appropriate for CIFAR-10
        num_samples=3,
        quick_mode=False  # Use full XAI analysis
    )
    
    # Test 2: ImageNet-like dataset with larger models  
    print("\n2. Testing with ImageNet-like dataset:")
    benchmark_imagenet = XAIBenchmark(output_dir="comprehensive_imagenet_results", dataset="imagenet")
    
    imagenet_results = benchmark_imagenet.run_full_benchmark(
        model_names=['mobilenet_v2', 'efficientnet_b0'],  # Models suitable for ImageNet
        num_samples=3,
        quick_mode=False  # Use full XAI analysis
    )
    
    # Test 3: Synthetic data for comparison
    print("\n3. Testing with synthetic dataset for comparison:")
    benchmark_synthetic = XAIBenchmark(output_dir="comprehensive_synthetic_results", dataset="synthetic")
    
    synthetic_results = benchmark_synthetic.run_full_benchmark(
        model_names=['simple_cnn', 'mobilenet_v2'],
        num_samples=3,
        quick_mode=False  # Use full XAI analysis
    )
    
    print("\n" + "=" * 60)
    print("Comprehensive test completed!")
    print("Check the following directories for detailed outputs:")
    print("- comprehensive_cifar10_results/")
    print("- comprehensive_imagenet_results/")
    print("- comprehensive_synthetic_results/")
    
    return {
        'cifar10': cifar_results,
        'imagenet': imagenet_results, 
        'synthetic': synthetic_results
    }

if __name__ == "__main__":
    run_comprehensive_test()