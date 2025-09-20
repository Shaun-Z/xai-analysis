#!/usr/bin/env python3
"""
Test script to compare XAI analysis between synthetic data and real datasets.
"""

import sys
import os
sys.path.append('src')

from src.main_analysis import XAIBenchmark

def test_cifar10_data():
    """Test XAI analysis with CIFAR-10 data."""
    print("Testing XAI Analysis with CIFAR-10 data")
    print("=" * 50)
    
    # Create benchmark with CIFAR-10 dataset
    benchmark = XAIBenchmark(output_dir="cifar10_results", dataset="cifar10")
    
    # Run benchmark on simple_cnn (appropriate for CIFAR-10 size)
    results = benchmark.run_full_benchmark(
        model_names=['simple_cnn'],  # Use simple_cnn for CIFAR-10
        num_samples=3,  # Small number for testing
        quick_mode=True  # Quick analysis
    )
    
    print("\nCIFAR-10 test completed!")
    print("Check the 'cifar10_results' directory for outputs.")
    
    return results

def test_synthetic_data():
    """Test XAI analysis with synthetic data for comparison."""
    print("\nTesting XAI Analysis with Synthetic data")
    print("=" * 50)
    
    # Create benchmark with synthetic dataset
    benchmark = XAIBenchmark(output_dir="synthetic_results", dataset="synthetic")
    
    # Run benchmark on simple_cnn for fair comparison
    results = benchmark.run_full_benchmark(
        model_names=['simple_cnn'],  # Use simple_cnn for comparison
        num_samples=3,  # Same number for comparison
        quick_mode=True  # Quick analysis
    )
    
    print("\nSynthetic test completed!")
    print("Check the 'synthetic_results' directory for outputs.")
    
    return results

def test_imagenet_like_data():
    """Test XAI analysis with ImageNet-like data."""
    print("\nTesting XAI Analysis with ImageNet-like data")
    print("=" * 50)
    
    # Create benchmark with ImageNet dataset
    benchmark = XAIBenchmark(output_dir="imagenet_results", dataset="imagenet")
    
    # Run benchmark on mobilenet_v2 (appropriate for ImageNet)
    results = benchmark.run_full_benchmark(
        model_names=['mobilenet_v2'],  # Use mobilenet_v2 for ImageNet-like data
        num_samples=3,  # Small number for testing
        quick_mode=True  # Quick analysis
    )
    
    print("\nImageNet-like test completed!")
    print("Check the 'imagenet_results' directory for outputs.")
    
    return results

if __name__ == "__main__":
    print("Running Real Dataset Tests")
    print("=" * 70)
    
    # Test different datasets
    cifar10_results = test_cifar10_data()
    synthetic_results = test_synthetic_data()
    imagenet_results = test_imagenet_like_data()
    
    print("\n" + "=" * 70)
    print("All tests completed!")
    print("=" * 70)
    print("Results directories:")
    print("- cifar10_results/")
    print("- synthetic_results/")
    print("- imagenet_results/")