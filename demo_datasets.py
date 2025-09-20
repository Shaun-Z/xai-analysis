#!/usr/bin/env python3
"""
Demo script showing the difference between synthetic and real dataset usage.
"""

import sys
import os
sys.path.append('src')

from src.main_analysis import XAIBenchmark

def demo_dataset_comparison():
    """Demonstrate the differences between datasets."""
    print("XAI Analysis Dataset Comparison Demo")
    print("=" * 60)
    
    # 1. Synthetic Dataset (Original approach)
    print("\n1. SYNTHETIC DATASET:")
    print("-" * 30)
    synthetic_benchmark = XAIBenchmark(output_dir="demo_synthetic", dataset="synthetic")
    synthetic_results = synthetic_benchmark.run_full_benchmark(
        model_names=['simple_cnn'],
        num_samples=2,
        quick_mode=True
    )
    
    # 2. CIFAR-10 Dataset (New approach)
    print("\n2. CIFAR-10 DATASET:")
    print("-" * 30)
    cifar_benchmark = XAIBenchmark(output_dir="demo_cifar10", dataset="cifar10")
    cifar_results = cifar_benchmark.run_full_benchmark(
        model_names=['simple_cnn'],
        num_samples=2,
        quick_mode=True
    )
    
    # 3. ImageNet-like Dataset (New approach)
    print("\n3. IMAGENET-LIKE DATASET:")
    print("-" * 30)
    imagenet_benchmark = XAIBenchmark(output_dir="demo_imagenet", dataset="imagenet")
    imagenet_results = imagenet_benchmark.run_full_benchmark(
        model_names=['mobilenet_v2'],
        num_samples=2,
        quick_mode=True
    )
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETED!")
    print("=" * 60)
    print("\nKey differences between datasets:")
    print("- Synthetic: Random noise patterns, no ground truth labels")
    print("- CIFAR-10: Class-specific patterns (airplanes, cars, etc.) with true labels")
    print("- ImageNet: High-resolution structured patterns with 1000 classes")
    print("\nCheck these directories for detailed results:")
    print("- demo_synthetic/")
    print("- demo_cifar10/")
    print("- demo_imagenet/")
    
    return {
        'synthetic': synthetic_results,
        'cifar10': cifar_results,
        'imagenet': imagenet_results
    }

if __name__ == "__main__":
    demo_dataset_comparison()