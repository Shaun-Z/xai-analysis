#!/usr/bin/env python3
"""
Quick validation script to test all XAI methods for compatibility
"""

import sys
from pathlib import Path
import torch
from benchmark_hardware_performance import load_model_and_data, XAIPerformanceBenchmark

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def test_all_methods():
    """Test all XAI methods for basic functionality."""
    print("Testing XAI method compatibility...")
    
    # Load model and data
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, test_loader = load_model_and_data(
        'cifar_cnn', 
        'checkpoints/cifar_cnn_cifar10/best_model.pt',
        'cifar10',
        device
    )
    
    # Create benchmark instance
    benchmark = XAIPerformanceBenchmark(model, device, test_loader)
    
    # Test each method with minimal samples
    methods_to_test = list(benchmark.xai_methods.keys())
    
    print(f"Testing {len(methods_to_test)} XAI methods: {methods_to_test}")
    
    working_methods = []
    failed_methods = []
    
    for method_name in methods_to_test:
        print(f"\nTesting {method_name}...", end=" ")
        try:
            result = benchmark.benchmark_method(method_name, num_samples=2, warmup_runs=1)
            if result and result['successful_runs'] > 0:
                print("✓ PASS")
                working_methods.append(method_name)
            else:
                print("✗ FAIL (no successful runs)")
                failed_methods.append(method_name)
        except Exception as e:
            print(f"✗ FAIL ({str(e)[:50]}...)")
            failed_methods.append(method_name)
    
    print("\n" + "="*60)
    print("COMPATIBILITY TEST RESULTS")
    print("="*60)
    print(f"Working methods ({len(working_methods)}): {working_methods}")
    if failed_methods:
        print(f"Failed methods ({len(failed_methods)}): {failed_methods}")
    
    print(f"\nCompatibility: {len(working_methods)}/{len(methods_to_test)} methods working")
    
    return working_methods, failed_methods

if __name__ == "__main__":
    test_all_methods()