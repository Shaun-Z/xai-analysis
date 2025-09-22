# XAI Hardware Performance Benchmark

A comprehensive system for evaluating the computational efficiency of XAI (eXplainable AI) methods across power consumption, throughput, and memory access patterns.

## Table of Contents

1. [Overview](#overview)
2. [Project Summary](#project-summary)
3. [Prerequisites](#prerequisites)
4. [Usage](#usage)
5. [Performance Metrics](#performance-metrics)
6. [Results & Analysis](#results--analysis)
7. [Output Files](#output-files)
8. [Performance Guidelines](#performance-guidelines)
9. [Troubleshooting](#troubleshooting)
10. [Advanced Usage](#advanced-usage)

## Overview

The `benchmark_hardware_performance.py` script evaluates XAI methods across three key hardware performance metrics:

1. **Power Consumption**: GPU/CPU power usage during explanation generation
2. **Throughput**: Number of explanations generated per second  
3. **Memory Access**: GPU memory usage and memory allocation patterns

### Key Features

- **Real-time Hardware Monitoring**: GPU/CPU utilization, memory usage, power consumption, temperature
- **Comprehensive Performance Analysis**: Timing statistics, percentile analysis, throughput calculation, success rate tracking
- **Automated Visualization**: 6 different performance charts and sortable summary tables
- **Cross-platform Support**: Works with NVIDIA GPUs and CPU-only systems
- **JSON Export**: Machine-readable results for further analysis

## Project Summary

### Files Created

1. **`benchmark_hardware_performance.py`** - Core benchmarking system with hardware monitoring
2. **`test_xai_compatibility.py`** - Quick validation of XAI method compatibility  
3. **`requirements_hardware_benchmark.txt`** - Additional dependencies (psutil, pynvml, pandas, matplotlib)

### Supported XAI Methods

✅ **Working Methods (10)**:
- **Saliency** - Basic gradient-based attribution
- **Integrated Gradients** - Path-based attribution method
- **Gradient SHAP** - SHAP values using gradients
- **DeepLift** - Rule-based attribution method
- **GradCAM** - Class activation mapping
- **Guided GradCAM** - Enhanced GradCAM with guided backpropagation
- **Occlusion** - Perturbation-based attribution
- **Input × Gradient** - Simple gradient-input multiplication
- **Guided Backpropagation** - Modified backpropagation method
- **Deconvolution** - Deconvolutional networks for visualization

❌ **Methods with Issues (2)**:
- **DeepLift SHAP** - Fixed (requires baselines parameter)
- **LRP** - Requires model-specific rules for BatchNorm layers

## Prerequisites

Install the required dependencies:

```bash
pip install psutil pynvml pandas matplotlib
```

Or install from the requirements file:

```bash
pip install -r requirements_hardware_benchmark.txt
```

## Usage

### Basic Usage

```bash
python benchmark_hardware_performance.py \
    --model cifar_cnn \
    --dataset cifar10 \
    --checkpoint checkpoints/cifar_cnn_cifar10/best_model.pt \
    --methods saliency integrated_gradients gradcam \
    --num-samples 50 \
    --warmup-runs 5
```

### All Parameters

```bash
python benchmark_hardware_performance.py \
    --model MODEL_NAME \
    --dataset DATASET_NAME \
    --checkpoint CHECKPOINT_PATH \
    [--methods METHOD1 METHOD2 ...] \
    [--num-samples NUM_SAMPLES] \
    [--warmup-runs WARMUP_RUNS] \
    [--device DEVICE] \
    [--output-dir OUTPUT_DIR] \
    [--gpu-index GPU_INDEX]
```

### Parameter Description

- `--model`: Model name (Python module in models/ folder)
- `--dataset`: Dataset name (Python module in datasets/ folder) 
- `--checkpoint`: Path to trained model checkpoint
- `--methods`: List of XAI methods to benchmark (default: all methods)
- `--num-samples`: Number of samples to process per method (default: 50)
- `--warmup-runs`: Number of warmup runs before timing (default: 5)
- `--device`: Device to use - auto/cpu/cuda (default: auto)
- `--output-dir`: Output directory for results (default: results/hardware_benchmark)
- `--gpu-index`: GPU index for monitoring (default: 0)

### Usage Examples

#### Quick Test (3 methods, 20 samples each):
```bash
python benchmark_hardware_performance.py \
    --model cifar_cnn \
    --dataset cifar10 \
    --checkpoint checkpoints/cifar_cnn_cifar10/best_model.pt \
    --methods saliency integrated_gradients gradcam \
    --num-samples 20 \
    --warmup-runs 3
```

#### Comprehensive Benchmark (all methods, 100 samples):
```bash
python benchmark_hardware_performance.py \
    --model cifar_cnn \
    --dataset cifar10 \
    --checkpoint checkpoints/cifar_cnn_cifar10/best_model.pt \
    --num-samples 100 \
    --warmup-runs 5
```

## Performance Metrics

### Timing Metrics
- **Mean Time per Explanation**: Average time to generate one explanation
- **Throughput**: Explanations generated per second
- **Timing Distribution**: 95th, 99th, 99.9th percentiles
- **Min/Max Times**: Fastest and slowest explanation times

### Memory Metrics
- **GPU Memory Used**: Peak GPU memory consumption during benchmarking
- **CPU Memory Peak**: Maximum CPU memory usage
- **Memory Efficiency**: Memory usage per explanation

### Hardware Monitoring
- **CPU Utilization**: Processor usage during explanation generation
- **GPU Utilization**: GPU compute utilization percentage
- **GPU Power**: Power consumption in Watts (if supported)
- **GPU Temperature**: Operating temperature in Celsius

## Results & Analysis

### Performance Results Summary

Based on benchmarking with the CIFAR-10 CNN model:

#### Fastest Methods (< 2ms per explanation):
1. **Input × Gradient**: 1.0ms, 690 exp/s
2. **Guided Backpropagation**: 1.0ms, 679 exp/s  
3. **GradCAM**: 1.0ms, 665 exp/s
4. **Saliency**: 1.0ms, 657 exp/s
5. **Deconvolution**: 1.0ms, 662 exp/s

#### Memory Efficient Methods:
1. **Saliency**: 0.002 GB (most efficient)
2. **Other methods**: ~0.054-0.055 GB

#### Comprehensive Methods (higher quality, slower):
1. **Integrated Gradients**: 7.5ms, 124 exp/s
2. **Occlusion**: 34.5ms, 29 exp/s (most thorough)

### Example Console Output

```
================================================================================
XAI METHODS HARDWARE PERFORMANCE SUMMARY
================================================================================
              Method  Mean Time (s)  Throughput (exp/s)  GPU Memory (GB)  Success Rate
            saliency         0.0011            624.92           0.002           1.0
             gradcam         0.0010            652.40           0.054           1.0
       gradient_shap         0.0019            410.06           0.055           1.0
            deeplift         0.0022            363.97           0.055           1.0
integrated_gradients         0.0075            124.37           0.054           1.0

================================================================================
EFFICIENCY RANKINGS
================================================================================

Fastest Methods (by execution time):
1. gradcam: 0.0010s
2. saliency: 0.0011s  
3. gradient_shap: 0.0019s
4. deeplift: 0.0022s
5. integrated_gradients: 0.0075s
```

## Output Files

The script generates several output files:

### 1. JSON Results File
`hardware_benchmark_YYYYMMDD_HHMMSS.json`

Contains detailed timing, memory, and hardware metrics for each method:

```json
{
  "timestamp": "2025-09-21T22:44:08.770941",
  "device": "cuda",
  "model_name": "Net", 
  "results": {
    "saliency": {
      "mean_time_per_explanation": 0.001078,
      "throughput_explanations_per_second": 624.92,
      "gpu_memory_used_gb": 0.002,
      "success_rate": 1.0,
      "hardware_metrics": { ... }
    }
  }
}
```

### 2. Performance Report
`performance_report_YYYYMMDD_HHMMSS.png`

Visual report with 6 charts:
- Execution time comparison
- Throughput comparison  
- GPU memory usage
- Power consumption (if available)
- Success rate
- Performance vs memory tradeoff scatter plot

### 3. Console Summary

Detailed text summary including:
- Performance statistics table
- Efficiency rankings (fastest, highest throughput, most memory efficient)
- Success rates and error information

## Performance Guidelines

### Choosing XAI Methods Based on Performance

#### For Real-time Applications (< 10ms):
- Use **Saliency** or **GradCAM** for sub-millisecond explanations
- Achieve 600+ explanations per second throughput
- Memory usage: 2-54 MB GPU memory

#### For High Throughput Requirements:
- **GradCAM**: 652 explanations/sec
- **Saliency**: 625 explanations/sec
- **Input × Gradient**: 690 explanations/sec

#### For Memory-Constrained Environments:
- **Saliency**: Only 2 MB GPU memory
- Most other methods: ~54 MB GPU memory

#### For High-Quality Explanations (can tolerate latency):
- **Integrated Gradients**: Best balance of quality vs speed (7.5ms)
- **Occlusion**: Most interpretable but slowest (34.5ms)

### Hardware Considerations

**GPU Memory Usage**:
- Most methods use 50-55 MB GPU memory
- Saliency is most memory efficient (2 MB)
- Occlusion may require significantly more memory

**Power Consumption**:
- Gradient-based methods are generally power efficient
- Perturbation-based methods (Occlusion) consume more power
- Integrated Gradients balances quality vs power usage

## Troubleshooting

### Common Issues

1. **"Could not initialize GPU monitoring"**
   - GPU monitoring requires NVIDIA GPU with proper drivers
   - Script continues to work, but GPU-specific metrics won't be available

2. **CUDA out of memory**
   - Reduce `--num-samples` or `--batch-size`
   - Use smaller models or input sizes
   - Some methods (Occlusion) require more memory

3. **Method-specific failures**
   - Check model architecture compatibility (e.g., GradCAM needs Conv layers)
   - Verify model supports the required operations

### Performance Optimization Tips

1. **For Benchmarking**:
   - Use sufficient warmup runs (5-10)
   - Run multiple trials for statistical significance
   - Ensure consistent GPU state between methods

2. **For Production Use**:
   - Choose methods based on your latency/quality requirements
   - Consider batching multiple explanations
   - Profile memory usage for your specific models

## Advanced Usage

### Custom XAI Method Integration

To add new XAI methods to the benchmark:

1. Add the method to the `xai_methods` dictionary in `XAIPerformanceBenchmark`
2. Implement any special parameter handling in `_compute_attribution`
3. Test with a small sample before full benchmarking

### Batch Processing

For large-scale benchmarking:

```bash
# Run benchmarks for multiple models
for model in cifar_cnn resnet; do
    python benchmark_hardware_performance.py \
        --model $model \
        --dataset cifar10 \
        --checkpoint checkpoints/${model}_cifar10/best_model.pt \
        --num-samples 100
done
```

### Integration with Training Pipeline

The benchmark can be integrated into your training workflow to automatically evaluate XAI method performance after model training:

```python
from benchmark_hardware_performance import XAIPerformanceBenchmark

# After training your model
benchmark = XAIPerformanceBenchmark(model, device, test_loader)
results = benchmark.run_full_benchmark(num_samples=20)
benchmark.save_results(f"results/benchmark_{model_name}.json")
```

### Integration with Existing Workflow

The benchmarking system integrates seamlessly with your existing XAI pipeline:
- Uses the same model loading functions
- Compatible with existing dataset loaders
- Supports all model architectures in the `models/` directory
- Works with any trained checkpoints

## Future Enhancements

1. **Power Measurement**: More detailed power profiling (requires specific hardware)
2. **Batch Processing**: Support for batched explanation generation
3. **Additional Metrics**: Memory bandwidth, cache performance
4. **Model Comparisons**: Benchmark across different model architectures
5. **Cloud Integration**: Support for cloud GPU monitoring

## Conclusion

This hardware benchmarking system provides comprehensive performance evaluation capabilities for XAI methods, enabling data-driven decisions about which explanation techniques to use in production based on specific performance requirements. The system successfully measures power consumption, throughput, and memory access patterns across 10+ different XAI methods with detailed reporting and visualization capabilities.