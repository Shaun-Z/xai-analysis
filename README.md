# XAI Analysis with Hardware Monitoring

This project implements comprehensive XAI (Explainable AI) analysis using the Captum library on various deep learning models (CNN, ResNet, MobileNet, etc.) while monitoring hardware performance indicators like power, throughput, CPU/GPU usage, and memory consumption.

## Features

- **Multiple Model Architectures**: CNN, ResNet (18/50/101), MobileNet (v2/v3), EfficientNet, VGG16
- **Comprehensive XAI Methods**: Integrated Gradients, SHAP, DeepLIFT, Saliency, Occlusion, LIME, and more
- **Real Dataset Support**: CIFAR-10, ImageNet-like datasets, and synthetic data for comparison
- **Hardware Performance Monitoring**: CPU, memory, GPU utilization, power consumption, and throughput
- **Automated Benchmarking**: Run comprehensive analysis across multiple models and samples
- **Visualization**: Generate attribution visualizations and performance comparison plots
- **Detailed Reporting**: JSON results with accuracy metrics, true labels, and comprehensive analysis reports

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Shaun-Z/xai-analysis.git
cd xai-analysis
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Note: For GPU monitoring, ensure you have NVIDIA drivers and nvidia-ml-py3 installed.

## Quick Start

Run a simple example with CIFAR-10 dataset:
```bash
python example.py
```

This will run XAI analysis on a subset of models using CIFAR-10-like data with hardware monitoring and save results to `example_results/`.

For synthetic data comparison:
```bash
python src/main_analysis.py --dataset synthetic --models simple_cnn --samples 3
```

## Advanced Usage

### Full Benchmark with Real Datasets

Run comprehensive analysis with CIFAR-10 dataset:
```bash
python src/main_analysis.py --dataset cifar10 --models simple_cnn --samples 5
```

Run analysis with ImageNet-like dataset:
```bash
python src/main_analysis.py --dataset imagenet --models mobilenet_v2 resnet18 --samples 5
```

### Command Line Options

```bash
python src/main_analysis.py [OPTIONS]

Options:
  --models MODEL [MODEL ...]    Models to benchmark (default: simple_cnn resnet18 mobilenet_v2)
  --samples INT                 Number of samples per model (default: 3)
  --dataset {synthetic,cifar10,imagenet}  Dataset to use (default: synthetic)
  --quick                       Use quick mode for faster analysis
  --output DIR                  Output directory (default: results)
  --device {cpu,cuda}           Device to use (auto-detect if not specified)
```

### Available Datasets

- `synthetic`: Generated synthetic images with structured patterns (default)
- `cifar10`: CIFAR-10 dataset or CIFAR-10-like synthetic data (32x32 images, 10 classes)
- `imagenet`: ImageNet-like synthetic data (224x224 images, 1000 classes)

**Note**: When real CIFAR-10 cannot be downloaded, the system automatically creates CIFAR-10-like synthetic data with realistic class-specific patterns (airplanes, automobiles, etc.).

### Available Models

- `simple_cnn`: Simple 4-layer CNN (~2M parameters)
- `resnet18`: ResNet-18 (~11M parameters)
- `resnet50`: ResNet-50 (~25M parameters)
- `resnet101`: ResNet-101 (~44M parameters)
- `mobilenet_v2`: MobileNet-V2 (~3.5M parameters)
- `mobilenet_v3_small`: MobileNet-V3 Small (~2.5M parameters)
- `mobilenet_v3_large`: MobileNet-V3 Large (~5.5M parameters)
- `efficientnet_b0`: EfficientNet-B0 (~5M parameters)
- `vgg16`: VGG-16 (~138M parameters)

### XAI Methods Implemented

1. **Integrated Gradients**: Path-based attribution method
2. **Gradient SHAP**: SHAP values using gradients
3. **DeepLIFT**: Deep Learning Important FeaTures
4. **Saliency Maps**: Simple gradient-based attribution
5. **Occlusion**: Feature removal-based explanation
6. **LIME**: Local Interpretable Model-agnostic Explanations
7. **Kernel SHAP**: Model-agnostic SHAP values

## Hardware Monitoring

The system monitors:
- **CPU Usage**: Percentage utilization over time
- **Memory Usage**: RAM consumption in GB and percentage
- **GPU Metrics**: Utilization, memory usage, power consumption, temperature (if NVIDIA GPU available)
- **Throughput**: Samples processed per second
- **Processing Time**: Time taken for each XAI method

## Output Structure

```
results/
├── benchmark_results.json          # Complete benchmark data with accuracy metrics
├── performance_comparison.png      # Performance comparison plots
├── hardware_monitoring.png         # Hardware usage plots
└── [model]_sample_[n]_[method].png # Individual attribution visualizations
```

## Results Analysis

The system generates:

1. **JSON Results**: Complete benchmark data with timing, hardware stats, method performance, and dataset-specific information:
   - True labels and class names (for real datasets)
   - Prediction accuracy and confidence scores
   - Dataset type used for each benchmark
   - Model-specific performance metrics

2. **Performance Plots**: Bar charts comparing processing times and throughput across models
3. **Hardware Plots**: Box plots showing CPU, memory, and GPU usage patterns
4. **Attribution Visualizations**: Heatmaps showing XAI method outputs for each sample

### Dataset-Specific Features

When using real datasets, the results include:
- **True Labels**: Actual class labels from the dataset
- **Class Names**: Human-readable class names (e.g., "airplane", "automobile" for CIFAR-10)
- **Accuracy Metrics**: Whether predictions match true labels
- **Dataset Information**: Which dataset was used for each benchmark

## Example Usage

### Compare Models on Different Datasets

```bash
# Test simple CNN on CIFAR-10-like data
python src/main_analysis.py --dataset cifar10 --models simple_cnn --samples 3

# Test multiple models on ImageNet-like data  
python src/main_analysis.py --dataset imagenet --models mobilenet_v2 resnet18 --samples 3

# Compare with synthetic data
python src/main_analysis.py --dataset synthetic --models simple_cnn mobilenet_v2 --samples 3
```

### Comprehensive Multi-Dataset Analysis

```bash
# Run comprehensive test across all datasets
python comprehensive_test.py
```

This will create separate result directories for CIFAR-10, ImageNet-like, and synthetic datasets, allowing for direct comparison of model performance across different data distributions.

## Example Results

The benchmark provides insights such as:
- Which models are most efficient for XAI analysis on real vs. synthetic data
- How model accuracy differs between datasets  
- Hardware resource requirements for different model sizes and data types
- Comparative performance of various XAI methods across datasets
- Memory and compute bottlenecks for different data distributions

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for:
- Additional model architectures
- New XAI methods
- Enhanced hardware monitoring
- Performance optimizations

## License

This project is licensed under the MIT License - see the LICENSE file for details.
