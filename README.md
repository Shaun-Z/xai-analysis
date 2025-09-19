# XAI Analysis with Hardware Monitoring

This project implements comprehensive XAI (Explainable AI) analysis using the Captum library on various deep learning models (CNN, ResNet, MobileNet, etc.) while monitoring hardware performance indicators like power, throughput, CPU/GPU usage, and memory consumption.

## Features

- **Multiple Model Architectures**: CNN, ResNet (18/50/101), MobileNet (v2/v3), EfficientNet, VGG16
- **Comprehensive XAI Methods**: Integrated Gradients, SHAP, DeepLIFT, Saliency, Occlusion, LIME, and more
- **Hardware Performance Monitoring**: CPU, memory, GPU utilization, power consumption, and throughput
- **Automated Benchmarking**: Run comprehensive analysis across multiple models and samples
- **Visualization**: Generate attribution visualizations and performance comparison plots
- **Detailed Reporting**: JSON results and comprehensive analysis reports

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

Run a simple example:
```bash
python example.py
```

This will run XAI analysis on a subset of models with hardware monitoring and save results to `example_results/`.

## Advanced Usage

### Full Benchmark

Run comprehensive analysis on all models:
```bash
python src/main_analysis.py --models simple_cnn resnet18 resnet50 mobilenet_v2 --samples 5
```

### Command Line Options

```bash
python src/main_analysis.py [OPTIONS]

Options:
  --models MODEL [MODEL ...]    Models to benchmark (default: simple_cnn resnet18 mobilenet_v2)
  --samples INT                 Number of samples per model (default: 3)
  --quick                       Use quick mode for faster analysis
  --output DIR                  Output directory (default: results)
  --device {cpu,cuda}           Device to use (auto-detect if not specified)
```

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
├── benchmark_results.json          # Complete benchmark data
├── performance_comparison.png      # Performance comparison plots
├── hardware_monitoring.png         # Hardware usage plots
└── [model]_sample_[n]_[method].png # Individual attribution visualizations
```

## Results Analysis

The system generates:

1. **JSON Results**: Complete benchmark data with timing, hardware stats, and method performance
2. **Performance Plots**: Bar charts comparing processing times and throughput across models
3. **Hardware Plots**: Box plots showing CPU, memory, and GPU usage patterns
4. **Attribution Visualizations**: Heatmaps showing XAI method outputs for each sample

## Example Results

The benchmark provides insights such as:
- Which models are most efficient for XAI analysis
- Hardware resource requirements for different model sizes
- Comparative performance of various XAI methods
- Memory and compute bottlenecks

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for:
- Additional model architectures
- New XAI methods
- Enhanced hardware monitoring
- Performance optimizations

## License

This project is licensed under the MIT License - see the LICENSE file for details.
