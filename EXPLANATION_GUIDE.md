# XAI Explanation Usage Guide

This guide explains how to use the `explain.py` script for explainable AI analysis of your trained models.

## Quick Start

```bash
# Basic usage with default methods
python explain.py --model cifar_cnn --dataset cifar10 --checkpoint checkpoints/cifar_cnn_cifar10/best_model.pt

# Specify multiple explanation methods
python explain.py --model cifar_cnn --dataset cifar10 --checkpoint checkpoints/cifar_cnn_cifar10/best_model.pt \
  --methods integrated_gradients saliency gradcam deeplift occlusion

# Analyze specific samples
python explain.py --model cifar_cnn --dataset cifar10 --checkpoint checkpoints/cifar_cnn_cifar10/best_model.pt \
  --num-samples 5 --sample-indices 0 10 25 50 100
```

## Available Attribution Methods

### 1. **Integrated Gradients** (`integrated_gradients`)
- **Description**: Computes gradients along a path from baseline to input
- **Best for**: Understanding pixel-level importance with high fidelity
- **Parameters**: `--n-steps` (default: 50), `--baseline-type`

### 2. **Saliency Maps** (`saliency`)
- **Description**: Simple gradient-based attribution
- **Best for**: Quick insights into input sensitivity
- **Speed**: Very fast

### 3. **GradCAM** (`gradcam`)
- **Description**: Gradient-weighted Class Activation Mapping
- **Best for**: Understanding which regions the model focuses on
- **Note**: Works with convolutional layers

### 4. **Gradient SHAP** (`gradient_shap`)
- **Description**: Combines gradients with SHAP values
- **Best for**: Game-theoretic explanations
- **Parameters**: Uses random baselines

### 5. **DeepLift** (`deeplift`)
- **Description**: Compares activation to reference activation
- **Best for**: Understanding activation differences

### 6. **Occlusion** (`occlusion`)
- **Description**: Systematically occludes parts of input
- **Best for**: Understanding spatial importance
- **Note**: Computationally intensive

### 7. **Layer-wise Relevance Propagation** (`lrp`)
- **Description**: Propagates relevance backwards through layers
- **Best for**: Layer-by-layer understanding

### 8. **Guided Backpropagation** (`guided_backprop`)
- **Description**: Modified backpropagation focusing on positive influences
- **Best for**: Sharp, detailed attributions

## Command Line Arguments

### Required Arguments
- `--model`: Model name (Python module in models/ folder)
- `--dataset`: Dataset name (Python module in datasets/ folder)  
- `--checkpoint`: Path to model checkpoint file

### Method Selection
- `--methods`: List of attribution methods to use (default: integrated_gradients saliency gradcam)

### Sample Selection
- `--num-samples`: Number of samples to explain (default: 10)
- `--sample-indices`: Specific sample indices to analyze
- `--target-classes`: Focus on specific target classes

### Attribution Parameters
- `--n-steps`: Steps for Integrated Gradients (default: 50)
- `--baseline-type`: Baseline type (zero, random, gaussian)
- `--noise-tunnel`: Use noise tunnel for robust attributions

### Visualization Options
- `--viz-method`: Visualization type (heat_map, blended_heat_map, original_image, masked_image)
- `--cmap`: Colormap for visualizations (default: RdYlBu_r)
- `--alpha-overlay`: Alpha for overlay visualizations (default: 0.4)

### Output Settings
- `--results-dir`: Directory to save results (default: results)
- `--experiment-name`: Experiment name for organization
- `--save-format`: Image format (png, jpg, pdf)

## Output Structure

```
results/
└── experiment_name/
    ├── attributions/          # Raw attribution tensors (.pt files)
    │   ├── sample_0_integrated_gradients_attribution.pt
    │   ├── sample_0_saliency_attribution.pt
    │   └── sample_0_gradcam_attribution.pt
    ├── visualizations/        # Attribution visualizations (.png files)
    │   ├── sample_0_integrated_gradients.png
    │   ├── sample_0_saliency.png
    │   └── sample_0_gradcam.png
    └── reports/              # JSON reports with metadata
        └── experiment_report.json
```

## Example Usage Scenarios

### 1. **Model Debugging**
```bash
# Find misclassified samples and understand why
python explain.py --model cifar_cnn --dataset cifar10 --checkpoint best_model.pt \
  --methods integrated_gradients gradcam \
  --target-classes 3 5 8 --num-samples 20
```

### 2. **Comprehensive Analysis**
```bash
# Use all available methods on selected samples
python explain.py --model cifar_cnn --dataset cifar10 --checkpoint best_model.pt \
  --methods integrated_gradients saliency gradcam deeplift gradient_shap occlusion \
  --num-samples 5 --experiment-name comprehensive_analysis
```

### 3. **High-Quality Explanations**
```bash
# High-resolution explanations with fine-tuned parameters
python explain.py --model cifar_cnn --dataset cifar10 --checkpoint best_model.pt \
  --methods integrated_gradients --n-steps 100 --noise-tunnel \
  --viz-method blended_heat_map --save-format pdf
```

### 4. **Batch Processing**
```bash
# Process specific important samples
python explain.py --model cifar_cnn --dataset cifar10 --checkpoint best_model.pt \
  --sample-indices 0 100 200 300 400 500 \
  --methods integrated_gradients saliency \
  --experiment-name batch_analysis
```

## Understanding Results

### Attribution Visualizations
Each visualization shows three panels:
1. **Original Image**: The input image with true/predicted labels
2. **Attribution Heatmap**: Color-coded importance map
3. **Overlay**: Attribution overlaid on original image

### Color Interpretation
- **Red/Warm colors**: Positive attribution (supports prediction)
- **Blue/Cool colors**: Negative attribution (contradicts prediction)
- **Intensity**: Strength of attribution

### Report File
The JSON report contains:
- Experiment metadata and parameters
- Per-sample results with confidence scores
- File paths to attributions and visualizations
- Summary statistics

## Performance Tips

### Speed Optimization
- Use `saliency` for quick insights
- Limit `--num-samples` for large datasets
- Use `--batch-size 1` (default) for memory efficiency

### Quality Optimization
- Increase `--n-steps` for Integrated Gradients (50-200)
- Use `--noise-tunnel` for more robust results
- Combine multiple methods for comprehensive understanding

### Memory Management
- Process samples in batches if memory is limited
- Use CPU (`--device cpu`) if GPU memory is insufficient
- Choose fewer methods for large-scale analysis

## Troubleshooting

### Common Issues
1. **CUDA out of memory**: Use `--device cpu` or reduce `--num-samples`
2. **Method not supported**: Check model architecture compatibility (e.g., GradCAM needs conv layers)
3. **Poor attribution quality**: Increase `--n-steps` or try different baseline types

### Model Compatibility
- Ensure your model is compatible with Captum
- CNN models work best with spatial attribution methods
- Some methods require specific layer types

## Integration with Training Pipeline

```bash
# After training, explain the best model
python train.py --model cifar_cnn --dataset cifar10 --epochs 100
python explain.py --model cifar_cnn --dataset cifar10 --checkpoint checkpoints/best_model.pt
```

This creates a complete XAI analysis pipeline from training to explanation!