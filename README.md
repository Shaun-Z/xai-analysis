# XAI Analysis

This project provides a comprehensive framework for XAI (Explainable AI) analysis with a focus on debugging hardware performance of the 'Captum' package. It includes a flexible training framework with Wandb integration for experiment tracking and model training.

## Overview

The project combines explainable AI research with practical model training capabilities:
- **XAI Analysis**: Debug and analyze hardware performance of Captum package
- **Training Framework**: Flexible model training with extensive logging and checkpointing
- **Experiment Tracking**: Integrated Wandb support for comprehensive experiment management

## Project Structure

```
xai-analysis/
├── train.py                          # Main training script
├── CIFAR_TorchVision_Interpret.ipynb # XAI analysis notebook
├── models/                           # Model definitions
│   ├── __init__.py
│   ├── cifar_cnn.py                  # Simple CNN for CIFAR-10
│   ├── resnet.py                     # ResNet architectures
│   └── cifar_torchvision.pt          # Pre-trained model
├── datasets/                         # Dataset loaders
│   ├── __init__.py
│   ├── cifar10.py                    # CIFAR-10 dataset
│   └── cifar100.py                   # CIFAR-100 dataset
├── checkpoints/                      # Training checkpoints
├── data/                            # Raw data storage
│   └── cifar-10-batches-py/         # CIFAR-10 data
├── requirements.txt                  # Python dependencies
└── README.md                        # This file
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Login to Wandb (optional, for experiment tracking):
```bash
wandb login
```

## Usage

### Training Framework

#### Basic Training

Train a simple CNN on CIFAR-10:
```bash
python train.py --model cifar_cnn --dataset cifar10 --epochs 20 --batch-size 128
```

Train ResNet-18 on CIFAR-10:
```bash
python train.py --model resnet --dataset cifar10 --epochs 50 --lr 0.01 --optimizer SGD
```

#### Advanced Training Options

Full training with custom settings:
```bash
python train.py \
    --model resnet \
    --dataset cifar10 \
    --epochs 100 \
    --batch-size 128 \
    --lr 0.1 \
    --optimizer SGD \
    --momentum 0.9 \
    --weight-decay 1e-4 \
    --scheduler StepLR \
    --experiment-name resnet18_cifar10_experiment \
    --save-every 10 \
    --wandb-project my-xai-project
```

#### Resume Training

Resume from a checkpoint:
```bash
python train.py \
    --model resnet \
    --dataset cifar10 \
    --resume checkpoints/resnet18_cifar10_experiment/epoch_50.pt
```

Resume from the best model:
```bash
python train.py \
    --model resnet \
    --dataset cifar10 \
    --resume checkpoints/resnet18_cifar10_experiment/best_model.pt
```

### XAI Analysis

Open and run the Jupyter notebook for XAI analysis:
```bash
jupyter notebook CIFAR_TorchVision_Interpret.ipynb
```

The notebook includes:
- Model training and evaluation
- Captum integration for explainability analysis
- Various attribution methods (Integrated Gradients, DeepLift, Saliency, etc.)
- Performance debugging for hardware optimization

### Command Line Options

- `--model`: Model name (must match a module in `models/` folder)
- `--dataset`: Dataset name (must match a module in `datasets/` folder)
- `--epochs`: Number of training epochs (default: 10)
- `--batch-size`: Batch size (default: 32)
- `--lr`: Learning rate (default: 0.001)
- `--optimizer`: Optimizer type [Adam, SGD, RMSprop] (default: Adam)
- `--scheduler`: LR scheduler [StepLR, CosineAnnealingLR, ReduceLROnPlateau]
- `--experiment-name`: Custom experiment name for logging
- `--checkpoint-dir`: Directory to save checkpoints (default: checkpoints)
- `--resume`: Path to checkpoint to resume from
- `--device`: Device to use [auto, cpu, cuda, mps] (default: auto)
- `--wandb-project`: Wandb project name (default: xai-analysis)
- `--no-wandb`: Disable Wandb logging

## Extending the Framework

### Adding New Models

Create a new model file in `models/` directory:

```python
# models/my_model.py
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, num_classes=10):
        super(MyModel, self).__init__()
        # Define your model architecture
        
    def forward(self, x):
        # Define forward pass
        return x

def create_model(num_classes=10):
    """Create and return model instance."""
    return MyModel(num_classes=num_classes)
```

Then train with:
```bash
python train.py --model my_model --dataset cifar10
```

### Adding New Datasets

Create a new dataset file in `datasets/` directory:

```python
# datasets/my_dataset.py
from torch.utils.data import DataLoader

def get_dataloaders(batch_size=32, num_workers=4):
    """Create train and validation data loaders."""
    # Implement your data loading logic
    return train_loader, val_loader
```

Then train with:
```bash
python train.py --model cifar_cnn --dataset my_dataset
```

## Features

### Wandb Integration

The training script automatically logs:
- Training and validation loss
- Training and validation accuracy
- Learning rate schedules
- Model hyperparameters
- System metrics

View your experiments at [wandb.ai](https://wandb.ai) after logging in.

### Checkpoint System

Checkpoints are automatically saved in experiment-specific subfolders:
- Every N epochs (configurable with `--save-every`)
- At the end of training
- When achieving the best validation loss (saved as `best_model.pt`)

**Checkpoint Organization:**
```
checkpoints/
├── experiment_name_1/
│   ├── epoch_10.pt
│   ├── epoch_20.pt
│   ├── best_model.pt
│   └── ...
└── experiment_name_2/
    ├── epoch_5.pt
    ├── best_model.pt
    └── ...
```

Each checkpoint contains:
- Model state dict
- Optimizer state dict
- Scheduler state dict
- Current epoch
- Current loss
- Timestamp

### XAI Analysis Tools

The notebook provides comprehensive XAI analysis:
- Multiple attribution methods via Captum
- Visualization of feature importance
- Hardware performance profiling
- Model interpretability analysis

## Examples

### Quick Start: CIFAR-10 with Simple CNN
```bash
python train.py --model cifar_cnn --dataset cifar10 --epochs 10
```

### Production Training: ResNet on CIFAR-10
```bash
python train.py \
    --model resnet \
    --dataset cifar10 \
    --epochs 200 \
    --batch-size 128 \
    --lr 0.1 \
    --optimizer SGD \
    --momentum 0.9 \
    --weight-decay 5e-4 \
    --scheduler CosineAnnealingLR \
    --experiment-name resnet18_cifar10_production
```

### Transfer Learning Setup
```bash
python train.py \
    --model resnet \
    --dataset cifar100 \
    --epochs 100 \
    --lr 0.01 \
    --scheduler ReduceLROnPlateau \
    --experiment-name transfer_learning_cifar100
```

## Contributing

When contributing to this project:
1. Follow the established model and dataset patterns
2. Update documentation for new features
3. Test both training framework and XAI analysis components
4. Ensure compatibility with existing Captum analysis workflow
