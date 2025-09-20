# Training Script for XAI Analysis

This repository contains training scripts for various deep learning models on CIFAR datasets, specifically designed for Explainable AI (XAI) analysis.

## Directory Structure

```
├── train.py                 # Main training script
├── train_from_config.py     # Training script with YAML config support
├── datasets.py              # Dataset loading utilities
├── models/                  # Model architectures
│   ├── __init__.py
│   └── cifar_models.py      # CIFAR model definitions
├── configs/                 # Configuration files
│   ├── cifar10_net.yaml
│   └── cifar10_resnet18.yaml
├── checkpoints/             # Model checkpoints (created during training)
├── data/                    # Dataset storage (created during first run)
└── logs/                    # Training logs
```

## Quick Start

### Basic Training

Train a simple CNN on CIFAR-10:
```bash
python train.py --model net --dataset cifar10 --epochs 20
```

Train ResNet18 on CIFAR-10 with data augmentation:
```bash
python train.py --model resnet18 --dataset cifar10 --epochs 100 --augmented --batch-size 128
```

### Training with Configuration Files

Use predefined configurations:
```bash
python train_from_config.py configs/cifar10_net.yaml
```

Override specific parameters:
```bash
python train_from_config.py configs/cifar10_resnet18.yaml --override epochs=50 lr=0.01
```

## Available Models

- **net**: Simple CNN from PyTorch CIFAR-10 tutorial
- **resnet18**: ResNet18 adapted for CIFAR datasets
- **vgg16**: VGG16 adapted for CIFAR datasets

## Available Datasets

- **cifar10**: CIFAR-10 (10 classes, 32x32 RGB images)
- **cifar100**: CIFAR-100 (100 classes, 32x32 RGB images)

## Training Arguments

### Model and Dataset
- `--model`: Model architecture (net, resnet18, vgg16)
- `--dataset`: Dataset to use (cifar10, cifar100)
- `--augmented`: Use data augmentation

### Training Hyperparameters
- `--epochs`: Number of training epochs (default: 20)
- `--batch-size`: Batch size (default: 32)
- `--lr`: Learning rate (default: 0.001)
- `--momentum`: Momentum for SGD (default: 0.9)
- `--weight-decay`: Weight decay (default: 0.0)
- `--optimizer`: Optimizer (sgd, adam, adamw)

### Learning Rate Scheduling
- `--scheduler`: LR scheduler (step, cosine)
- `--step-size`: Step size for StepLR (default: 10)
- `--gamma`: Gamma for StepLR (default: 0.1)

### I/O Options
- `--data-root`: Data directory (default: ./data)
- `--checkpoint-dir`: Checkpoint directory (default: ./checkpoints)
- `--log-dir`: Log directory (default: ./logs)
- `--resume`: Resume from checkpoint

### Other Options
- `--num-workers`: Data loading workers (default: 2)
- `--print-freq`: Print frequency (default: 100)
- `--save-freq`: Checkpoint save frequency (default: 5)
- `--tensorboard`: Enable tensorboard logging
- `--tensorboard-dir`: Tensorboard log directory (default: ./runs)

## Example Training Sessions

### 1. Simple CNN on CIFAR-10
```bash
python train.py \
    --model net \
    --dataset cifar10 \
    --epochs 20 \
    --batch-size 32 \
    --lr 0.001 \
    --optimizer sgd \
    --scheduler step \
    --tensorboard
```

### 2. ResNet18 on CIFAR-10 with Augmentation
```bash
python train.py \
    --model resnet18 \
    --dataset cifar10 \
    --epochs 100 \
    --batch-size 128 \
    --lr 0.1 \
    --augmented \
    --optimizer sgd \
    --scheduler step \
    --step-size 30 \
    --gamma 0.1 \
    --weight-decay 0.0001 \
    --tensorboard
```

### 3. VGG16 on CIFAR-100
```bash
python train.py \
    --model vgg16 \
    --dataset cifar100 \
    --epochs 200 \
    --batch-size 64 \
    --lr 0.01 \
    --optimizer adam \
    --scheduler cosine \
    --weight-decay 0.001 \
    --tensorboard
```

## Monitoring Training

### Logs
Training logs are saved to the `logs/` directory with timestamps. They include:
- Training progress
- Loss and accuracy metrics
- Model checkpoints information

### Tensorboard
If enabled, tensorboard logs are saved to `runs/` directory:
```bash
tensorboard --logdir runs
```

### Checkpoints
Model checkpoints are saved to `checkpoints/` directory:
- Regular checkpoints every `save_freq` epochs
- Best model checkpoint (`best_model.pth`)
- Final model checkpoint (`final_model.pth`)

## Resuming Training

Resume from a checkpoint:
```bash
python train.py --resume checkpoints/checkpoint_epoch_10.pth --epochs 50
```

## Dependencies

Required packages:
```
torch
torchvision
tensorboard
pyyaml
```

Install with pip:
```bash
pip install torch torchvision tensorboard pyyaml
```

## Integration with Original Notebook

The trained models can be used with the original XAI analysis notebook by:

1. Loading a trained model:
```python
from models.cifar_models import get_model

# Load model
model = get_model('net', num_classes=10)
checkpoint = torch.load('checkpoints/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

2. Using the same data transformations and classes defined in `datasets.py`

This enables seamless integration between training and XAI analysis workflows.