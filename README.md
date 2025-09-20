# XAI Analysis Project

This project provides tools for training deep learning models and performing Explainable AI (XAI) analysis using the Captum library. It includes both a Jupyter notebook for interactive analysis and a comprehensive training framework for various model architectures.

## Features

- **Interactive XAI Analysis**: Jupyter notebook (`CIFAR_TorchVision_Interpret.ipynb`) for exploring model interpretability
- **Training Framework**: Comprehensive training scripts supporting multiple models and datasets
- **Multiple Model Architectures**: Support for CNN, ResNet18, and VGG16 models
- **Multiple Datasets**: CIFAR-10 and CIFAR-100 support with data augmentation options
- **Flexible Configuration**: YAML-based configuration system and command-line interface
- **Experiment Tracking**: Built-in logging and TensorBoard integration
- **Model Checkpointing**: Automatic model saving and resuming capabilities

## Quick Start

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Train a Model

Basic training:
```bash
python train.py --model net --dataset cifar10 --epochs 20
```

Using configuration file:
```bash
python train_from_config.py configs/cifar10_net.yaml
```

### 3. XAI Analysis

Open the Jupyter notebook:
```bash
jupyter notebook CIFAR_TorchVision_Interpret.ipynb
```

Or use the integration example:
```bash
python example_integration.py
```

## Project Structure

```
├── CIFAR_TorchVision_Interpret.ipynb  # Original XAI analysis notebook
├── train.py                           # Main training script
├── train_from_config.py               # YAML config-based training
├── datasets.py                        # Dataset utilities
├── models/                            # Model architectures
├── configs/                           # Configuration files
├── checkpoints/                       # Saved model checkpoints
├── example_integration.py             # Integration example
└── TRAINING_README.md                 # Detailed training documentation
```

## Detailed Documentation

- See [TRAINING_README.md](TRAINING_README.md) for comprehensive training documentation
- Explore the Jupyter notebook for interactive XAI analysis examples
- Check `example_integration.py` for model integration patterns

## Original Purpose

This project was originally created to debug the hardware performance of the Captum package and has been extended into a full training and analysis framework for XAI research.
