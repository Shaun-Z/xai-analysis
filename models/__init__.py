"""
Models Package
This package contains various neural network models for XAI analysis.

Available models:
- cifar_cnn: Simple CNN for CIFAR-10
- resnet: ResNet architectures (ResNet-18, ResNet-34)

Each model module should provide:
- A create_model() function that returns a model instance
- Model classes that inherit from torch.nn.Module

Usage:
    from models.cifar_cnn import create_model
    model = create_model(num_classes=10)
"""

__all__ = ['cifar_cnn', 'resnet']