"""
Datasets Package
This package contains various dataset loaders for XAI analysis.

Available datasets:
- cifar10: CIFAR-10 classification dataset
- cifar100: CIFAR-100 classification dataset

Each dataset module should provide:
- A get_dataloaders() function that returns (train_loader, val_loader)
- A get_test_dataloader() function that returns test_loader
- Optional utility functions like get_class_names()

Usage:
    from datasets.cifar10 import get_dataloaders
    train_loader, val_loader = get_dataloaders(batch_size=32)
"""

__all__ = ['cifar10', 'cifar100']