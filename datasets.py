"""
Dataset utilities for training
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def get_cifar10_loaders(data_root='./data', batch_size=32, num_workers=2):
    """
    Get CIFAR-10 train and test data loaders
    
    Args:
        data_root (str): Root directory for data
        batch_size (int): Batch size for training
        num_workers (int): Number of workers for data loading
    
    Returns:
        tuple: (trainloader, testloader, classes)
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    trainset = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=transform
    )
    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    
    testset = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=transform
    )
    testloader = DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    return trainloader, testloader, classes


def get_cifar100_loaders(data_root='./data', batch_size=32, num_workers=2):
    """
    Get CIFAR-100 train and test data loaders
    
    Args:
        data_root (str): Root directory for data
        batch_size (int): Batch size for training
        num_workers (int): Number of workers for data loading
    
    Returns:
        tuple: (trainloader, testloader, classes)
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    trainset = torchvision.datasets.CIFAR100(
        root=data_root, train=True, download=True, transform=transform
    )
    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    
    testset = torchvision.datasets.CIFAR100(
        root=data_root, train=False, download=True, transform=transform
    )
    testloader = DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    
    # CIFAR-100 has 100 classes
    classes = None  # Too many to list, can be retrieved from dataset if needed
    
    return trainloader, testloader, classes


def get_augmented_cifar10_loaders(data_root='./data', batch_size=32, num_workers=2):
    """
    Get CIFAR-10 loaders with data augmentation
    
    Args:
        data_root (str): Root directory for data
        batch_size (int): Batch size for training
        num_workers (int): Number of workers for data loading
    
    Returns:
        tuple: (trainloader, testloader, classes)
    """
    # Training transform with augmentation
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # Test transform without augmentation
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    trainset = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=train_transform
    )
    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    
    testset = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=test_transform
    )
    testloader = DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    return trainloader, testloader, classes


def get_dataset_loaders(dataset_name, data_root='./data', batch_size=32, num_workers=2, augmented=False):
    """
    Factory function to get dataset loaders by name
    
    Args:
        dataset_name (str): Name of the dataset ('cifar10', 'cifar100')
        data_root (str): Root directory for data
        batch_size (int): Batch size for training
        num_workers (int): Number of workers for data loading
        augmented (bool): Whether to use data augmentation
    
    Returns:
        tuple: (trainloader, testloader, classes, num_classes)
    """
    if dataset_name.lower() == 'cifar10':
        if augmented:
            trainloader, testloader, classes = get_augmented_cifar10_loaders(
                data_root, batch_size, num_workers
            )
        else:
            trainloader, testloader, classes = get_cifar10_loaders(
                data_root, batch_size, num_workers
            )
        return trainloader, testloader, classes, 10
    
    elif dataset_name.lower() == 'cifar100':
        trainloader, testloader, classes = get_cifar100_loaders(
            data_root, batch_size, num_workers
        )
        return trainloader, testloader, classes, 100
    
    else:
        raise ValueError(f"Dataset {dataset_name} not supported. Available: ['cifar10', 'cifar100']")