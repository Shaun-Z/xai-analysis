"""
CIFAR-10 Dataset
Provides data loaders for CIFAR-10 classification task.
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split


def get_dataloaders(batch_size=32, num_workers=4, validation_split=0.1, data_dir='./data'):
    """
    Create CIFAR-10 train and validation data loaders.
    
    Args:
        batch_size (int): Batch size for data loaders
        num_workers (int): Number of worker processes for data loading
        validation_split (float): Fraction of training data to use for validation
        data_dir (str): Directory to store/load CIFAR-10 data
    
    Returns:
        tuple: (train_loader, val_loader)
    """
    
    # Define transforms
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Load datasets
    full_trainset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform_train
    )
    
    testset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform_test
    )

    # Split training set into train and validation
    if validation_split > 0:
        val_size = int(validation_split * len(full_trainset))
        train_size = len(full_trainset) - val_size
        
        trainset, valset = random_split(
            full_trainset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Create validation dataset with test transforms
        val_dataset = torch.utils.data.Subset(
            torchvision.datasets.CIFAR10(
                root=data_dir, train=True, download=False, transform=transform_test
            ),
            valset.indices
        )
    else:
        trainset = full_trainset
        val_dataset = testset

    # Create data loaders
    train_loader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader


def get_test_dataloader(batch_size=32, num_workers=4, data_dir='./data'):
    """
    Create CIFAR-10 test data loader.
    
    Args:
        batch_size (int): Batch size for data loader
        num_workers (int): Number of worker processes for data loading
        data_dir (str): Directory to store/load CIFAR-10 data
    
    Returns:
        DataLoader: Test data loader
    """
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    testset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform_test
    )

    test_loader = DataLoader(
        testset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return test_loader


def get_class_names():
    """Get CIFAR-10 class names."""
    return ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


# Compatibility aliases
create_dataloaders = get_dataloaders