"""
Utility functions for data handling and sample generation.
"""

import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List


class SampleDataset(Dataset):
    """Generate synthetic sample data for testing."""
    
    def __init__(self, num_samples: int = 100, image_size: Tuple[int, int] = (224, 224), 
                 num_classes: int = 10):
        self.num_samples = num_samples
        self.image_size = image_size
        self.num_classes = num_classes
        
        # Generate random data
        self.data = torch.randn(num_samples, 3, *image_size)
        self.labels = torch.randint(0, num_classes, (num_samples,))
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def get_sample_images(num_samples: int = 5, 
                     image_size: Tuple[int, int] = (224, 224),
                     device: str = 'cpu') -> torch.Tensor:
    """
    Generate sample images for XAI analysis.
    
    Args:
        num_samples: Number of sample images to generate
        image_size: Size of generated images (H, W)
        device: Device to place tensors on
        
    Returns:
        Tensor of sample images
    """
    # Create more realistic sample images with patterns
    images = []
    
    for i in range(num_samples):
        # Create a base image with gradients and patterns
        h, w = image_size
        
        # Create gradient patterns
        x = torch.linspace(0, 1, w).repeat(h, 1)
        y = torch.linspace(0, 1, h).repeat(w, 1).t()
        
        # Different patterns for different samples
        if i % 5 == 0:
            # Circular pattern
            center_x, center_y = w // 2, h // 2
            xx, yy = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
            circle = ((xx - center_y) ** 2 + (yy - center_x) ** 2) / (min(h, w) / 4) ** 2
            pattern = torch.exp(-circle)
        elif i % 5 == 1:
            # Diagonal stripes
            pattern = torch.sin(0.1 * (x + y) * np.pi)
        elif i % 5 == 2:
            # Checkerboard
            pattern = torch.sin(0.3 * x * np.pi) * torch.sin(0.3 * y * np.pi)
        elif i % 5 == 3:
            # Radial pattern
            center_x, center_y = w // 2, h // 2
            xx, yy = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
            radius = torch.sqrt((xx - center_y) ** 2 + (yy - center_x) ** 2)
            pattern = torch.sin(0.1 * radius)
        else:
            # Random noise with structure
            pattern = torch.randn(h, w) * 0.3 + x * y
        
        # Create RGB channels with variations
        r_channel = pattern + 0.1 * torch.randn(h, w)
        g_channel = pattern * 0.8 + 0.1 * torch.randn(h, w) 
        b_channel = pattern * 0.6 + 0.1 * torch.randn(h, w)
        
        # Stack channels and normalize
        image = torch.stack([r_channel, g_channel, b_channel])
        image = torch.clamp(image, -2, 2)  # Reasonable range for normalized images
        
        images.append(image)
    
    return torch.stack(images).to(device)


def get_imagenet_transforms(image_size: int = 224) -> transforms.Compose:
    """Get standard ImageNet preprocessing transforms."""
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])


def get_cifar10_transforms(image_size: int = 32) -> transforms.Compose:
    """Get standard CIFAR-10 preprocessing transforms."""
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], 
                           std=[0.2023, 0.1994, 0.2010])
    ])


def create_sample_dataloader(batch_size: int = 32, 
                           num_samples: int = 100,
                           image_size: Tuple[int, int] = (224, 224),
                           num_classes: int = 10) -> DataLoader:
    """Create a DataLoader with sample data."""
    dataset = SampleDataset(num_samples, image_size, num_classes)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def denormalize_image(tensor: torch.Tensor, 
                     mean: List[float] = [0.485, 0.456, 0.406],
                     std: List[float] = [0.229, 0.224, 0.225]) -> torch.Tensor:
    """
    Denormalize an image tensor for visualization.
    
    Args:
        tensor: Normalized image tensor
        mean: Normalization mean values
        std: Normalization std values
        
    Returns:
        Denormalized tensor
    """
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    
    return torch.clamp(tensor, 0, 1)


def visualize_batch(images: torch.Tensor, 
                   labels: Optional[torch.Tensor] = None,
                   predictions: Optional[torch.Tensor] = None,
                   num_images: int = 8,
                   save_path: Optional[str] = None) -> plt.Figure:
    """
    Visualize a batch of images.
    
    Args:
        images: Batch of images to visualize
        labels: True labels (optional)
        predictions: Predicted labels (optional)
        num_images: Number of images to show
        save_path: Path to save the visualization
        
    Returns:
        Matplotlib figure
    """
    num_images = min(num_images, images.shape[0])
    rows = int(np.sqrt(num_images))
    cols = int(np.ceil(num_images / rows))
    
    fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_images):
        row, col = i // cols, i % cols
        
        # Get image and convert to displayable format
        img = images[i].cpu()
        if img.shape[0] == 3:  # RGB
            img = img.permute(1, 2, 0)
        
        # Normalize for display
        img = (img - img.min()) / (img.max() - img.min())
        
        axes[row, col].imshow(img)
        axes[row, col].axis('off')
        
        # Add title with labels/predictions
        title = f"Sample {i}"
        if labels is not None:
            title += f"\nTrue: {labels[i].item()}"
        if predictions is not None:
            title += f"\nPred: {predictions[i].item()}"
        
        axes[row, col].set_title(title, fontsize=10)
    
    # Hide extra subplots
    for i in range(num_images, rows * cols):
        row, col = i // cols, i % cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def get_model_input_size(model_name: str) -> Tuple[int, int]:
    """Get the expected input size for a model."""
    size_mapping = {
        'simple_cnn': (32, 32),
        'resnet18': (224, 224),
        'resnet50': (224, 224),
        'resnet101': (224, 224),
        'mobilenet_v2': (224, 224),
        'mobilenet_v3_small': (224, 224),
        'mobilenet_v3_large': (224, 224),
        'efficientnet_b0': (224, 224),
        'vgg16': (224, 224),
    }
    
    return size_mapping.get(model_name.lower(), (224, 224))