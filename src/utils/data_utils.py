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


def load_cifar10_data(num_samples: int = 10, 
                     batch_size: int = 32, 
                     shuffle: bool = True,
                     download: bool = True) -> Tuple[DataLoader, List[str]]:
    """
    Load CIFAR-10 test dataset.
    
    Args:
        num_samples: Maximum number of samples to load (None for all)
        batch_size: Batch size for DataLoader
        shuffle: Whether to shuffle the data
        download: Whether to download the dataset if not present
        
    Returns:
        Tuple of (DataLoader, class_names)
    """
    # CIFAR-10 class names
    class_names = [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]
    
    try:
        # Try to use real CIFAR-10 if available
        transform = get_cifar10_transforms(32)  # Keep original CIFAR-10 size
        
        dataset = datasets.CIFAR10(
            root='./data', 
            train=False, 
            download=download, 
            transform=transform
        )
        
        # Limit number of samples if specified
        if num_samples is not None and num_samples < len(dataset):
            indices = torch.randperm(len(dataset))[:num_samples]
            dataset = torch.utils.data.Subset(dataset, indices)
            
    except Exception as e:
        print(f"Could not download CIFAR-10 ({e}), creating CIFAR-10-like synthetic dataset...")
        
        # Create CIFAR-10-like synthetic dataset
        class CIFAR10LikeDataset(Dataset):
            def __init__(self, num_samples: int):
                self.num_samples = num_samples
                self.data = []
                self.labels = []
                
                for i in range(num_samples):
                    # Create CIFAR-10 sized images (32x32) with different patterns
                    if i % 10 == 0:  # airplane-like
                        img = self._create_airplane_like()
                    elif i % 10 == 1:  # automobile-like
                        img = self._create_automobile_like()
                    elif i % 10 == 2:  # bird-like
                        img = self._create_bird_like()
                    else:  # other patterns
                        img = self._create_generic_pattern(i % 10)
                    
                    # Convert to PIL Image for transforms
                    img_pil = transforms.ToPILImage()(img)
                    self.data.append(img_pil)
                    self.labels.append(i % 10)  # Cycle through 10 classes
            
            def _create_airplane_like(self):
                # Create airplane-like pattern (horizontal lines in sky)
                img = torch.zeros(3, 32, 32)
                # Sky blue background
                img[0] = 0.5  # R
                img[1] = 0.7  # G  
                img[2] = 1.0  # B
                # Add horizontal lines for airplane body/wings
                img[:, 14:18, 8:24] = 0.8  # Body
                img[:, 15:17, 6:26] = 0.6  # Wings
                return img
                
            def _create_automobile_like(self):
                # Create automobile-like pattern (rectangular with wheels)
                img = torch.zeros(3, 32, 32)
                # Road background
                img[0] = 0.3  # R
                img[1] = 0.3  # G
                img[2] = 0.3  # B
                # Car body (rectangle)
                img[:, 12:20, 8:24] = torch.tensor([0.8, 0.2, 0.2])[:, None, None]  # Red car
                # Wheels
                img[:, 18:22, 10:14] = 0.1  # Left wheel
                img[:, 18:22, 18:22] = 0.1  # Right wheel
                return img
                
            def _create_bird_like(self):
                # Create bird-like pattern (flying shape in sky)
                img = torch.zeros(3, 32, 32)
                # Sky background
                img[0] = 0.6  # R
                img[1] = 0.8  # G
                img[2] = 1.0  # B
                # Bird body (small oval)
                img[:, 14:18, 14:18] = torch.tensor([0.4, 0.3, 0.1])[:, None, None]  # Brown bird
                # Wings (triangular shapes)
                img[:, 15:17, 10:14] = 0.5  # Left wing
                img[:, 15:17, 18:22] = 0.5  # Right wing
                return img
                
            def _create_generic_pattern(self, class_idx):
                # Create generic patterns for other classes
                img = torch.randn(3, 32, 32) * 0.1 + 0.5
                # Add class-specific patterns
                if class_idx == 3:  # cat-like
                    img[:, 10:22, 10:22] = torch.tensor([0.9, 0.6, 0.3])[:, None, None]  # Orange cat
                elif class_idx == 4:  # deer-like  
                    img[:, 8:24, 12:20] = torch.tensor([0.6, 0.4, 0.2])[:, None, None]  # Brown deer
                elif class_idx == 5:  # dog-like
                    img[:, 12:20, 12:20] = torch.tensor([0.8, 0.7, 0.5])[:, None, None]  # Light brown dog
                else:
                    # Random colored rectangles for other classes
                    color = torch.rand(3)
                    img[:, 10:22, 10:22] = color[:, None, None]
                    
                return torch.clamp(img, 0, 1)
            
            def __len__(self):
                return self.num_samples
            
            def __getitem__(self, idx):
                img = self.data[idx]
                transform = get_cifar10_transforms(32)
                if transform:
                    img = transform(img)
                return img, self.labels[idx]
        
        dataset = CIFAR10LikeDataset(num_samples)
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle,
        num_workers=0  # Set to 0 to avoid multiprocessing issues
    )
    
    return dataloader, class_names


def load_imagenet_subset(num_samples: int = 10,
                        batch_size: int = 32,
                        shuffle: bool = True) -> Tuple[DataLoader, List[str]]:
    """
    Load a subset of ImageNet validation data.
    Since ImageNet is large, we'll create a representative subset.
    
    Args:
        num_samples: Number of samples to generate
        batch_size: Batch size for DataLoader
        shuffle: Whether to shuffle the data
        
    Returns:
        Tuple of (DataLoader, class_names)
    """
    # For now, we'll create synthetic ImageNet-like data with proper preprocessing
    # In a real implementation, you would load actual ImageNet validation set
    transform = get_imagenet_transforms(224)
    
    # Create synthetic ImageNet-like dataset with proper transforms
    class ImageNetLikeDataset(Dataset):
        def __init__(self, num_samples: int):
            self.num_samples = num_samples
            # Generate more realistic synthetic images for ImageNet-like data
            self.data = []
            self.labels = []
            
            for i in range(num_samples):
                # Create more structured synthetic images
                img = torch.randn(3, 224, 224)
                # Add some structure
                img[0] = torch.sin(torch.linspace(0, 10, 224).unsqueeze(0) + 
                                  torch.linspace(0, 10, 224).unsqueeze(1)) + torch.randn(224, 224) * 0.1
                img[1] = torch.cos(torch.linspace(0, 8, 224).unsqueeze(0) * 
                                  torch.linspace(0, 8, 224).unsqueeze(1)) + torch.randn(224, 224) * 0.1  
                img[2] = torch.sin(torch.linspace(0, 6, 224).unsqueeze(0) * 
                                  torch.linspace(0, 6, 224).unsqueeze(1)) + torch.randn(224, 224) * 0.1
                
                # Convert to PIL Image for transforms
                img = torch.clamp(img, 0, 1)
                img_pil = transforms.ToPILImage()(img)
                
                self.data.append(img_pil)
                self.labels.append(i % 1000)  # ImageNet has 1000 classes
        
        def __len__(self):
            return self.num_samples
        
        def __getitem__(self, idx):
            img = self.data[idx]
            if transform:
                img = transform(img)
            return img, self.labels[idx]
    
    dataset = ImageNetLikeDataset(num_samples)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
    
    # Simplified class names (in practice, you'd load the full ImageNet class list)
    class_names = [f"class_{i}" for i in range(1000)]
    
    return dataloader, class_names


def get_real_data_samples(dataset_name: str,
                         num_samples: int = 5,
                         device: str = 'cpu') -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """
    Get sample images from real datasets.
    
    Args:
        dataset_name: Name of dataset ('cifar10', 'imagenet', or 'synthetic')
        num_samples: Number of samples to get
        device: Device to place tensors on
        
    Returns:
        Tuple of (images, labels, class_names)
    """
    if dataset_name.lower() == 'cifar10':
        dataloader, class_names = load_cifar10_data(num_samples=num_samples, batch_size=num_samples)
        # Get one batch
        images, labels = next(iter(dataloader))
        return images.to(device), labels.to(device), class_names
    
    elif dataset_name.lower() == 'imagenet':
        dataloader, class_names = load_imagenet_subset(num_samples=num_samples, batch_size=num_samples)
        # Get one batch  
        images, labels = next(iter(dataloader))
        return images.to(device), labels.to(device), class_names
    
    else:  # Default to synthetic data
        # Use existing synthetic data generation
        images = get_sample_images(num_samples=num_samples, image_size=(224, 224), device=device)
        labels = torch.randint(0, 1000, (num_samples,), device=device)
        class_names = [f"synthetic_class_{i}" for i in range(1000)]
        return images, labels, class_names


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