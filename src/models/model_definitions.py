"""
Model definitions for XAI analysis including CNN, ResNet, MobileNet variants.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Dict, Any, Tuple


class SimpleCNN(nn.Module):
    """Simple CNN for CIFAR-10/ImageNet classification."""
    
    def __init__(self, num_classes: int = 10, input_channels: int = 3):
        super(SimpleCNN, self).__init__()
        
        self.features = nn.Sequential(
            # First convolutional block
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Second convolutional block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Third convolutional block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Fourth convolutional block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class ModelFactory:
    """Factory class for creating different model architectures."""
    
    @staticmethod
    def create_model(model_name: str, num_classes: int = 1000, pretrained: bool = True) -> nn.Module:
        """
        Create a model instance.
        
        Args:
            model_name: Name of the model architecture
            num_classes: Number of output classes
            pretrained: Whether to load pretrained weights
            
        Returns:
            PyTorch model instance
        """
        model_name = model_name.lower()
        
        if model_name == 'simple_cnn':
            return SimpleCNN(num_classes=num_classes)
        
        elif model_name == 'resnet18':
            model = models.resnet18(pretrained=pretrained)
            if num_classes != 1000:
                model.fc = nn.Linear(model.fc.in_features, num_classes)
            return model
        
        elif model_name == 'resnet50':
            model = models.resnet50(pretrained=pretrained)
            if num_classes != 1000:
                model.fc = nn.Linear(model.fc.in_features, num_classes)
            return model
        
        elif model_name == 'resnet101':
            model = models.resnet101(pretrained=pretrained)
            if num_classes != 1000:
                model.fc = nn.Linear(model.fc.in_features, num_classes)
            return model
        
        elif model_name == 'mobilenet_v2':
            model = models.mobilenet_v2(pretrained=pretrained)
            if num_classes != 1000:
                model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
            return model
        
        elif model_name == 'mobilenet_v3_small':
            model = models.mobilenet_v3_small(pretrained=pretrained)
            if num_classes != 1000:
                model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
            return model
        
        elif model_name == 'mobilenet_v3_large':
            model = models.mobilenet_v3_large(pretrained=pretrained)
            if num_classes != 1000:
                model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
            return model
        
        elif model_name == 'efficientnet_b0':
            model = models.efficientnet_b0(pretrained=pretrained)
            if num_classes != 1000:
                model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
            return model
        
        elif model_name == 'vgg16':
            model = models.vgg16(pretrained=pretrained)
            if num_classes != 1000:
                model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
            return model
        
        else:
            raise ValueError(f"Unsupported model: {model_name}")
    
    @staticmethod
    def get_model_info(model_name: str) -> Dict[str, Any]:
        """Get information about a model architecture."""
        info = {
            'simple_cnn': {
                'description': 'Simple 4-layer CNN',
                'parameters': '~2M',
                'input_size': (3, 32, 32),
                'suitable_for': ['CIFAR-10', 'small datasets']
            },
            'resnet18': {
                'description': 'ResNet-18 architecture',
                'parameters': '~11M',
                'input_size': (3, 224, 224),
                'suitable_for': ['ImageNet', 'general purpose']
            },
            'resnet50': {
                'description': 'ResNet-50 architecture',
                'parameters': '~25M',
                'input_size': (3, 224, 224),
                'suitable_for': ['ImageNet', 'high accuracy tasks']
            },
            'resnet101': {
                'description': 'ResNet-101 architecture',
                'parameters': '~44M',
                'input_size': (3, 224, 224),
                'suitable_for': ['ImageNet', 'complex tasks']
            },
            'mobilenet_v2': {
                'description': 'MobileNet-V2 efficient architecture',
                'parameters': '~3.5M',
                'input_size': (3, 224, 224),
                'suitable_for': ['mobile deployment', 'edge devices']
            },
            'mobilenet_v3_small': {
                'description': 'MobileNet-V3 Small',
                'parameters': '~2.5M',
                'input_size': (3, 224, 224),
                'suitable_for': ['very constrained devices']
            },
            'mobilenet_v3_large': {
                'description': 'MobileNet-V3 Large',
                'parameters': '~5.5M',
                'input_size': (3, 224, 224),
                'suitable_for': ['mobile deployment']
            },
            'efficientnet_b0': {
                'description': 'EfficientNet-B0',
                'parameters': '~5M',
                'input_size': (3, 224, 224),
                'suitable_for' : ['efficient high-accuracy models']
            },
            'vgg16': {
                'description': 'VGG-16 architecture',
                'parameters': '~138M',
                'input_size': (3, 224, 224),
                'suitable_for': ['feature extraction', 'transfer learning']
            }
        }
        
        return info.get(model_name.lower(), {'description': 'Unknown model'})
    
    @staticmethod
    def get_available_models() -> list:
        """Get list of available model names."""
        return [
            'simple_cnn', 'resnet18', 'resnet50', 'resnet101',
            'mobilenet_v2', 'mobilenet_v3_small', 'mobilenet_v3_large',
            'efficientnet_b0', 'vgg16'
        ]


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size_mb(model: nn.Module) -> float:
    """Estimate model size in MB."""
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb