"""
Enhanced CIFAR-10 CNN Model
A modern CNN architecture with improved techniques for CIFAR-10 classification.
Includes batch normalization, dropout, deeper layers, and better architectural choices.
"""

import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    """Enhanced CNN for CIFAR-10 classification with modern techniques."""
    
    def __init__(self, num_classes=10, dropout_rate=0.5):
        super(Net, self).__init__()
        
        # First convolutional block
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Second convolutional block
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Third convolutional block
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # Global average pooling and classifier
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(128, 512)
        self.bn7 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # First block: 32x32 -> 16x16
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        
        # Second block: 16x16 -> 8x8
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        
        # Third block: 8x8 -> 4x4
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.pool3(x)
        
        # Global average pooling and classification
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = F.relu(self.bn7(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class SimpleCNN(nn.Module):
    """Original simple CNN for comparison."""
    
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.relu4 = nn.ReLU()

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = self.relu3(self.fc1(x))
        x = self.relu4(self.fc2(x))
        x = self.fc3(x)
        return x


def create_model(num_classes=10, model_type='enhanced', dropout_rate=0.5):
    """
    Create and return a CIFAR-10 CNN model.
    
    Args:
        num_classes (int): Number of output classes (default: 10)
        model_type (str): 'enhanced' for modern CNN or 'simple' for basic CNN
        dropout_rate (float): Dropout rate for enhanced model (default: 0.5)
    
    Returns:
        torch.nn.Module: CNN model
    """
    if model_type == 'enhanced':
        return Net(num_classes=num_classes, dropout_rate=dropout_rate)
    elif model_type == 'simple':
        return SimpleCNN(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Choose 'enhanced' or 'simple'.")