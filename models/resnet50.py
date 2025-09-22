"""
ResNet50 model for ImageNet50 classification.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class ResNet50(nn.Module):
    """ResNet50 model for classification."""
    
    def __init__(self, num_classes=50, pretrained=True, freeze_backbone=False):
        """
        Args:
            num_classes (int): Number of output classes
            pretrained (bool): Whether to use pretrained ImageNet weights
            freeze_backbone (bool): Whether to freeze the backbone for fine-tuning
        """
        super(ResNet50, self).__init__()
        
        # Load pretrained ResNet50
        self.backbone = models.resnet50(pretrained=pretrained)
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Replace the final fully connected layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)
        
        # Initialize the new classifier layer
        nn.init.xavier_uniform_(self.backbone.fc.weight)
        nn.init.zeros_(self.backbone.fc.bias)
    
    def forward(self, x):
        return self.backbone(x)
    
    def get_features(self, x):
        """Get features before the final classification layer."""
        # Forward through all layers except the final fc layer
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        
        return x
    
    def unfreeze_backbone(self):
        """Unfreeze the backbone for full fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True
    
    def freeze_backbone(self):
        """Freeze the backbone for feature extraction only."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        # Keep the final layer unfrozen
        for param in self.backbone.fc.parameters():
            param.requires_grad = True


def create_resnet50(num_classes=50, pretrained=True, freeze_backbone=False):
    """Factory function to create ResNet50 model."""
    return ResNet50(num_classes=num_classes, pretrained=pretrained, freeze_backbone=freeze_backbone)


if __name__ == "__main__":
    # Test the model
    model = create_resnet50(num_classes=50, pretrained=True)
    
    # Test forward pass
    x = torch.randn(2, 3, 224, 224)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test feature extraction
    features = model.get_features(x)
    print(f"Features shape: {features.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")