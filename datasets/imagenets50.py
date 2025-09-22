"""
ImageNet50 Dataset
A subset of ImageNet with 50 classes for XAI analysis.
"""

import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class ImageNets50Dataset(Dataset):
    """Dataset class for ImageNet50 subset."""
    
    def __init__(self, root_dir, split='train', transform=None):
        """
        Args:
            root_dir (str): Path to imagenets50 directory
            split (str): 'train', 'validation', or 'test'
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        # Set the appropriate data directory
        if split == 'train':
            self.data_dir = os.path.join(root_dir, 'train')
        elif split == 'validation' or split == 'val':
            self.data_dir = os.path.join(root_dir, 'validation')
        elif split == 'test':
            self.data_dir = os.path.join(root_dir, 'test')
        else:
            raise ValueError("Split must be 'train', 'validation', 'val', or 'test'")
        
        # Get class names (folder names)
        self.classes = sorted([d for d in os.listdir(self.data_dir) 
                              if os.path.isdir(os.path.join(self.data_dir, d))])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        
        # Build file list
        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(self.data_dir, class_name)
            class_idx = self.class_to_idx[class_name]
            
            for filename in os.listdir(class_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    filepath = os.path.join(class_dir, filename)
                    self.samples.append((filepath, class_idx))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        filepath, label = self.samples[idx]
        
        # Load image
        image = Image.open(filepath).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_class_names(self):
        """Return list of class names."""
        return self.classes
    
    def get_num_classes(self):
        """Return number of classes."""
        return len(self.classes)


def get_imagenets50_transforms(input_size=224, is_training=True):
    """Get standard transforms for ImageNet50 dataset."""
    
    if is_training:
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(input_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    return transform


def create_imagenets50_dataloaders(data_dir, batch_size=32, input_size=224, num_workers=4):
    """Create train and validation dataloaders for ImageNet50."""
    
    # Create transforms
    train_transform = get_imagenets50_transforms(input_size, is_training=True)
    val_transform = get_imagenets50_transforms(input_size, is_training=False)
    
    # Create datasets
    train_dataset = ImageNets50Dataset(data_dir, split='train', transform=train_transform)
    val_dataset = ImageNets50Dataset(data_dir, split='validation', transform=val_transform)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, train_dataset.get_num_classes()


def get_class_info(data_dir):
    """Get class information for ImageNet50 dataset."""
    train_dataset = ImageNets50Dataset(data_dir, split='train')
    return {
        'classes': train_dataset.get_class_names(),
        'num_classes': train_dataset.get_num_classes(),
        'class_to_idx': train_dataset.class_to_idx
    }


if __name__ == "__main__":
    # Test the dataset
    data_dir = "../data/imagenets50"
    
    # Create dataset
    dataset = ImageNets50Dataset(data_dir, split='train')
    print(f"Dataset size: {len(dataset)}")
    print(f"Number of classes: {dataset.get_num_classes()}")
    print(f"Classes: {dataset.get_class_names()[:5]}...")  # Show first 5 classes
    
    # Test dataloader
    train_loader, val_loader, num_classes = create_imagenets50_dataloaders(
        data_dir, batch_size=4, num_workers=0
    )
    
    # Test loading a batch
    for images, labels in train_loader:
        print(f"Batch shape: {images.shape}")
        print(f"Labels: {labels}")
        break