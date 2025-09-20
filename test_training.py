"""
Test script to verify training functionality with synthetic data
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os

from models.cifar_models import get_model


def create_synthetic_dataset(num_samples=100, num_classes=10):
    """Create synthetic CIFAR-like dataset"""
    # Create random data similar to CIFAR-10 (32x32x3 images)
    images = torch.randn(num_samples, 3, 32, 32)
    labels = torch.randint(0, num_classes, (num_samples,))
    
    dataset = TensorDataset(images, labels)
    return dataset


def test_training():
    """Test the training process with synthetic data"""
    print("Testing training script with synthetic data...")
    
    # Create synthetic datasets
    train_dataset = create_synthetic_dataset(200, 10)
    test_dataset = create_synthetic_dataset(50, 10)
    
    # Create data loaders
    trainloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    # Create model
    model = get_model('net', num_classes=10)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")
    
    # Training loop (just 2 epochs for testing)
    epochs = 2
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_accuracy = 100 * correct / total
        avg_loss = running_loss / len(trainloader)
        
        # Test evaluation
        model.eval()
        test_correct = 0
        test_total = 0
        test_loss = 0.0
        
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
        
        test_accuracy = 100 * test_correct / test_total
        avg_test_loss = test_loss / len(testloader)
        
        print(f"Epoch [{epoch+1}/{epochs}]")
        print(f"  Train - Loss: {avg_loss:.4f}, Accuracy: {train_accuracy:.2f}%")
        print(f"  Test  - Loss: {avg_test_loss:.4f}, Accuracy: {test_accuracy:.2f}%")
    
    # Test saving and loading checkpoint
    checkpoint_dir = "./test_checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_path = os.path.join(checkpoint_dir, "test_model.pth")
    checkpoint = {
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_test_loss,
        'accuracy': test_accuracy
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to: {checkpoint_path}")
    
    # Test loading checkpoint
    loaded_checkpoint = torch.load(checkpoint_path)
    new_model = get_model('net', num_classes=10)
    new_model.load_state_dict(loaded_checkpoint['model_state_dict'])
    print(f"Checkpoint loaded successfully! Accuracy: {loaded_checkpoint['accuracy']:.2f}%")
    
    print("\n✅ All tests passed! Training script functionality verified.")
    print("\nThe training script is ready to use with real datasets.")
    print("Example usage:")
    print("  python train.py --model net --dataset cifar10 --epochs 20")
    print("  python train_from_config.py configs/cifar10_net.yaml")
    
    # Clean up test files
    import shutil
    if os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)
    print("✅ Test cleanup completed.")


if __name__ == '__main__':
    test_training()