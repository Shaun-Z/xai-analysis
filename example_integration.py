"""
Example script showing how to integrate trained models with XAI analysis
This demonstrates how to load a trained model and use it with Captum for interpretability
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

from models.cifar_models import get_model
from datasets import get_dataset_loaders

# Import Captum for XAI analysis (similar to original notebook)
try:
    from captum.attr import IntegratedGradients, Saliency, DeepLift, NoiseTunnel
    from captum.attr import visualization as viz
    CAPTUM_AVAILABLE = True
except ImportError:
    print("Captum not available. Install with: pip install captum")
    CAPTUM_AVAILABLE = False


def load_trained_model(checkpoint_path, model_name, num_classes=10):
    """Load a trained model from checkpoint"""
    model = get_model(model_name, num_classes)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Loaded model from {checkpoint_path}")
    print(f"Training accuracy: {checkpoint.get('accuracy', 'N/A'):.2f}%")
    return model


def demonstrate_xai_analysis(model, testloader, device):
    """Demonstrate XAI analysis similar to the original notebook"""
    if not CAPTUM_AVAILABLE:
        print("Captum not available, skipping XAI analysis")
        return
    
    # Get CIFAR-10 class names
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    # Get a batch of test data
    dataiter = iter(testloader)
    images, labels = next(dataiter)
    images, labels = images.to(device), labels.to(device)
    
    # Make predictions
    with torch.no_grad():
        outputs = model(images)
        probabilities = F.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)
    
    # Select first image for analysis
    ind = 0
    input_img = images[ind].unsqueeze(0)
    input_img.requires_grad_()
    
    print(f'Predicted: {classes[predicted[ind]]}')
    print(f'Actual: {classes[labels[ind]]}')
    print(f'Confidence: {probabilities[ind][predicted[ind]].item():.4f}')
    
    # Initialize attribution methods
    integrated_gradients = IntegratedGradients(model)
    saliency = Saliency(model)
    deep_lift = DeepLift(model)
    noise_tunnel = NoiseTunnel(integrated_gradients)
    
    # Compute attributions
    target_class = predicted[ind].item()
    
    # Integrated Gradients
    attr_ig = integrated_gradients.attribute(input_img, target=target_class, n_steps=50)
    
    # Saliency
    attr_saliency = saliency.attribute(input_img, target=target_class)
    
    # DeepLift
    attr_dl = deep_lift.attribute(input_img, target=target_class)
    
    # Integrated Gradients with Noise Tunnel
    attr_ig_nt = noise_tunnel.attribute(input_img, nt_samples=10, nt_type='smoothgrad_sq', 
                                       target=target_class)
    
    # Convert for visualization
    original_image = np.transpose((input_img.squeeze().cpu().detach().numpy() / 2) + 0.5, (1, 2, 0))
    
    print("Visualizing attributions...")
    
    # Display original image
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 3, 1)
    plt.imshow(original_image)
    plt.title(f"Original Image\nPredicted: {classes[predicted[ind]]}")
    plt.axis('off')
    
    # Note: For full visualization, you would use captum's viz.visualize_image_attr
    # This is a simplified version showing the concept
    
    attributions = [
        (attr_saliency, "Saliency"),
        (attr_ig, "Integrated Gradients"), 
        (attr_dl, "DeepLift"),
        (attr_ig_nt, "IG + SmoothGrad")
    ]
    
    for i, (attr, name) in enumerate(attributions, 2):
        plt.subplot(2, 3, i)
        attr_np = np.transpose(attr.squeeze().cpu().detach().numpy(), (1, 2, 0))
        # Simple visualization - just show the magnitude
        attr_magnitude = np.mean(np.abs(attr_np), axis=2)
        plt.imshow(attr_magnitude, cmap='hot', alpha=0.7)
        plt.title(name)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('xai_analysis_example.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("XAI analysis visualization saved as 'xai_analysis_example.png'")


def main():
    """Main function demonstrating model loading and XAI analysis"""
    print("XAI Training Integration Example")
    print("=" * 40)
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Example 1: Create a fresh model for demonstration
    print("\n1. Creating and testing a fresh model...")
    model = get_model('net', num_classes=10)
    model.to(device)
    
    # Load test data
    print("Loading CIFAR-10 test data...")
    try:
        _, testloader, _, num_classes = get_dataset_loaders('cifar10', batch_size=4, num_workers=0)
        print("Dataset loaded successfully")
        
        # Test the model
        dataiter = iter(testloader)
        images, labels = next(dataiter)
        images = images.to(device)
        
        with torch.no_grad():
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
        
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        print(f"Sample predictions: {[classes[p] for p in predicted[:4]]}")
        
        # Demonstrate XAI analysis
        print("\n2. Demonstrating XAI analysis...")
        demonstrate_xai_analysis(model, testloader, device)
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("This might be due to network connectivity issues.")
        print("In a real environment, CIFAR-10 would be downloaded automatically.")
    
    print("\n" + "=" * 40)
    print("Example completed!")
    print("\nTo use with a trained model:")
    print("1. Train a model: python train.py --model net --dataset cifar10 --epochs 20")
    print("2. Load and analyze: model = load_trained_model('checkpoints/best_model.pth', 'net')")
    print("3. Use with original notebook by replacing the model creation section")


if __name__ == '__main__':
    main()