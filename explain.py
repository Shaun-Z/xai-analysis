#!/usr/bin/env python3
"""
Explainable AI script for analyzing trained models using Captum.
Supports various attribution methods for model interpretability.
"""

import argparse
import importlib
import sys
import json
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# Captum imports
from captum.attr import (
    IntegratedGradients,
    Saliency,
    GradientShap,
    DeepLift,
    DeepLiftShap,
    GuidedGradCam,
    LayerGradCam,
    Occlusion,
    LRP,
    InputXGradient,
    GuidedBackprop,
    Deconvolution
)

# Add project root to path for imports
project_root = Path(__file__).parent
sys.path.append(str(project_root))


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Explain XAI models using Captum')
    
    # Model and dataset selection
    parser.add_argument('--model', type=str, required=True,
                       help='Model name (Python module in models/ folder)')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset name (Python module in datasets/ folder)')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    
    # Explanation methods
    parser.add_argument('--methods', type=str, nargs='+', 
                       default=['integrated_gradients', 'saliency', 'gradcam'],
                       choices=[
                           'integrated_gradients', 'saliency', 'gradient_shap',
                           'deeplift', 'deeplift_shap', 'gradcam', 'guided_gradcam',
                           'occlusion', 'lrp', 'input_x_gradient', 'guided_backprop',
                           'deconvolution'
                       ],
                       help='Attribution methods to use')
    
    # Sample selection
    parser.add_argument('--num-samples', type=int, default=10,
                       help='Number of samples to explain')
    parser.add_argument('--sample-indices', type=int, nargs='+', default=None,
                       help='Specific sample indices to explain')
    parser.add_argument('--target-classes', type=int, nargs='+', default=None,
                       help='Specific target classes to focus on')
    
    # Attribution parameters
    parser.add_argument('--n-steps', type=int, default=50,
                       help='Number of steps for Integrated Gradients')
    parser.add_argument('--noise-tunnel', action='store_true',
                       help='Use noise tunnel for more robust attributions')
    parser.add_argument('--baseline-type', type=str, default='zero',
                       choices=['zero', 'random', 'gaussian'],
                       help='Baseline type for attribution methods')
    
    # Visualization settings
    parser.add_argument('--viz-method', type=str, default='heat_map',
                       choices=['heat_map', 'blended_heat_map', 'original_image', 'masked_image'],
                       help='Visualization method')
    parser.add_argument('--cmap', type=str, default='RdYlBu_r',
                       help='Colormap for visualizations')
    parser.add_argument('--alpha-overlay', type=float, default=0.4,
                       help='Alpha for overlay visualizations')
    
    # Output settings
    parser.add_argument('--results-dir', type=str, default='results',
                       help='Directory to save results')
    parser.add_argument('--experiment-name', type=str, default=None,
                       help='Experiment name for organization')
    parser.add_argument('--save-format', type=str, default='png',
                       choices=['png', 'jpg', 'pdf'],
                       help='Image save format')
    
    # Hardware and misc
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda, mps)')
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Batch size for explanation (usually 1)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    return parser.parse_args()


def get_device(device_arg):
    """Determine the best available device."""
    if device_arg == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
    else:
        return torch.device(device_arg)


def set_seed(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import numpy as np
    import random
    np.random.seed(seed)
    random.seed(seed)


def load_model(model_name, checkpoint_path, device):
    """Load model from checkpoint."""
    try:
        # Load model architecture
        model_module = importlib.import_module(f'models.{model_name}')
        if hasattr(model_module, 'create_model'):
            model = model_module.create_model()
        elif hasattr(model_module, 'Model'):
            model = model_module.Model()
        elif hasattr(model_module, 'Net'):
            model = model_module.Net()
        else:
            # Try to find the first class that inherits from nn.Module
            for attr_name in dir(model_module):
                attr = getattr(model_module, attr_name)
                if (isinstance(attr, type) and 
                    issubclass(attr, nn.Module) and 
                    attr != nn.Module):
                    model = attr()
                    break
            else:
                raise ValueError(f"No suitable model class found in {model_name}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        
        print(f"Loaded model from epoch {checkpoint['epoch']} with val_loss: {checkpoint['loss']:.4f}")
        return model
        
    except ImportError:
        raise ImportError(f"Could not import model '{model_name}' from models/{model_name}.py")


def load_dataset(dataset_name, batch_size=1, num_workers=0):
    """Load dataset for explanation."""
    try:
        dataset_module = importlib.import_module(f'datasets.{dataset_name}')
        
        # Get test loader for explanation
        if hasattr(dataset_module, 'get_dataloaders'):
            _, val_loader = dataset_module.get_dataloaders(
                batch_size=batch_size, num_workers=num_workers
            )
        elif hasattr(dataset_module, 'create_dataloaders'):
            _, val_loader = dataset_module.create_dataloaders(
                batch_size=batch_size, num_workers=num_workers
            )
        else:
            raise ValueError(f"No suitable dataloader function found in {dataset_name}")
        
        return val_loader
        
    except ImportError:
        raise ImportError(f"Could not import dataset '{dataset_name}' from datasets/{dataset_name}.py")


def get_baseline(input_tensor, baseline_type='zero'):
    """Generate baseline for attribution methods."""
    if baseline_type == 'zero':
        return torch.zeros_like(input_tensor)
    elif baseline_type == 'random':
        return torch.rand_like(input_tensor)
    elif baseline_type == 'gaussian':
        return torch.randn_like(input_tensor) * 0.1
    else:
        raise ValueError(f"Unknown baseline type: {baseline_type}")


def create_attribution_method(method_name, model, device):
    """Create attribution method instance."""
    if method_name == 'integrated_gradients':
        return IntegratedGradients(model)
    elif method_name == 'saliency':
        return Saliency(model)
    elif method_name == 'gradient_shap':
        return GradientShap(model)
    elif method_name == 'deeplift':
        return DeepLift(model)
    elif method_name == 'deeplift_shap':
        return DeepLiftShap(model)
    elif method_name == 'gradcam':
        # For GradCAM, we need to specify the target layer
        # This assumes the model has a 'features' attribute (common in CNNs)
        if hasattr(model, 'features'):
            target_layer = model.features[-1]  # Last conv layer
        else:
            # Try to find the last convolutional layer
            conv_layers = [module for module in model.modules() if isinstance(module, nn.Conv2d)]
            if conv_layers:
                target_layer = conv_layers[-1]
            else:
                raise ValueError("Could not find convolutional layer for GradCAM")
        return LayerGradCam(model, target_layer)
    elif method_name == 'guided_gradcam':
        if hasattr(model, 'features'):
            target_layer = model.features[-1]
        else:
            conv_layers = [module for module in model.modules() if isinstance(module, nn.Conv2d)]
            if conv_layers:
                target_layer = conv_layers[-1]
            else:
                raise ValueError("Could not find convolutional layer for Guided GradCAM")
        return GuidedGradCam(model, target_layer)
    elif method_name == 'occlusion':
        return Occlusion(model)
    elif method_name == 'lrp':
        return LRP(model)
    elif method_name == 'input_x_gradient':
        return InputXGradient(model)
    elif method_name == 'guided_backprop':
        return GuidedBackprop(model)
    elif method_name == 'deconvolution':
        return Deconvolution(model)
    else:
        raise ValueError(f"Unknown attribution method: {method_name}")


def compute_attribution(method, input_tensor, target, baseline=None, n_steps=50):
    """Compute attribution using the specified method."""
    method_name = method.__class__.__name__.lower()
    
    try:
        if 'integratedgradients' in method_name:
            attribution = method.attribute(
                input_tensor, 
                baselines=baseline, 
                target=target, 
                n_steps=n_steps
            )
        elif 'gradcam' in method_name or 'layergradcam' in method_name:
            attribution = method.attribute(input_tensor, target=target)
            # Interpolate to input size if needed
            if attribution.shape != input_tensor.shape:
                attribution = F.interpolate(
                    attribution, 
                    size=input_tensor.shape[-2:], 
                    mode='bilinear', 
                    align_corners=False
                )
        elif 'occlusion' in method_name:
            attribution = method.attribute(
                input_tensor,
                target=target,
                sliding_window_shapes=(3, 8, 8),  # (channels, height, width)
                strides=(3, 4, 4)
            )
        elif 'gradientshap' in method_name:
            # Generate random baselines for GradientSHAP
            rand_baseline = torch.randn_like(input_tensor) * 0.1
            attribution = method.attribute(
                input_tensor,
                baselines=rand_baseline,
                target=target,
                n_samples=50
            )
        else:
            # For methods that don't need special parameters
            if baseline is not None and hasattr(method, '__call__'):
                # Check if method accepts baselines
                try:
                    attribution = method.attribute(
                        input_tensor,
                        baselines=baseline,
                        target=target
                    )
                except TypeError:
                    # Method doesn't accept baselines
                    attribution = method.attribute(input_tensor, target=target)
            else:
                attribution = method.attribute(input_tensor, target=target)
        
        return attribution
        
    except Exception as e:
        print(f"Error computing attribution for {method_name}: {e}")
        return None


def save_visualization(attribution, original_image, predicted_class, true_class, method_name, 
                      sample_idx, save_path, viz_method='heat_map', cmap='RdYlBu_r', alpha_overlay=0.4):
    """Save attribution visualization."""
    try:
        # Convert tensors to numpy and handle dimensions
        if isinstance(attribution, torch.Tensor):
            attr_np = attribution.cpu().detach().numpy()
        else:
            attr_np = attribution
            
        if isinstance(original_image, torch.Tensor):
            img_np = original_image.cpu().detach().numpy()
        else:
            img_np = original_image
        
        # Handle batch dimension
        if len(attr_np.shape) == 4:
            attr_np = attr_np[0]
        if len(img_np.shape) == 4:
            img_np = img_np[0]
        
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        if img_np.shape[0] == 3:  # RGB
            img_display = np.transpose(img_np, (1, 2, 0))
            # Normalize to [0, 1] if needed
            if img_display.max() > 1.0:
                img_display = (img_display - img_display.min()) / (img_display.max() - img_display.min())
        else:
            img_display = img_np[0]  # Grayscale
        
        axes[0].imshow(img_display, cmap='gray' if len(img_display.shape) == 2 else None)
        axes[0].set_title(f'Original\nTrue: {true_class}, Pred: {predicted_class}')
        axes[0].axis('off')
        
        # Attribution heatmap
        if len(attr_np.shape) == 3:  # Multi-channel attribution
            attr_display = np.mean(np.abs(attr_np), axis=0)
        else:
            attr_display = attr_np
        
        im1 = axes[1].imshow(attr_display, cmap=cmap)
        axes[1].set_title(f'{method_name.replace("_", " ").title()} Attribution')
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        
        # Overlay visualization
        if viz_method == 'blended_heat_map' and len(img_display.shape) == 3:
            # Create blended visualization
            attr_norm = (attr_display - attr_display.min()) / (attr_display.max() - attr_display.min())
            
            # Create colored heatmap
            cmap_obj = plt.cm.get_cmap(cmap)
            colored_attr = cmap_obj(attr_norm)[:, :, :3]  # Remove alpha channel
            
            # Blend with original image
            blended = alpha_overlay * colored_attr + (1 - alpha_overlay) * img_display
            axes[2].imshow(blended)
        else:
            # Simple overlay
            axes[2].imshow(img_display, cmap='gray' if len(img_display.shape) == 2 else None)
            im2 = axes[2].imshow(attr_display, cmap=cmap, alpha=alpha_overlay)
            plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
        
        axes[2].set_title('Overlay')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved visualization: {save_path}")
        
    except Exception as e:
        print(f"Error saving visualization: {e}")


def generate_report(results, save_path):
    """Generate a comprehensive report of explanations."""
    report = {
        'timestamp': datetime.now().isoformat(),
        'summary': {
            'total_samples': len(results),
            'methods_used': list(set([r['method'] for r in results])),
            'classes_analyzed': list(set([r['true_class'] for r in results]))
        },
        'results': results
    }
    
    with open(save_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Generated report: {save_path}")


def main():
    args = parse_args()
    
    # Set up experiment name
    if args.experiment_name is None:
        date_str = datetime.now().strftime("%Y%m%d_%H%M")
        args.experiment_name = f"explain_{args.model}_{date_str}"
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Determine device
    device = get_device(args.device)
    print(f"Using device: {device}")
    
    # Create results directories
    results_base = Path(args.results_dir) / args.experiment_name
    attr_dir = results_base / 'attributions'
    viz_dir = results_base / 'visualizations'
    report_dir = results_base / 'reports'
    
    for dir_path in [attr_dir, viz_dir, report_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load model
        print(f"Loading model: {args.model}")
        model = load_model(args.model, args.checkpoint, device)
        
        # Load dataset
        print(f"Loading dataset: {args.dataset}")
        val_loader = load_dataset(args.dataset, batch_size=args.batch_size)
        
        # Get class names if available
        class_names = getattr(val_loader.dataset, 'classes', None)
        if class_names is None:
            # For CIFAR-10 default classes
            class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                          'dog', 'frog', 'horse', 'ship', 'truck']
        
        print(f"Starting explanation for {args.num_samples} samples...")
        print(f"Methods: {args.methods}")
        
        # Collect samples for explanation
        samples_data = []
        sample_count = 0
        
        for batch_idx, (data, target) in enumerate(val_loader):
            if sample_count >= args.num_samples:
                break
                
            data, target = data.to(device), target.to(device)
            
            # Check if specific indices are requested
            if args.sample_indices and batch_idx not in args.sample_indices:
                continue
            
            # Get model prediction
            with torch.no_grad():
                output = model(data)
                pred_class = output.argmax(dim=1).item()
                confidence = F.softmax(output, dim=1).max().item()
            
            # Check if specific target classes are requested
            if args.target_classes and target.item() not in args.target_classes:
                continue
            
            samples_data.append({
                'data': data,
                'target': target.item(),
                'prediction': pred_class,
                'confidence': confidence,
                'sample_idx': batch_idx
            })
            
            sample_count += 1
        
        print(f"Collected {len(samples_data)} samples for explanation")
        
        # Results storage
        all_results = []
        
        # Process each sample
        for sample_info in samples_data:
            data = sample_info['data']
            true_class = sample_info['target']
            pred_class = sample_info['prediction']
            sample_idx = sample_info['sample_idx']
            
            print(f"\nProcessing sample {sample_idx}: True={class_names[true_class]}, "
                  f"Pred={class_names[pred_class]} ({sample_info['confidence']:.3f})")
            
            # Apply each attribution method
            for method_name in args.methods:
                print(f"  Computing {method_name}...")
                
                try:
                    # Create attribution method
                    attribution_method = create_attribution_method(method_name, model, device)
                    
                    # Generate baseline if needed
                    baseline = get_baseline(data, args.baseline_type)
                    
                    # Compute attribution
                    attribution = compute_attribution(
                        attribution_method, 
                        data, 
                        target=pred_class,  # Explain the predicted class
                        baseline=baseline,
                        n_steps=args.n_steps
                    )
                    
                    if attribution is not None:
                        # Save attribution tensor
                        attr_path = attr_dir / f"sample_{sample_idx}_{method_name}_attribution.pt"
                        torch.save(attribution, attr_path)
                        
                        # Create and save visualization
                        viz_path = viz_dir / f"sample_{sample_idx}_{method_name}.{args.save_format}"
                        save_visualization(
                            attribution, 
                            data[0],  # Remove batch dimension
                            class_names[pred_class],
                            class_names[true_class],
                            method_name,
                            sample_idx,
                            viz_path,
                            args.viz_method,
                            args.cmap,
                            args.alpha_overlay
                        )
                        
                        # Store results
                        all_results.append({
                            'sample_idx': sample_idx,
                            'method': method_name,
                            'true_class': true_class,
                            'true_class_name': class_names[true_class],
                            'predicted_class': pred_class,
                            'predicted_class_name': class_names[pred_class],
                            'confidence': sample_info['confidence'],
                            'attribution_path': str(attr_path),
                            'visualization_path': str(viz_path)
                        })
                        
                        print(f"    ✓ Completed {method_name}")
                    else:
                        print(f"    ✗ Failed {method_name}")
                        
                except Exception as e:
                    print(f"    ✗ Error with {method_name}: {e}")
        
        # Generate comprehensive report
        report_path = report_dir / f"{args.experiment_name}_report.json"
        generate_report(all_results, report_path)
        
        print("\n🎉 Explanation completed!")
        print(f"Results saved in: {results_base}")
        print(f"Total explanations generated: {len(all_results)}")
        
        # Summary statistics
        methods_count = {}
        for result in all_results:
            method = result['method']
            methods_count[method] = methods_count.get(method, 0) + 1
        
        print("\nMethod summary:")
        for method, count in methods_count.items():
            print(f"  {method}: {count} explanations")
        
    except KeyboardInterrupt:
        print("\nExplanation interrupted by user")
    except Exception as e:
        print(f"\nError during explanation: {e}")
        raise


if __name__ == '__main__':
    main()