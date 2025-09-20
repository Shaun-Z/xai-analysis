"""
Implementation of various XAI methods using Captum library.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

# Captum imports
from captum.attr import (
    IntegratedGradients,
    GradientShap,
    DeepLift,
    DeepLiftShap,
    Saliency,
    InputXGradient,
    GuidedBackprop,
    GuidedGradCam,
    Occlusion,
    LRP,
    FeaturePermutation,
    Lime,
    KernelShap,
    LayerConductance,
    LayerActivation,
    LayerGradientXActivation
)
from captum.attr._utils.visualization import visualize_image_attr
import matplotlib.pyplot as plt


@dataclass
class XAIResult:
    """Container for XAI method results."""
    method_name: str
    attributions: torch.Tensor
    prediction: torch.Tensor
    target_class: int
    confidence: float
    processing_time: float
    memory_used_mb: Optional[float] = None


class CaptumXAI:
    """Wrapper class for various Captum XAI methods."""
    
    def __init__(self, model: nn.Module, device: str = 'cpu'):
        """
        Initialize XAI analyzer.
        
        Args:
            model: PyTorch model to analyze
            device: Device to run computations on
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        
        # Initialize attribution methods
        self.methods = self._initialize_methods()
    
    def _initialize_methods(self) -> Dict[str, Any]:
        """Initialize all available attribution methods."""
        methods = {
            'integrated_gradients': IntegratedGradients(self.model),
            'gradient_shap': GradientShap(self.model),
            'deep_lift': DeepLift(self.model),
            'deep_lift_shap': DeepLiftShap(self.model),
            'saliency': Saliency(self.model),
            'input_x_gradient': InputXGradient(self.model),
            'guided_backprop': GuidedBackprop(self.model),
            'occlusion': Occlusion(self.model),
            'lime': Lime(self.model),
            'kernel_shap': KernelShap(self.model),
        }
        
        # Add LRP if available (depends on model architecture)
        try:
            methods['lrp'] = LRP(self.model)
        except Exception:
            pass  # LRP not supported for this model
        
        return methods
    
    def get_prediction(self, input_tensor: torch.Tensor) -> Tuple[torch.Tensor, int, float]:
        """
        Get model prediction for input.
        
        Args:
            input_tensor: Input tensor
            
        Returns:
            Tuple of (output, predicted_class, confidence)
        """
        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            predicted_class = output.argmax(dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        return output, predicted_class, confidence
    
    def explain_integrated_gradients(self, 
                                   input_tensor: torch.Tensor,
                                   target: Optional[int] = None,
                                   n_steps: int = 50) -> XAIResult:
        """Apply Integrated Gradients method."""
        import time
        start_time = time.time()
        
        if target is None:
            _, target, confidence = self.get_prediction(input_tensor)
        else:
            _, _, confidence = self.get_prediction(input_tensor)
        
        ig = self.methods['integrated_gradients']
        attributions = ig.attribute(input_tensor, target=target, n_steps=n_steps)
        
        processing_time = time.time() - start_time
        
        return XAIResult(
            method_name='Integrated Gradients',
            attributions=attributions,
            prediction=self.model(input_tensor),
            target_class=target,
            confidence=confidence,
            processing_time=processing_time
        )
    
    def explain_gradient_shap(self,
                            input_tensor: torch.Tensor,
                            baseline_dist: torch.Tensor,
                            target: Optional[int] = None,
                            n_samples: int = 50) -> XAIResult:
        """Apply Gradient SHAP method."""
        import time
        start_time = time.time()
        
        if target is None:
            _, target, confidence = self.get_prediction(input_tensor)
        else:
            _, _, confidence = self.get_prediction(input_tensor)
        
        gs = self.methods['gradient_shap']
        attributions = gs.attribute(input_tensor, baselines=baseline_dist, 
                                  target=target, n_samples=n_samples)
        
        processing_time = time.time() - start_time
        
        return XAIResult(
            method_name='Gradient SHAP',
            attributions=attributions,
            prediction=self.model(input_tensor),
            target_class=target,
            confidence=confidence,
            processing_time=processing_time
        )
    
    def explain_saliency(self, 
                        input_tensor: torch.Tensor,
                        target: Optional[int] = None) -> XAIResult:
        """Apply Saliency method."""
        import time
        start_time = time.time()
        
        if target is None:
            _, target, confidence = self.get_prediction(input_tensor)
        else:
            _, _, confidence = self.get_prediction(input_tensor)
        
        saliency = self.methods['saliency']
        attributions = saliency.attribute(input_tensor, target=target)
        
        processing_time = time.time() - start_time
        
        return XAIResult(
            method_name='Saliency',
            attributions=attributions,
            prediction=self.model(input_tensor),
            target_class=target,
            confidence=confidence,
            processing_time=processing_time
        )
    
    def explain_deep_lift(self,
                         input_tensor: torch.Tensor,
                         baseline: Optional[torch.Tensor] = None,
                         target: Optional[int] = None) -> XAIResult:
        """Apply DeepLIFT method."""
        import time
        start_time = time.time()
        
        if target is None:
            _, target, confidence = self.get_prediction(input_tensor)
        else:
            _, _, confidence = self.get_prediction(input_tensor)
        
        if baseline is None:
            baseline = torch.zeros_like(input_tensor)
        
        dl = self.methods['deep_lift']
        attributions = dl.attribute(input_tensor, baselines=baseline, target=target)
        
        processing_time = time.time() - start_time
        
        return XAIResult(
            method_name='DeepLIFT',
            attributions=attributions,
            prediction=self.model(input_tensor),
            target_class=target,
            confidence=confidence,
            processing_time=processing_time
        )
    
    def explain_occlusion(self,
                         input_tensor: torch.Tensor,
                         target: Optional[int] = None,
                         sliding_window_shapes: Tuple[int, ...] = (3, 8, 8),
                         strides: Tuple[int, ...] = (3, 4, 4)) -> XAIResult:
        """Apply Occlusion method."""
        import time
        start_time = time.time()
        
        if target is None:
            _, target, confidence = self.get_prediction(input_tensor)
        else:
            _, _, confidence = self.get_prediction(input_tensor)
        
        occlusion = self.methods['occlusion']
        attributions = occlusion.attribute(input_tensor, 
                                         target=target,
                                         sliding_window_shapes=sliding_window_shapes,
                                         strides=strides)
        
        processing_time = time.time() - start_time
        
        return XAIResult(
            method_name='Occlusion',
            attributions=attributions,
            prediction=self.model(input_tensor),
            target_class=target,
            confidence=confidence,
            processing_time=processing_time
        )
    
    def explain_lime(self,
                    input_tensor: torch.Tensor,
                    target: Optional[int] = None,
                    n_samples: int = 500) -> XAIResult:
        """Apply LIME method."""
        import time
        start_time = time.time()
        
        if target is None:
            _, target, confidence = self.get_prediction(input_tensor)
        else:
            _, _, confidence = self.get_prediction(input_tensor)
        
        lime = self.methods['lime']
        attributions = lime.attribute(input_tensor, target=target, n_samples=n_samples)
        
        processing_time = time.time() - start_time
        
        return XAIResult(
            method_name='LIME',
            attributions=attributions,
            prediction=self.model(input_tensor),
            target_class=target,
            confidence=confidence,
            processing_time=processing_time
        )
    
    def explain_all_methods(self,
                           input_tensor: torch.Tensor,
                           target: Optional[int] = None,
                           quick_mode: bool = False) -> List[XAIResult]:
        """
        Apply all available XAI methods to the input.
        
        Args:
            input_tensor: Input tensor to explain
            target: Target class (if None, uses prediction)
            quick_mode: If True, uses faster parameters for time-intensive methods
            
        Returns:
            List of XAI results from all methods
        """
        results = []
        
        # Get baseline for methods that need it
        baseline = torch.zeros_like(input_tensor)
        baseline_dist = torch.randn(10, *input_tensor.shape[1:]).to(self.device)
        
        try:
            # Fast methods
            results.append(self.explain_saliency(input_tensor, target))
            results.append(self.explain_integrated_gradients(
                input_tensor, target, n_steps=25 if quick_mode else 50))
            results.append(self.explain_deep_lift(input_tensor, baseline, target))
            
            # Medium speed methods
            if not quick_mode:
                results.append(self.explain_gradient_shap(
                    input_tensor, baseline_dist, target, n_samples=25))
                
                # Slower methods
                results.append(self.explain_occlusion(
                    input_tensor, target, 
                    sliding_window_shapes=(3, 16, 16),
                    strides=(3, 8, 8)))
                
                results.append(self.explain_lime(input_tensor, target, n_samples=100))
        
        except Exception as e:
            print(f"Error in XAI method: {e}")
        
        return results
    
    def get_available_methods(self) -> List[str]:
        """Get list of available XAI methods."""
        return list(self.methods.keys())


def visualize_attributions(xai_result: XAIResult,
                         original_image: torch.Tensor,
                         save_path: Optional[str] = None) -> plt.Figure:
    """
    Visualize attribution results.
    
    Args:
        xai_result: XAI result to visualize
        original_image: Original input image
        save_path: Path to save the visualization
        
    Returns:
        Matplotlib figure
    """
    # Convert tensors to numpy
    if original_image.dim() == 4:
        original_image = original_image.squeeze(0)
    if xai_result.attributions.dim() == 4:
        attributions = xai_result.attributions.squeeze(0)
    else:
        attributions = xai_result.attributions
    
    # Create visualization
    fig, axes = visualize_image_attr(
        attributions.cpu().detach().numpy(),
        original_image.cpu().detach().numpy(),
        method='blended_heat_map',
        sign='all',
        show_colorbar=True,
        title=f'{xai_result.method_name} - Class: {xai_result.target_class} '
              f'(Conf: {xai_result.confidence:.3f})',
        use_pyplot=False
    )
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig