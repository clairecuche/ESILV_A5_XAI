"""
Grad-CAM (Gradient-weighted Class Activation Mapping) Implementation
For image classification explainability
"""

import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F

class GradCAM:
    def __init__(self, model, target_layer=None):
        """
        Initialize Grad-CAM
        Args:
            model: PyTorch model
            target_layer: Layer to visualize (if None, uses last conv layer)
        """
        self.model = model
        self.model.eval()
        
        # Storage for activations and gradients
        self.activations = None
        self.gradients = None
        
        # Determine target layer automatically if not specified
        if target_layer is None:
            self.target_layer = self._get_last_conv_layer()
        else:
            self.target_layer = target_layer
        
        # Register hooks
        self.forward_handle = self.target_layer.register_forward_hook(self._forward_hook)
        self.backward_handle = self.target_layer.register_full_backward_hook(self._backward_hook)
    
    def _get_last_conv_layer(self):
        """Automatically find last convolutional layer"""
        # For DenseNet
        if hasattr(self.model, 'features'):
            return self.model.features[-1]
        # For AlexNet
        elif hasattr(self.model, 'features'):
            for layer in reversed(list(self.model.features)):
                if isinstance(layer, torch.nn.Conv2d):
                    return layer
        else:
            raise ValueError("Cannot automatically determine last conv layer")
    
    def _forward_hook(self, module, input, output):
        """Hook to capture activations during forward pass"""
        self.activations = output.detach()
    
    def _backward_hook(self, module, grad_input, grad_output):
        """Hook to capture gradients during backward pass"""
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, image_tensor, target_class=None):
        """
        Generate Grad-CAM heatmap
        Args:
            image_tensor: Preprocessed image tensor (1, C, H, W)
            target_class: Target class index (if None, uses predicted class)
        Returns:
            Grad-CAM heatmap as numpy array
        """
        # Forward pass
        image_tensor.requires_grad = True
        output = self.model(image_tensor)
        
        # Use predicted class if target not specified
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        output[0, target_class].backward()
        
        # Get activations and gradients
        activations = self.activations
        gradients = self.gradients
        
        # Global average pooling on gradients
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        
        # Weighted combination of activation maps
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        
        # Apply ReLU
        cam = F.relu(cam)
        
        # Normalize to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam
    
    def visualize(self, image_path, image_tensor, save_path=None):
        """
        Create Grad-CAM visualization overlaid on original image
        Args:
            image_path: Path to original image or PIL Image
            image_tensor: Preprocessed tensor
            save_path: Optional path to save visualization
        Returns:
            PIL Image with Grad-CAM overlay
        """
        # Generate CAM
        cam = self.generate_cam(image_tensor)
        
        # Load original image
        if isinstance(image_path, str):
            original_image = Image.open(image_path).convert('RGB')
        else:
            original_image = image_path.convert('RGB')
        
        original_image = original_image.resize((224, 224))
        
        # Resize CAM to match image size
        cam_resized = cv2.resize(cam, (224, 224))
        
        # Convert to heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Overlay on original image
        original_array = np.array(original_image)
        superimposed = cv2.addWeighted(original_array, 0.6, heatmap, 0.4, 0)
        
        result = Image.fromarray(superimposed)
        
        if save_path:
            result.save(save_path)
        
        return result, cam_resized
    
    def __del__(self):
        """Remove hooks when object is destroyed"""
        self.forward_handle.remove()
        self.backward_handle.remove()


def apply_gradcam(model, image_path, image_tensor, model_name='densenet'):
    """
    Convenience function to apply Grad-CAM
    Args:
        model: PyTorch model
        image_path: Path to image or PIL Image
        image_tensor: Preprocessed tensor
        model_name: Model architecture name
    Returns:
        Grad-CAM visualization and heatmap
    """
    gradcam = GradCAM(model)
    visualization, heatmap = gradcam.visualize(image_path, image_tensor)
    return visualization, heatmap