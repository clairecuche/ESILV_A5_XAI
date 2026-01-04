"""
SHAP (SHapley Additive exPlanations) for Images
Using DeepExplainer for neural networks
"""

import torch
import numpy as np
import shap
from PIL import Image
import matplotlib.pyplot as plt

class ImageSHAPExplainer:
    def __init__(self, model, transform, classes, device='cpu'):
        """
        Initialize SHAP explainer for images
        Args:
            model: PyTorch model
            transform: Image preprocessing transform
            classes: List of class names
            device: 'cpu' or 'cuda'
        """
        self.model = model
        self.transform = transform
        self.classes = classes
        self.device = device
        self.model.eval()
        
        # Create background dataset (using small set of images)
        self.background = None
    
    def set_background(self, background_images):
        """
        Set background dataset for SHAP
        Args:
            background_images: List of background images (PIL Images or paths)
        """
        background_tensors = []
        for img in background_images:
            if isinstance(img, str):
                img = Image.open(img).convert('RGB')
            img_tensor = self.transform(img).unsqueeze(0)
            background_tensors.append(img_tensor)
        
        self.background = torch.cat(background_tensors, dim=0).to(self.device)
    
    def explain(self, image_path, num_samples=100):
        """
        Generate SHAP explanation using GradientExplainer (plus stable)
        """
        # Load and preprocess image
        if isinstance(image_path, str):
            image = Image.open(image_path).convert('RGB')
        else:
            image = image_path.convert('RGB')
        
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Create background if not set
        if self.background is None:
            self.background = image_tensor
        
        # Utiliser GradientExplainer (plus compatible)
        explainer = shap.GradientExplainer(self.model, self.background)
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(image_tensor)
        
        # Get prediction
        with torch.no_grad():
            output = self.model(image_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0, pred_class].item()
        
        return {
            'shap_values': shap_values,
            'image_tensor': image_tensor.cpu().numpy(),
            'predicted_class': self.classes[pred_class],
            'confidence': confidence,
            'probabilities': {self.classes[i]: float(probs[0, i]) 
                            for i in range(len(self.classes))}
        }
    
    def visualize(self, shap_values, image_tensor, predicted_class_idx=1):
        """
        Create SHAP visualization
        Args:
            shap_values: SHAP values array
            image_tensor: Original image tensor
            predicted_class_idx: Index of class to visualize
        Returns:
            Visualization as PIL Image
        """
        # Convert to proper format for visualization
        # SHAP values shape: (num_classes, channels, height, width)
        if isinstance(shap_values, list):
            shap_values_class = shap_values[predicted_class_idx]
        else:
            shap_values_class = shap_values[predicted_class_idx]
        
        # Average across color channels for visualization
        shap_sum = np.sum(np.abs(shap_values_class), axis=0)
        
        # Normalize to [0, 1]
        shap_sum = (shap_sum - shap_sum.min()) / (shap_sum.max() - shap_sum.min() + 1e-8)
        
        # Create heatmap
        plt.figure(figsize=(8, 8))
        plt.imshow(shap_sum, cmap='hot', interpolation='nearest')
        plt.colorbar(label='SHAP Value Magnitude')
        plt.title(f'SHAP Explanation - Class: {predicted_class_idx}')
        plt.axis('off')
        
        # Convert plot to image
        plt.tight_layout()
        plt.savefig('/tmp/shap_viz.png', bbox_inches='tight', dpi=150)
        plt.close()
        
        result = Image.open('/tmp/shap_viz.png')
        return result, shap_sum
    
    def visualize_overlay(self, image_path, shap_values, predicted_class_idx=1):
        """
        Create SHAP visualization overlaid on original image
        Args:
            image_path: Path to original image or PIL Image
            shap_values: SHAP values
            predicted_class_idx: Class index to visualize
        Returns:
            Overlaid visualization
        """
        # Load original image
        if isinstance(image_path, str):
            original_image = Image.open(image_path).convert('RGB')
        else:
            original_image = image_path.convert('RGB')
        
        original_image = original_image.resize((224, 224))
        original_array = np.array(original_image)
        
        # Get SHAP heatmap
        if isinstance(shap_values, list):
            shap_values_class = shap_values[predicted_class_idx]
        else:
            shap_values_class = shap_values[predicted_class_idx]
        
        shap_sum = np.sum(np.abs(shap_values_class), axis=0)
        shap_sum = (shap_sum - shap_sum.min()) / (shap_sum.max() - shap_sum.min() + 1e-8)
        
        # Create colored heatmap
        import cv2
        heatmap = cv2.applyColorMap(np.uint8(255 * shap_sum), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Overlay
        superimposed = cv2.addWeighted(original_array, 0.6, heatmap, 0.4, 0)
        result = Image.fromarray(superimposed)
        
        return result


def apply_shap_image(model, image_path, transform, classes, device='cpu',
                    background_images=None):
    """
    Convenience function to apply SHAP to image
    Args:
        model: PyTorch model
        image_path: Path to image or PIL Image
        transform: Preprocessing transform
        classes: List of class names
        device: 'cpu' or 'cuda'
        background_images: Optional background dataset
    Returns:
        SHAP explanation and visualizations
    """
    explainer = ImageSHAPExplainer(model, transform, classes, device)
    
    if background_images:
        explainer.set_background(background_images)
    
    result = explainer.explain(image_path)
    
    # Get predicted class index
    pred_class_idx = classes.index(result['predicted_class'])
    
    # Create visualizations
    viz_heatmap, shap_sum = explainer.visualize(
        result['shap_values'],
        result['image_tensor'],
        pred_class_idx
    )
        
    viz_overlay = explainer.visualize_overlay(
        image_path,
        result['shap_values'],
        pred_class_idx
    )
    
    return {
        'shap_values': result['shap_values'],
        'predicted_class': result['predicted_class'],
        'confidence': result['confidence'],
        'probabilities': result['probabilities'],
        'visualization_heatmap': viz_heatmap,
        'visualization_overlay': viz_overlay,
        'shap_sum': shap_sum
    }