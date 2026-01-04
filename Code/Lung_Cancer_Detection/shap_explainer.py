"""
SHAP (SHapley Additive exPlanations) for Images
Using DeepExplainer for neural networks
"""

import torch
import numpy as np
import shap
from PIL import Image
import matplotlib.pyplot as plt
import cv2

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
    
    def explain(self, image_path):
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
            self.background = torch.zeros_like(image_tensor).to(self.device)
        
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
        Crée une visualisation SHAP superposée avec correction du type de données pour OpenCV
        """
        # 1. Préparation de l'image originale
        if isinstance(image_path, str):
            original_image = Image.open(image_path).convert('RGB')
        else:
            original_image = image_path.convert('RGB')
        
        original_image = original_image.resize((224, 224))
        original_array = np.array(original_image)
        
        # 2. Extraction et nettoyage des dimensions SHAP
        if isinstance(shap_values, list):
            shap_values_class = shap_values[predicted_class_idx]
        else:
            shap_values_class = shap_values[predicted_class_idx]

        # Si SHAP renvoie (1, 3, 224, 224), on retire la dimension de batch
        if len(shap_values_class.shape) == 4:
            shap_values_class = shap_values_class[0]

        # Conversion en numpy si c'est encore un tenseur PyTorch
        if torch.is_tensor(shap_values_class):
            shap_values_class = shap_values_class.detach().cpu().numpy()

        # 3. Création de la heatmap 2D
        # On somme les valeurs absolues des canaux (C, H, W) -> (H, W)
        shap_sum = np.sum(np.abs(shap_values_class), axis=0)
        
        # Normalisation entre 0 et 1
        shap_min, shap_max = shap_sum.min(), shap_sum.max()
        shap_norm = (shap_sum - shap_min) / (shap_max - shap_min + 1e-8)

        # On transforme les floats [0.0, 1.0] en entiers [0, 255] de type uint8
        heatmap_8bit = np.uint8(255 * shap_norm)
        
        # Application de la ColorMap
        heatmap_color = cv2.applyColorMap(heatmap_8bit, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
        
        # 4. Superposition (Overlay)
        superimposed = cv2.addWeighted(original_array, 0.6, heatmap_rgb, 0.4, 0)
        
        return Image.fromarray(superimposed)


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
            
    viz_overlay = explainer.visualize_overlay(
        image_path,
        result['shap_values'],
        pred_class_idx
    )
    
    return {
        'predicted_class': result['predicted_class'],
        'confidence': result['confidence'],
        'visualization_overlay': viz_overlay,
        'shap_values': result['shap_values']
    }