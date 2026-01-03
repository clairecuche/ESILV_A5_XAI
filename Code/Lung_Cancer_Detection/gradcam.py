"""
Grad-CAM Implementation using pytorch-grad-cam library
"""

import torch
import numpy as np
import cv2
from PIL import Image
from pytorch_grad_cam import GradCAM as GradCAMLib
from pytorch_grad_cam.utils.image import show_cam_on_image

class GradCAM:
    def __init__(self, model, target_layer=None):
        self.model = model
        self.model.eval()
        
        if target_layer is None:
            # Pour DenseNet, la dernière couche de features
            if hasattr(self.model, 'features'):
                target_layers = [self.model.features[-1]]
            else:
                raise ValueError("Cannot find target layer")
        else:
            target_layers = [target_layer]
        
        # Utiliser la bibliothèque grad-cam
        self.cam = GradCAMLib(model=self.model, target_layers=target_layers)
    
    def generate_cam(self, image_tensor, target_class=None):
        """Generate Grad-CAM heatmap"""
        # Convertir en numpy pour grad-cam
        input_tensor = image_tensor
        
        # Si target_class est None, utiliser la prédiction
        targets = None
        if target_class is not None:
            from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
            targets = [ClassifierOutputTarget(target_class)]
        
        # Générer CAM
        grayscale_cam = self.cam(input_tensor=input_tensor, targets=targets)
        
        # Retourner le premier (et seul) résultat
        return grayscale_cam[0]
    
    def visualize(self, image_path, image_tensor, save_path=None):
        """Create Grad-CAM visualization"""
        # Générer CAM
        cam = self.generate_cam(image_tensor)
        
        # Charger l'image originale
        if isinstance(image_path, str):
            original_image = Image.open(image_path).convert('RGB')
        else:
            original_image = image_path.convert('RGB')
        
        original_image = original_image.resize((224, 224))
        original_array = np.array(original_image) / 255.0  # Normaliser [0, 1]
        
        # Créer la visualisation avec overlay
        visualization = show_cam_on_image(original_array, cam, use_rgb=True)
        
        result = Image.fromarray(visualization)
        
        if save_path:
            result.save(save_path)
        
        return result, cam


def apply_gradcam(model, image_path, image_tensor, model_name='densenet'):
    """Apply Grad-CAM using pytorch-grad-cam library"""
    gradcam = GradCAM(model)
    visualization, heatmap = gradcam.visualize(image_path, image_tensor)
    return visualization, heatmap