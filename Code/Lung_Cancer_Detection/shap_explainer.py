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
from matplotlib.colors import LinearSegmentedColormap
from skimage.segmentation import slic

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

    def _get_color_map(self):
        """Defines the custom Red/Green colormap."""
        colors = []
        for l in np.linspace(1, 0, 100): colors.append((245/255, 39/255, 87/255, l)) # Rouge
        for l in np.linspace(0, 1, 100): colors.append((24/255, 196/255, 93/255, l)) # Vert
        return LinearSegmentedColormap.from_list("shap", colors)
    
    def _fill_segmentation(self, values, segmentation):
        """Fills the segments with the calculated SHAP values."""
        out = np.zeros(segmentation.shape)
        for i in range(len(values)):
            out[segmentation == i] = values[i]
        return out
    
    def explain(self, image_path, nsamples=100, n_segments=50):
        """
        Run KernelSHAP on the segmented image.
        """
        # Load and preprocess image
        if isinstance(image_path, str):
            img_pil = Image.open(image_path).convert('RGB')
        else:
            img_pil = image_path.convert('RGB')
        
        img_pil = img_pil.resize((224, 224))
        img_array = np.array(img_pil).astype('float32') / 255.0

        # Segmentation (Super-pixels)
        segments_slic = slic(img_array, n_segments=n_segments, compactness=10, sigma=1)
        
        # Function of prediction for KernelExplainer
        def predict_fn(z):
            mask_value = img_array.mean()
            tensors = []
            for i in range(z.shape[0]):
                temp_img = img_array.copy()
                for j in range(z.shape[1]):
                    if z[i, j] == 0:
                        temp_img[segments_slic == j] = mask_value
                
                temp_pil = Image.fromarray((temp_img * 255).astype(np.uint8))
                tensors.append(self.transform(temp_pil))
            
            batch_tensor = torch.stack(tensors).to(self.device)
            with torch.no_grad():
                outputs = self.model(batch_tensor)
                return torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()

        # Compute SHAP values
        explainer = shap.KernelExplainer(predict_fn, np.zeros((1, n_segments)))
        shap_values = explainer.shap_values(np.ones((1, n_segments)), nsamples=nsamples)

        preds = predict_fn(np.ones((1, n_segments)))
        top_pred_idx = np.argmax(preds[0])
        
        return {
            'shap_values': shap_values,
            'segments': segments_slic,
            'img_array': img_array,
            'predicted_class_idx': top_pred_idx,
            'confidence': float(np.max(preds[0]))
        }
    
    def visualize(self, result):
        """
        Generates the Matplotlib figure with the SHAP segments overlay.
        """
        img_array = result['img_array']
        segments_slic = result['segments']
        shap_values = result['shap_values']
        pred_idx = result['predicted_class_idx']
        
        cm = self._get_color_map()    
        m = self._fill_segmentation(shap_values[pred_idx][0], segments_slic)
        
        fig, ax = plt.subplots(figsize=(8, 8))
        max_val = np.max(np.abs(m))
        
        ax.imshow(img_array)
        im = ax.imshow(m, cmap=cm, vmin=-max_val, vmax=max_val, alpha=0.6)
        ax.axis('off')
        
        plt.colorbar(im, ax=ax, label="Importance SHAP")
        
        return fig
    
def apply_shap_image(model, image_path, transform, classes, device='cpu'):
    """
    Fonction bridge for new_app.py
    """
    explainer = ImageSHAPExplainer(model, transform, classes, device)
    result = explainer.explain(image_path, nsamples=100)
    fig = explainer.visualize(result)
    
    return {
        'fig': fig,
        'predicted_class': classes[result['predicted_class_idx']],
        'confidence': result['confidence']
    }

