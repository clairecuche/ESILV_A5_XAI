"""
LIME (Local Interpretable Model-agnostic Explanations) for Images
"""

import torch
import numpy as np
from PIL import Image
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt

class ImageLIMEExplainer:
    def __init__(self, model, transform, classes, device='cpu'):
        """
        Initialize LIME explainer for images
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
        
        # Initialize LIME image explainer
        self.explainer = lime_image.LimeImageExplainer()
    
    def predict_fn(self, images):
        """
        Prediction function for LIME
        Args:
            images: Batch of images as numpy arrays (N, H, W, C)
        Returns:
            Prediction probabilities
        """
        batch_predictions = []
        
        for img in images:
            # Convert numpy array to PIL Image
            pil_img = Image.fromarray(img.astype('uint8'))
            
            # Apply preprocessing
            img_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
            
            # Get prediction
            with torch.no_grad():
                output = self.model(img_tensor)
                probs = torch.nn.functional.softmax(output, dim=1)
                batch_predictions.append(probs.cpu().numpy()[0])
        
        return np.array(batch_predictions)
    
    def explain(self, image_path, top_labels=2, num_samples=1000, num_features=10):
        """
        Generate LIME explanation for image
        Args:
            image_path: Path to image or PIL Image
            top_labels: Number of top classes to explain
            num_samples: Number of perturbed samples
            num_features: Number of superpixels to highlight
        Returns:
            Explanation object and visualization
        """
        # Load and prepare image
        if isinstance(image_path, str):
            image = Image.open(image_path).convert('RGB')
        else:
            image = image_path.convert('RGB')
        
        image = image.resize((224, 224))
        image_array = np.array(image)
        
        # Generate explanation
        explanation = self.explainer.explain_instance(
            image_array,
            self.predict_fn,
            top_labels=top_labels,
            hide_color=0,
            num_samples=num_samples
        )
        
        return explanation, image_array
    
    def visualize(self, explanation, image_array, label_idx=1, positive_only=True, 
                  num_features=10, hide_rest=False):
        """
        Create visualization of LIME explanation
        Args:
            explanation: LIME explanation object
            image_array: Original image as numpy array
            label_idx: Class index to visualize
            positive_only: Show only positive contributions
            num_features: Number of features to show
            hide_rest: Hide non-important regions
        Returns:
            PIL Image with LIME visualization
        """
        # Get image and mask
        temp, mask = explanation.get_image_and_mask(
            label_idx,
            positive_only=positive_only,
            num_features=num_features,
            hide_rest=hide_rest
        )
        
        # Create boundary visualization
        img_boundry = mark_boundaries(temp / 255.0, mask)
        
        # Convert to PIL Image
        img_boundry = (img_boundry * 255).astype(np.uint8)
        result = Image.fromarray(img_boundry)
        
        return result, mask
    
    def get_top_features(self, explanation, label_idx):
        """
        Get top contributing features
        Args:
            explanation: LIME explanation object
            label_idx: Class index
        Returns:
            List of (superpixel_id, weight) tuples
        """
        return explanation.local_exp[label_idx]


def apply_lime_image(model, image_path, transform, classes, device='cpu', 
                     num_samples=500, num_features=10):
    """
    Convenience function to apply LIME to image
    Args:
        model: PyTorch model
        image_path: Path to image or PIL Image
        transform: Preprocessing transform
        classes: List of class names
        device: 'cpu' or 'cuda'
        num_samples: Number of LIME samples
        num_features: Number of superpixels to highlight
    Returns:
        Explanation and visualizations
    """
    explainer = ImageLIMEExplainer(model, transform, classes, device)
    explanation, image_array = explainer.explain(
        image_path, 
        num_samples=num_samples,
        num_features=num_features
    )
    
    # Get prediction
    pred_probs = explainer.predict_fn(np.array([image_array]))[0]
    pred_class = np.argmax(pred_probs)
    
    # Visualize for predicted class
    visualization, mask = explainer.visualize(
        explanation, 
        image_array, 
        label_idx=pred_class,
        num_features=num_features
    )
    
    return {
        'explanation': explanation,
        'visualization': visualization,
        'mask': mask,
        'predicted_class': classes[pred_class],
        'probabilities': {classes[i]: float(pred_probs[i]) for i in range(len(classes))}
    }