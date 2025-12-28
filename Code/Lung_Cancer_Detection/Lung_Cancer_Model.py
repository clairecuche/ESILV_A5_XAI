"""
Lung Cancer Detection - Image Classification Models
Supports AlexNet and DenseNet with pre-trained weights
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import numpy as np

class LungCancerClassifier:
    def __init__(self, model_name='densenet', device='cpu'):
        """
        Initialize lung cancer classifier
        Args:
            model_name: 'alexnet' or 'densenet'
            device: 'cpu' or 'cuda'
        """
        self.device = device
        self.model_name = model_name
        self.model = self._load_model(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Class labels (binary classification)
        self.classes = ['No Finding', 'Malignant']
    
    def _load_model(self, model_name):
        """Load pre-trained model and adapt for binary classification"""
        if model_name == 'alexnet':
            model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
            # Modify classifier for binary classification
            num_features = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(num_features, 2)
            
        elif model_name == 'densenet':
            model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
            # Modify classifier for binary classification
            num_features = model.classifier.in_features
            model.classifier = nn.Linear(num_features, 2)
        else:
            raise ValueError(f"Model {model_name} not supported")
        
        return model
    
    def preprocess_image(self, image_path):
        """
        Preprocess image for model input
        Args:
            image_path: Path to image file or PIL Image
        Returns:
            Preprocessed tensor
        """
        if isinstance(image_path, str):
            image = Image.open(image_path).convert('RGB')
        else:
            image = image_path.convert('RGB')
        
        image_tensor = self.transform(image).unsqueeze(0)
        return image_tensor.to(self.device)
    
    def predict(self, image_path):
        """
        Predict class for input image
        Args:
            image_path: Path to image or PIL Image
        Returns:
            dict with prediction, confidence, and probabilities
        """
        image_tensor = self.preprocess_image(image_path)
        
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        pred_class = self.classes[predicted.item()]
        conf_score = confidence.item()
        probs = probabilities.cpu().numpy()[0]
        
        return {
            'prediction': pred_class,
            'confidence': conf_score,
            'probabilities': {
                self.classes[0]: probs[0],
                self.classes[1]: probs[1]
            },
            'raw_output': outputs.cpu().numpy()[0]
        }
    
    def get_feature_maps(self, image_path):
        """
        Extract feature maps for Grad-CAM
        Args:
            image_path: Path to image
        Returns:
            Feature maps and gradients
        """
        image_tensor = self.preprocess_image(image_path)
        image_tensor.requires_grad = True
        
        # Get activations from last conv layer
        activations = []
        gradients = []
        
        def forward_hook(module, input, output):
            activations.append(output)
        
        def backward_hook(module, grad_input, grad_output):
            gradients.append(grad_output[0])
        
        # Register hooks based on model architecture
        if self.model_name == 'densenet':
            target_layer = self.model.features[-1]
        else:  # alexnet
            target_layer = self.model.features[-1]
        
        forward_handle = target_layer.register_forward_hook(forward_hook)
        backward_handle = target_layer.register_full_backward_hook(backward_hook)
        
        # Forward pass
        output = self.model(image_tensor)
        
        # Backward pass
        self.model.zero_grad()
        pred_class = output.argmax(dim=1)
        output[0, pred_class].backward()
        
        # Remove hooks
        forward_handle.remove()
        backward_handle.remove()
        
        return activations[0], gradients[0], pred_class.item()


# Utility function for model loading
def load_lung_cancer_model(model_name='densenet', device='cpu'):
    """
    Convenience function to load model
    Args:
        model_name: 'alexnet' or 'densenet'
        device: 'cpu' or 'cuda'
    Returns:
        Initialized classifier
    """
    return LungCancerClassifier(model_name=model_name, device=device)