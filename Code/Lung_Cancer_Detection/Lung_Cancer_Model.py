"""
Lung Cancer Detection - Image Classification Models
Supports AlexNet, DenseNet with ImageNet weights, and TorchXRayVision medical models
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import numpy as np

# Try importing TorchXRayVision for medical-specific models
try:
    import torchxrayvision as xrv
    XRV_AVAILABLE = True
except ImportError:
    XRV_AVAILABLE = False
    
class LungCancerClassifier:
    def __init__(self, model_name='densenet', device='cpu', use_medical_weights=True):
        """
        Initialize lung cancer classifier
        Args:
            model_name: 'alexnet', 'densenet', or 'xrv-densenet' (medical-specific)
            device: 'cpu' or 'cuda'
            use_medical_weights: If True and XRV available, use medical pretrained weights
        """
        self.device = device
        self.model_name = model_name
        self.use_medical = use_medical_weights and XRV_AVAILABLE and model_name in ['densenet', 'xrv-densenet']
        
        if self.use_medical:
            self.model = self._load_xrv_model()
        else:
            self.model = self._load_model(model_name)
        
        self.model.to(self.device)
        self.model.eval()
        
        # Image preprocessing
        if self.use_medical:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x * 2048.0 - 1024.0),  # Scale [0,1] to [-1024, 1024]
            ])
        else:
            # ImageNet preprocessing
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
        
        # Class labels (binary classification)
        self.classes = ['No Finding', 'Malignant']
    
    def _load_xrv_model(self):
        """Load TorchXRayVision medical pretrained model"""
        xrv_base = xrv.models.DenseNet(weights="densenet121-res224-all")
        
        # Create a wrapper that uses XRV features but outputs binary classification
        class XRVBinaryWrapper(nn.Module):
            def __init__(self, xrv_model):
                super().__init__()
                self.features = xrv_model.features
                # Get feature dimension from the XRV model
                num_features = xrv_model.classifier.in_features
                # New binary classifier head
                self.classifier = nn.Linear(num_features, 2)
                
            def forward(self, x):
                features = self.features(x)
                features = torch.nn.functional.relu(features, inplace=True)
                features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
                features = torch.flatten(features, 1)
                out = self.classifier(features)
                return out
        
        wrapped_model = XRVBinaryWrapper(xrv_base)
        return wrapped_model
    
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
        
        # Apply transforms
        image_tensor = self.transform(image)
        
        # For XRV models, ensure single channel
        if self.use_medical and image_tensor.shape[0] != 1:
            # If RGB, convert to grayscale by averaging channels
            image_tensor = image_tensor.mean(dim=0, keepdim=True)
        
        image_tensor = image_tensor.unsqueeze(0)
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
        if self.use_medical or self.model_name == 'densenet' or self.model_name == 'xrv-densenet':
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
def load_lung_cancer_model(model_name='densenet', device='cpu', use_medical_weights=True):
    """
    Convenience function to load model
    Args:
        model_name: 'alexnet', 'densenet', or 'xrv-densenet'
        device: 'cpu' or 'cuda'
        use_medical_weights: Use TorchXRayVision medical weights if available
    Returns:
        Initialized classifier
    """
    return LungCancerClassifier(model_name=model_name, device=device, use_medical_weights=use_medical_weights)