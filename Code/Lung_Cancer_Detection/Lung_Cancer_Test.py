"""
Example usage of Lung Cancer Detection with XAI
"""

import torch
from Lung_Cancer_Model import LungCancerClassifier
from gradcam import apply_gradcam
from lime_explainer import apply_lime_image
from shap_explainer import apply_shap_image

# Check for GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# ============================================
# 1. LOAD MODEL
# ============================================
print("\n1. Loading model...")
classifier = LungCancerClassifier(model_name='densenet', device=device)
print(f"Model loaded: {classifier.model_name}")

# ============================================
# 2. MAKE PREDICTION
# ============================================
print("\n2. Making prediction...")
#image_path = "../../img/Lung_Cancer/image3.png"  # Replace with your image
image_path = "C:\\Users\\benoi\\OneDrive - De Vinci\\A5 ESILV\\Explainability AI\\Projet\\ESILV_A5_XAI\\img\\Lung_Cancer\\image3.png"

result = classifier.predict(image_path)
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.4f}")
print(f"Probabilities: {result['probabilities']}")

# ============================================
# 3. APPLY GRAD-CAM
# ============================================
print("\n3. Generating Grad-CAM explanation...")
image_tensor = classifier.preprocess_image(image_path)
gradcam_viz, gradcam_heatmap = apply_gradcam(
    classifier.model,
    image_path,
    image_tensor,
    model_name=classifier.model_name
)
gradcam_viz.save("gradcam_output.png")
print("Grad-CAM saved to: gradcam_output.png")

# ============================================
# 4. APPLY LIME
# ============================================
print("\n4. Generating LIME explanation...")
lime_result = apply_lime_image(
    classifier.model,
    image_path,
    classifier.transform,
    classifier.classes,
    device=device,
    num_samples=500,  # Reduce for faster computation
    num_features=10
)
lime_result['visualization'].save("lime_output.png")
print("LIME saved to: lime_output.png")
print(f"LIME Prediction: {lime_result['predicted_class']}")
print(f"LIME Probabilities: {lime_result['probabilities']}")

# ============================================
# 5. APPLY SHAP
# ============================================
print("\n5. Generating SHAP explanation...")
shap_result = apply_shap_image(
    classifier.model,
    image_path,
    classifier.transform,
    classifier.classes,
    device=device
)
shap_result['visualization_overlay'].save("shap_output.png")
print("SHAP saved to: shap_output.png")
print(f"SHAP Prediction: {shap_result['predicted_class']}")
print(f"SHAP Confidence: {shap_result['confidence']:.4f}")

# ============================================
# 6. COMPARE ALL METHODS
# ============================================
print("\n" + "="*50)
print("COMPARISON OF ALL XAI METHODS")
print("="*50)
print(f"\nOriginal Prediction: {result['prediction']} ({result['confidence']:.4f})")
print(f"LIME Prediction: {lime_result['predicted_class']}")
print(f"SHAP Prediction: {shap_result['predicted_class']}")
print("\nAll explanations generated successfully!")
print("Check output files:")
print("  - gradcam_output.png")
print("  - lime_output.png")
print("  - shap_output.png")


# ============================================
# BATCH PROCESSING EXAMPLE
# ============================================
def process_multiple_images(image_paths, model_name='densenet'):
    """Process multiple images with all XAI methods"""
    classifier = LungCancerClassifier(model_name=model_name, device=device)
    results = []
    
    for img_path in image_paths:
        print(f"\nProcessing: {img_path}")
        
        # Predict
        pred = classifier.predict(img_path)
        
        # Generate explanations
        img_tensor = classifier.preprocess_image(img_path)
        gradcam_viz, _ = apply_gradcam(classifier.model, img_path, img_tensor)
        
        lime_result = apply_lime_image(
            classifier.model, img_path, 
            classifier.transform, classifier.classes, device
        )
        
        results.append({
            'image': img_path,
            'prediction': pred['prediction'],
            'confidence': pred['confidence'],
            'gradcam': gradcam_viz,
            'lime': lime_result['visualization']
        })
    
    return results


# Example: Process multiple images
image_list = ["xray1.jpg", "xray2.jpg", "xray3.jpg"]
batch_results = process_multiple_images(image_list)