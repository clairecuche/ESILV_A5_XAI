"""
Unified Explainable AI Interface
Multi-modal classification with automatic XAI compatibility
"""

import streamlit as st
import pandas as pd
import numpy as np
import librosa
import os
from tensorflow.keras.models import load_model
import librosa.display
import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf
import shap
import lime
from lime import lime_image
from skimage.segmentation import mark_boundaries
import cv2
from PIL import Image
import sys
import torch
import uuid
from pathlib import Path

import numpy as np
if not hasattr(np, 'int'):
    np.int = int
if not hasattr(np, 'float'):
    np.float = float
if not hasattr(np, 'bool'):
    np.bool = bool

# Ensure project Code/ modules are importable when running from Streamlit folder
project_root = os.path.dirname(os.path.dirname(__file__))
code_path = os.path.join(project_root, 'Code', 'Lung_Cancer_Detection')
if code_path not in sys.path:
    sys.path.insert(0, code_path)

# Import Lung Cancer modules (PyTorch) if available
try:
    from Lung_Cancer_Model import LungCancerClassifier
    from gradcam import apply_gradcam
    from lime_explainer import apply_lime_image
    from shap_explainer import apply_shap_image
    LUNG_CANCER_AVAILABLE = True
except Exception as e:
    LUNG_CANCER_AVAILABLE = False
    print(f"❌ Failed to import Lung Cancer modules: {e}")

# Page configuration
st.set_page_config(
    page_title="Unified XAI Interface",
    page_icon="🔬",
    layout="wide"
)

# ============================================
# COMPATIBILITY SYSTEM
# ============================================

class XAICompatibility:
    """Automatic XAI method filtering based on input type"""
    
    @staticmethod
    def get_available_xai_methods(input_type):
        """Return compatible XAI methods for input type"""
        if input_type == "audio":
            return ["LIME", "Grad-CAM", "SHAP"]
        elif input_type == "image":
            return ["LIME", "Grad-CAM", "SHAP"]  
        return []
    
    @staticmethod
    def get_available_models(input_type):
        """Return compatible models for input type"""
        if input_type == "audio":
            return ["MobileNet (Audio)", "VGG16 (Audio)", "Custom CNN"]
        elif input_type == "image":
            return ["DenseNet121", "AlexNet"]
        return []
    

@st.cache(allow_output_mutation=True)
def get_lung_classifier(model_name):
    """Charge le modèle PyTorch une seule fois et le garde en RAM"""
    model_key = model_name.lower().replace("121", "").replace(" ", "")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    classifier = LungCancerClassifier(model_name=model_key, device=device)
    classifier.model.eval() # Fixe les couches de Dropout/BatchNormalization
    return classifier

# ============================================
# AUDIO FUNCTIONS 
# ============================================

audio_class_names = ['real', 'fake']

def save_file(sound_file):
    """Save uploaded sound file to the project's root `audio_files/` and keep backups in Code/img folders."""
    filename = sound_file.name
    streamlit_dir = os.path.dirname(__file__)
    project_root = os.path.dirname(streamlit_dir)

    primary_dir = os.path.join(project_root, 'audio_files')

    if hasattr(sound_file, 'getbuffer'):
        data = sound_file.getbuffer()
    else:
        try:
            sound_file.seek(0)
        except Exception:
            pass
        data = sound_file.read()

    os.makedirs(primary_dir, exist_ok=True)
    primary_path = os.path.join(primary_dir, filename)
    with open(primary_path, 'wb') as f:
        f.write(data)

    return filename


def create_spectrogram(sound):
    """Create a melspectrogram from audio stored in the project root `audio_files/` folder."""
    streamlit_dir = os.path.dirname(__file__)
    project_root = os.path.dirname(streamlit_dir)
    audio_file = os.path.join(project_root, 'audio_files', sound)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    y, sr = librosa.load(audio_file)
    ms = librosa.feature.melspectrogram(y=y, sr=sr)
    log_ms = librosa.power_to_db(ms, ref=np.max)
    librosa.display.specshow(log_ms, sr=sr)
    plt.savefig('melspectrogram.png')
    image_data = load_img('melspectrogram.png', target_size=(224, 224))
    return(image_data)

def model_predict_numpy(model, img_batch: np.ndarray):
    """Call model with a numpy array and return numpy predictions.
    Supports Keras models, saved_model with 'serving_default' signature, and callable trackable objects."""
    try:
        # Keras models
        if hasattr(model, 'predict'):
            preds = model.predict(img_batch)
            return preds

        # SavedModel with signatures
        if hasattr(model, 'signatures') and 'serving_default' in model.signatures:
            fn = model.signatures['serving_default']
            # convert to tf.Tensor
            tf_in = tf.constant(img_batch)
            out = fn(tf_in)
            if isinstance(out, dict):
                # take first item
                out = list(out.values())[0].numpy()
            else:
                out = out.numpy()
            return out

        # Some TF Trackable objects are callable
        if callable(model):
            out = model(tf.constant(img_batch), training=False)
            # If returns a tensor or numpy array
            if isinstance(out, dict):
                out = list(out.values())[0].numpy()
            elif hasattr(out, 'numpy'):
                out = out.numpy()
            return out

        raise ValueError('Model type not supported for prediction')
    except Exception as e:
        # Re-raise with context
        raise RuntimeError(f'Model inference failed: {e}') from e


def predictions(image_data, model):
    img_array = np.array(image_data)
    img_array1 = img_array / 255.0
    img_array1 = img_array1.astype(np.float32)  # Ensure float32 for SavedModel
    img_batch = np.expand_dims(img_array1, axis=0)

    prediction = model_predict_numpy(model, img_batch)
    class_label = np.argmax(prediction)
    return class_label, prediction

def lime_predict(image_data, model):
    img_array = np.array(image_data)
    img_array1 = img_array / 255.0
    img_array1 = img_array1.astype(np.float32)
    img_batch = np.expand_dims(img_array1, axis=0)

    prediction = model_predict_numpy(model, img_batch)
    class_label = np.argmax(prediction)

    explainer = lime.lime_image.LimeImageExplainer()

    # Use our numpy prediction wrapper inside LIME
    explanation = explainer.explain_instance(
        img_array1.astype('float32'),
        lambda x: model_predict_numpy(model, x.astype('float32')),
        hide_color=0,
        num_samples=1000
    )
    
    fig, axs = plt.subplots(1, 2, figsize=(10, 25))
    temp, mask = explanation.get_image_and_mask(np.argmax(prediction[0], axis=0), 
                                                positive_only=False, 
                                                num_features=8, 
                                                hide_rest=True)
    axs[0].imshow(image_data)
    axs[1].imshow(mark_boundaries(temp, mask))
    axs[1].set_title(f"Predicted class: {audio_class_names[class_label]}")
    plt.tight_layout()
    return(fig)

def grad_predict(image_data, model_mob, preds, class_idx):
    img_array = img_to_array(image_data)
    x = np.expand_dims(img_array, axis=0)
    x = tf.keras.applications.vgg16.preprocess_input(x)
    
    model = tf.keras.applications.VGG16(weights='imagenet', include_top=True)
    last_conv_layer = model.get_layer('block5_conv3')
    grad_model = tf.keras.models.Model([model.inputs], [last_conv_layer.output, model.output])

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(x)
        class_output = preds[:, class_idx]
    grads = tape.gradient(class_output, last_conv_layer_output)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

    heatmap = cv2.resize(np.float32(heatmap), (x.shape[2], x.shape[1]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = heatmap.astype(np.float32)
    superimposed_img = cv2.addWeighted(x[0], 0.6, heatmap, 0.4, 0, dtype=cv2.CV_32F)

    fig1, ax = plt.subplots(1, 2, figsize=(10, 25))
    ax[0].imshow(image_data)
    ax[1].imshow(superimposed_img)
    ax[1].set_title(f"Predicted class: {audio_class_names[class_idx]}")
    plt.tight_layout()
    return(fig1)

def shap_predict(image_data, model):
    """SHAP-like explanation for audio using superpixel occlusion method"""
    from skimage.segmentation import slic
    from skimage.util import img_as_float
    
    img_array = np.array(image_data)
    img_array1 = img_array / 255.0
    img_array1 = img_array1.astype(np.float32)
    
    # Prédiction initiale
    img_batch = np.expand_dims(img_array1, axis=0)
    prediction = model_predict_numpy(model, img_batch)
    class_label = np.argmax(prediction)
    baseline_prob = prediction[0][class_label]
    
    try:
        st.info("🔄 Computing feature importance using superpixel occlusion (~20 seconds)...")
        
        # Créer des superpixels (segments)
        segments = slic(img_as_float(img_array), n_segments=80, compactness=10, sigma=1, start_label=0)
        n_segments = len(np.unique(segments))
        
        st.text(f"Analyzing {n_segments} superpixels...")
        
        # Calculer l'importance de chaque superpixel
        importance_values = np.zeros(n_segments)
        progress_bar = st.progress(0)
        
        for seg_id in range(n_segments):
            # Masquer ce superpixel
            masked_img = img_array1.copy()
            masked_img[segments == seg_id] = 0
            
            # Prédire avec superpixel masqué
            masked_batch = np.expand_dims(masked_img, axis=0)
            masked_pred = model_predict_numpy(model, masked_batch)
            masked_prob = masked_pred[0][class_label]
            
            # Importance = différence de probabilité
            importance_values[seg_id] = baseline_prob - masked_prob
            
            progress_bar.progress((seg_id + 1) / n_segments)
        
        progress_bar.empty()
        
        # Créer la carte d'importance
        importance_map = np.zeros((224, 224))
        for seg_id in range(n_segments):
            importance_map[segments == seg_id] = importance_values[seg_id]
        
        # Stats
        st.text(f"Importance range: [{importance_map.min():.4f}, {importance_map.max():.4f}]")
        
        # Créer la visualisation
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        
        # Image originale
        axs[0].imshow(image_data)
        axs[0].set_title("Original Spectrogram", fontsize=12, fontweight='bold')
        axs[0].axis('off')
        
        # Importance map
        vmax = max(np.abs(importance_map.min()), np.abs(importance_map.max()))
        im = axs[1].imshow(importance_map, cmap='RdBu_r', interpolation='bilinear', 
                          vmin=-vmax, vmax=vmax)
        axs[1].set_title("Feature Importance\n(SHAP-like method)", fontsize=12, fontweight='bold')
        axs[1].axis('off')
        plt.colorbar(im, ax=axs[1], fraction=0.046, label='Importance')
        
        # Superposition (utiliser valeurs absolues pour highlight)
        importance_abs = np.abs(importance_map)
        importance_norm = importance_abs / (importance_abs.max() + 1e-8)
        importance_colored = plt.cm.hot(importance_norm)[:, :, :3]
        
        img_normalized = np.array(image_data) / 255.0
        overlay = 0.6 * img_normalized + 0.4 * importance_colored
        overlay = np.clip(overlay, 0, 1)
        
        axs[2].imshow(overlay)
        axs[2].set_title(f"Importance Overlay\nPredicted: {audio_class_names[class_label]}", 
                        fontsize=12, fontweight='bold')
        axs[2].axis('off')
        
        plt.tight_layout()
        
        st.success("✅ Feature importance computation completed!")
        return fig
        
    except Exception as e:
        st.error(f"Computation failed: {str(e)}")
        print(f"Full error: {e}")
        import traceback
        traceback.print_exc()
        
        # Fallback: méthode plus simple par patches
        st.warning("Falling back to patch-based method...")
        return shap_predict_fallback(image_data, model, class_label)


def shap_predict_fallback(image_data, model, class_label):
    """Fallback method using simple occlusion"""
    img_array = np.array(image_data)
    img_array1 = img_array / 255.0
    img_array1 = img_array1.astype(np.float32)
    
    # Prédiction de base
    img_batch = np.expand_dims(img_array1, axis=0)
    baseline_pred = model_predict_numpy(model, img_batch)
    baseline_prob = baseline_pred[0][class_label]
    
    # Diviser en patches
    patch_size = 28  # 224/8 = 28
    n_patches = 224 // patch_size
    
    importance_map = np.zeros((n_patches, n_patches))
    
    progress_bar = st.progress(0)
    total_patches = n_patches * n_patches
    
    for i in range(n_patches):
        for j in range(n_patches):
            # Masquer ce patch
            masked_img = img_array1.copy()
            masked_img[i*patch_size:(i+1)*patch_size, 
                      j*patch_size:(j+1)*patch_size, :] = 0
            
            # Prédire
            masked_batch = np.expand_dims(masked_img, axis=0)
            masked_pred = model_predict_numpy(model, masked_batch)
            masked_prob = masked_pred[0][class_label]
            
            # Importance = différence de probabilité
            importance_map[i, j] = baseline_prob - masked_prob
            
            progress_bar.progress((i * n_patches + j + 1) / total_patches)
    
    progress_bar.empty()
    
    # Upscale
    from scipy.ndimage import zoom
    importance_full = zoom(importance_map, (224/n_patches, 224/n_patches), order=1)
    
    # Créer visualisation
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    axs[0].imshow(image_data)
    axs[0].set_title("Original Spectrogram", fontsize=12, fontweight='bold')
    axs[0].axis('off')
    
    im = axs[1].imshow(importance_full, cmap='RdBu_r', interpolation='bilinear')
    axs[1].set_title("Feature Importance\n(Occlusion-based)", fontsize=12, fontweight='bold')
    axs[1].axis('off')
    plt.colorbar(im, ax=axs[1], fraction=0.046, label='Importance')
    
    importance_abs = np.abs(importance_full)
    importance_norm = (importance_abs - importance_abs.min()) / (importance_abs.max() - importance_abs.min() + 1e-8)
    importance_colored = plt.cm.hot(importance_norm)[:, :, :3]
    
    img_normalized = np.array(image_data) / 255.0
    overlay = 0.6 * img_normalized + 0.4 * importance_colored
    overlay = np.clip(overlay, 0, 1)
    
    axs[2].imshow(overlay)
    axs[2].set_title(f"Importance Overlay\nPredicted: {audio_class_names[class_label]}", 
                    fontsize=12, fontweight='bold')
    axs[2].axis('off')
    
    plt.tight_layout()
    return fig


# ============================================
# IMAGE FUNCTIONS (LUNG CANCER)
# ============================================

image_class_names = ['No Finding', 'Malignant']

def save_image_file(image_file):
    """Save uploaded image file"""
    filename = image_file.name
    image_dir = 'image_files'
    os.makedirs(image_dir, exist_ok=True)
    
    filepath = os.path.join(image_dir, filename)
    with open(filepath, 'wb') as f:
        f.write(image_file.getbuffer())
    
    return filepath

def save_pil_image(pil_image, filename=None):
    """Save a PIL Image to image_files/ and return filepath"""
    image_dir = Path('image_files')
    image_dir.mkdir(parents=True, exist_ok=True)
    if filename is None:
        filename = f"xray_{uuid.uuid4().hex[:8]}.png"
    filepath = image_dir / filename
    pil_image.save(filepath)
    return str(filepath)

def load_and_preprocess_image(image_path, target_size=(224, 224)):
    """Load and preprocess chest X-ray image"""
    if isinstance(image_path, str):
        image = Image.open(image_path).convert('RGB')
    else:
        image = Image.open(image_path).convert('RGB')
    image = image.resize(target_size)
    return image

def predictions_image(image_data, model_name):
    """Predict image class using LungCancerClassifier"""
    if not LUNG_CANCER_AVAILABLE:
        st.error("Lung Cancer modules not available")
        return 0, np.array([[0.5, 0.5]])
    
    try:
        classifier = get_lung_classifier(model_name)

        with torch.no_grad():
            # Prédiction
            result = classifier.predict(image_data)
        
        # Convertir en format compatible
        probs = np.array([[
            result['probabilities'][classifier.classes[0]], 
            result['probabilities'][classifier.classes[1]]
        ]])
        
        class_label = 0 if result['prediction'] == classifier.classes[0] else 1
        
        return class_label, probs
    
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
        return 0, np.array([[0.5, 0.5]])

def lime_predict_image(image_data, model_name):
    """LIME explanation for chest X-ray using dedicated lime_explainer module"""
    if not LUNG_CANCER_AVAILABLE:
        st.error("Lung Cancer modules not available")
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        axs[0].imshow(image_data)
        axs[0].set_title("Original X-ray")
        axs[0].axis('off')
        axs[1].imshow(image_data)
        axs[1].set_title("LIME (Not Available)")
        axs[1].axis('off')
        plt.tight_layout()
        return fig
    
    try:
        # Sauvegarder l'image
        temp_path = save_pil_image(image_data)
        
        model_key = model_name.lower().replace("121", "").replace(" ", "")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        classifier = get_lung_classifier(model_name)
        classifier.model.eval()
        
        # Utiliser le module dédié lime_explainer
        lime_result = apply_lime_image(
            classifier.model,
            temp_path,
            classifier.transform,
            classifier.classes,
            device=device,
            num_samples=500,
            num_features=10
        )
        
        # Créer figure à partir du résultat
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        axs[0].imshow(image_data)
        axs[0].set_title("Original X-ray", fontsize=12, fontweight='bold')
        axs[0].axis('off')
        
        # Afficher la visualisation LIME
        axs[1].imshow(lime_result['visualization'])
        axs[1].set_title(f"LIME Explanation\nPredicted: {lime_result['predicted_class']}", 
                        fontsize=12, fontweight='bold')
        axs[1].axis('off')
        
        plt.tight_layout()
        
        return fig
    
    except Exception as e:
        st.error(f"Error in LIME: {str(e)}")
        st.exception(e)
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(image_data)
        ax.set_title("Error generating LIME")
        ax.axis('off')
        return fig


def grad_predict_image(image_data, model_name, preds=None, class_idx=None):
    """Grad-CAM explanation using dedicated gradcam module"""
    if not LUNG_CANCER_AVAILABLE:
        st.error("Lung Cancer modules not available")
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].imshow(image_data)
        ax[0].set_title("Original X-ray")
        ax[0].axis('off')
        ax[1].imshow(image_data)
        ax[1].set_title("Grad-CAM (Not Available)")
        ax[1].axis('off')
        plt.tight_layout()
        return fig
    
    try:
        # Sauvegarder l'image
        temp_path = save_pil_image(image_data)
        
        model_key = model_name.lower().replace("121", "").replace(" ", "")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        classifier = get_lung_classifier(model_name)
        classifier.model.eval()
        
        # Préparer le tenseur
        image_tensor = classifier.preprocess_image(temp_path)
        
        # Utiliser le module dédié gradcam
        vis, heatmap = apply_gradcam(
            classifier.model, 
            temp_path, 
            image_tensor, 
            model_name=model_key
        )
        
        # Créer figure
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        axs[0].imshow(image_data)
        axs[0].set_title("Original X-ray", fontsize=12, fontweight='bold')
        axs[0].axis('off')
        axs[1].imshow(vis)
        axs[1].set_title("Grad-CAM Visualization", fontsize=12, fontweight='bold')
        axs[1].axis('off')
        plt.tight_layout()
        
        return fig
    
    except Exception as e:
        st.error(f"Error in Grad-CAM: {str(e)}")
        st.exception(e)
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(image_data)
        ax.set_title("Error generating Grad-CAM")
        ax.axis('off')
        return fig

# ============================================
# MAIN NAVIGATION
# ============================================

def main():
    st.sidebar.title("🔬 Navigation")
    page = st.sidebar.selectbox(
        "App Sections",
        ["Classification", "Comparison"]
    )
    
    if page == "Classification":
        classification_page()
    elif page == "Comparison":
        comparison_page()

# ============================================
# PAGE: CLASSIFICATION
# ============================================

def classification_page():
    st.title("📊 Multi-Modal Classification")
    st.markdown("---")
    
    input_type = st.radio(
        "Select Input Type:",
        ["Audio (.wav)", "Image (X-ray)"],
        horizontal=True
    )
    
    input_mode = "audio" if "Audio" in input_type else "image"
    
    if input_mode == "audio":
        st.write('### Choose a wav file')
        uploaded_file = st.file_uploader(' ', type='wav', key="audio_upload")
        class_names = audio_class_names
    else:
        st.write('### Choose a chest X-ray image')
        uploaded_file = st.file_uploader(' ', type=['jpg', 'jpeg', 'png'], key="image_upload")
        class_names = image_class_names
    
    if uploaded_file is not None:
        available_xai = XAICompatibility.get_available_xai_methods(input_mode)
        
        st.subheader("⚙️ Configuration")
              
        col1, col2 = st.columns(2)
              
        with col1:
            if input_mode == "image":
                selected_model = st.selectbox("Select Model", ["AlexNet", "DenseNet"])
            

        with col2:
            selected_xai = st.selectbox(
                "Select XAI Method",
                available_xai,
                help=f"XAI methods available for {input_mode}"
            )
    
        st.write('___')
        # ==================== AUDIO PROCESSING ====================
        if input_mode == "audio":
            st.write('### Play audio')
            audio_bytes = uploaded_file.read()
            st.audio(audio_bytes, format='audio/wav')

            st.write('### Spectrogram Image:')
            save_file(uploaded_file)
            sound = uploaded_file.name
            
            with st.spinner('Generating spectrogram...'):
                spec = create_spectrogram(sound)
                st.image(spec, width=700)
                model = tf.saved_model.load('saved_model/model')
            
            st.write('### Classification results:')
            class_label, prediction = predictions(spec, model)
            st.write("#### The uploaded audio file is " + class_names[class_label])
            
            confidence = float(np.max(prediction))
            st.metric("Confidence", f"{confidence*100:.2f}%")
            
            if st.button('Show XAI Metrics'):
                if selected_xai == 'LIME':
                    st.write('### XAI Metrics using LIME')
                    with st.spinner('Generating LIME...'):
                        fig2 = lime_predict(spec, model)
                        st.pyplot(fig2)
                
                elif selected_xai == 'Grad-CAM':
                    st.write('### XAI Metrics using Grad-CAM')
                    with st.spinner('Generating Grad-CAM...'):
                        grad_img = grad_predict(spec, model, prediction, class_label)
                        st.pyplot(grad_img)

                elif selected_xai == 'SHAP':
                    st.write('### XAI Metrics using SHAP')
                    with st.spinner('Generating SHAP values... This may take a moment.'):
                        shap_fig = shap_predict(spec, model)
                        st.pyplot(shap_fig)
                    st.info("SHAP highlights the most important features in the spectrogram for classification.")
                
        
        # ==================== IMAGE PROCESSING ====================
        else:
            st.write('### Uploaded X-ray:')
            image = load_and_preprocess_image(uploaded_file)
            st.image(image, width=700)
            
            image_path = save_image_file(uploaded_file)
            
            st.write('### Classification results:')
            with st.spinner('Classifying...'):
                class_label, prediction = predictions_image(image, selected_model)
            
            st.write("#### The uploaded X-ray is " + class_names[class_label])
            
            confidence = float(np.max(prediction))
            st.metric("Confidence", f"{confidence*100:.2f}%")
            
            if st.button('Show XAI Metrics'):
                if selected_xai == 'LIME':
                    st.write('### XAI Metrics using LIME')
                    with st.spinner('Generating LIME explanation...'):
                        fig2 = lime_predict_image(image, selected_model)
                        st.pyplot(fig2)
                
                elif selected_xai == 'Grad-CAM':
                    st.write('### XAI Metrics using Grad-CAM')
                    with st.spinner('Generating Grad-CAM...'):
                        grad_img = grad_predict_image(image, selected_model, prediction, class_label)
                        st.pyplot(grad_img)
                
                elif selected_xai == "SHAP":
                    st.markdown(f"### {selected_xai}")
                    with st.spinner(f"Generating SHAP..."):
                        # On récupère le classifier complet pour avoir accès au transform et aux classes
                        classifier = get_lung_classifier(selected_model)
                        temp_path = save_pil_image(image)
                        
                        shap_result = apply_shap_image(
                            model=classifier.model,
                            image_path=temp_path,
                            transform=classifier.transform,
                            classes=classifier.classes,
                            device=classifier.device
                        )
                        # Affichage du résultat SHAP
                        st.image(shap_result['visualization_overlay'], use_column_width=True)
    
    elif uploaded_file is None:
        if input_mode == "audio":
            st.info("📁 Please upload a .wav file")
        else:
            st.info("📁 Please upload an image file (jpg, jpeg, png)")

# ============================================
# PAGE: COMPARISON
# ============================================

def comparison_page():
    st.title("🔄 XAI Methods Comparison")
    st.markdown("---")
    
    st.info("Upload a file and compare multiple XAI methods side-by-side")
    
    input_type = st.radio(
        "Select Input Type:",
        ["Audio (.wav)", "Image (X-ray)"],
        horizontal=True,
        key="comparison_input_type"
    )
    
    input_mode = "audio" if "Audio" in input_type else "image"
    
    if input_mode == "audio":
        uploaded_file = st.file_uploader("Upload Audio", type=['wav'], key="compare_audio")
        class_names = audio_class_names
    else:
        uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'], key="compare_image")
        class_names = image_class_names
    
    if uploaded_file is not None:
        available_xai = XAICompatibility.get_available_xai_methods(input_mode)
        available_models = XAICompatibility.get_available_models(input_mode)
        
        # Sélection du modèle
        selected_model = st.selectbox("Select Model", available_models)
        
        # Multi-select XAI methods
        selected_methods = st.multiselect(
            "Select XAI Methods to Compare",
            available_xai,
            default=available_xai[:2] if len(available_xai) >= 2 else available_xai
        )
        
        if len(selected_methods) < 2:
            st.warning("Please select at least 2 methods for comparison")
            return
        
        # Display input
        if input_mode == "audio":
            st.audio(uploaded_file, format='audio/wav')
            save_file(uploaded_file)
            sound = uploaded_file.name
        else:
            image = load_and_preprocess_image(uploaded_file)
            st.image(image, width=700)
        
        if st.button("🔍 Compare Methods", type="primary"):
            st.subheader("📊 Comparison Results")
            
            cols = st.columns(len(selected_methods))
            
            if input_mode == "audio":
                with st.spinner('Generating spectrogram...'):
                    spec = create_spectrogram(sound)
                    model = tf.saved_model.load('saved_model/model')
                    class_label, prediction = predictions(spec, model)
                
                for idx, method in enumerate(selected_methods):
                    with cols[idx]:
                        st.markdown(f"### {method}")
                        with st.spinner(f"Generating {method}..."):
                            if method == "LIME":
                                fig = lime_predict(spec, model)
                                st.pyplot(fig)
                            elif method == "Grad-CAM":
                                fig = grad_predict(spec, model, prediction, class_label)
                                st.pyplot(fig)
            
            else:  # image
                for idx, method in enumerate(selected_methods):
                    with cols[idx]:
                        st.markdown(f"### {method}")
                        with st.spinner(f"Generating {method}..."):
                            if method == "LIME":
                                fig = lime_predict_image(image, selected_model)
                                st.pyplot(fig)
                            elif method == "Grad-CAM":
                                class_label, prediction = predictions_image(image, selected_model)
                                fig = grad_predict_image(image, selected_model, prediction, class_label)
                                st.pyplot(fig)

if __name__ == "__main__":
    main()