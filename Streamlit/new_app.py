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
import lime
from lime import lime_image
from skimage.segmentation import mark_boundaries
import cv2
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="Unified XAI Interface",
    page_icon=""
)


# ============================================
# COMPATIBILITY SYSTEM (NEW)
# ============================================

class XAICompatibility:
    """Automatic XAI method filtering based on input type"""
    
    @staticmethod
    def get_available_xai_methods(input_type):
        """Return compatible XAI methods for input type"""
        if input_type == "audio":
            return ["LIME", "Grad-CAM"]
        elif input_type == "image":
            return ["LIME", "Grad-CAM", "SHAP"]
        return []
    
    @staticmethod
    def get_available_models(input_type):
        """Return compatible models for input type"""
        if input_type == "audio":
            return ["MobileNet (Audio)", "VGG16 (Audio)", "Custom CNN"]
        elif input_type == "image":
            return ["AlexNet", "DenseNet121"]
        return []

# ============================================
# AUDIO FUNCTIONS 
# ============================================

audio_class_names = ['real', 'fake']

def save_file(sound_file):
    # Save the uploaded sound file to three locations:
    # 1) local Streamlit/audio_files (used by this app)
    # 2) Code/Deepfake_Audio/audio_files (project code folder)
    # 3) img/Deepfake_Audio/audio_files (image assets folder)
    filename = sound_file.name
    streamlit_dir = os.path.dirname(__file__)
    project_root = os.path.dirname(streamlit_dir)

    target_dirs = [
        os.path.join(streamlit_dir, 'audio_files'),
        os.path.join(project_root, 'Code', 'Deepfake_Audio', 'audio_files'),
        os.path.join(project_root, 'img', 'Deepfake_Audio', 'audio_files'),
    ]

    # Read data once (handle UploadedFile API differences)
    if hasattr(sound_file, 'getbuffer'):
        data = sound_file.getbuffer()
    else:
        try:
            sound_file.seek(0)
        except Exception:
            pass
        data = sound_file.read()

    for d in target_dirs:
        os.makedirs(d, exist_ok=True)
        target_path = os.path.join(d, filename)
        with open(target_path, 'wb') as f:
            f.write(data)

    return filename

def create_spectrogram(sound):
    audio_file = os.path.join('audio_files/', sound)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    y, sr = librosa.load(audio_file)
    ms = librosa.feature.melspectrogram(y=y, sr=sr)
    log_ms = librosa.power_to_db(ms, ref=np.max)
    librosa.display.specshow(log_ms, sr=sr)
    # st.pyplot(fig)
    plt.savefig('melspectrogram.png')
    image_data = load_img('melspectrogram.png', target_size=(224, 224))
    # st.image(image_data)
    return(image_data)

def predictions(image_data, model):
    img_array = np.array(image_data)
    img_array1 = img_array / 255.0
    img_array1 = img_array1.astype(np.float32)  # Ensure float32 for SavedModel
    img_batch = np.expand_dims(img_array1, axis=0)

    prediction = model(img_batch, training=False)
    class_label = np.argmax(prediction)
    return class_label, prediction

def lime_predict(image_data, model):
    img_array = np.array(image_data)
    img_array1 = img_array / 255.0
    img_array1 = img_array1.astype(np.float32)  # Ensure float32 for SavedModel
    img_batch = np.expand_dims(img_array1, axis=0)

    prediction = model(img_batch, training=False)
    class_label = np.argmax(prediction)

    explainer = lime.lime_image.LimeImageExplainer()
    # explanation = explainer.explain_instance(img_array.astype('float64'), model.predict, hide_color=0, num_samples=1000)
    explanation = explainer.explain_instance(img_array1.astype('float32'), lambda x: model(x, training=False).numpy(), hide_color=0, num_samples=1000)
    
    fig, axs = plt.subplots(1, 2, figsize=(10, 25))
    for i in range(2):
        # Show the original image and the explanation
        temp, mask = explanation.get_image_and_mask(np.argmax(prediction[0], axis=0), positive_only=False, num_features=8, hide_rest=True)
        axs[0].imshow(image_data)
        axs[1].imshow(mark_boundaries(temp, mask))
        axs[1].set_title(f"Predicted class: {audio_class_names[class_label]}")
    plt.tight_layout()
    # plt.show()
    # plt.savefig('XAI_output.png')
    # st.pyplot(fig)
    return(fig)

def grad_predict(image_data, model_mob, preds, class_idx):
    img_array = img_to_array(image_data)
    # img_array1 = img_array / 255
    x = np.expand_dims(img_array, axis=0)
    x = tf.keras.applications.vgg16.preprocess_input(x)
    
    # Note: grad_predict uses a separate VGG16 model, not the loaded SavedModel

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

    # fig, ax = plt.subplots()
    # ax.title('Grad-CAM visualization')
    # st.write(superimposed_img)
    # plt.imshow(superimposed_img)
    # plt.savefig('XAI_output.png')
    # st.pyplot(fig)
    # st.pyplot(superimposed_img)

    fig1, ax = plt.subplots(1, 2, figsize=(10, 25))
    for i in range(2):
        # Show the original image and the explanation
        ax[0].imshow(image_data)
        ax[1].imshow(superimposed_img)
        ax[1].set_title(f"Predicted class: {audio_class_names[class_idx]}")
    plt.tight_layout()
    # plt.show()
    # plt.savefig('XAI_output.png')
    # st.pyplot(fig1)
    return(fig1)

# ============================================
# IMAGE FUNCTIONS 
# ============================================

image_class_names = ['Benign', 'Malignant']

def save_image_file(image_file):
    """Save uploaded image file"""
    filename = image_file.name
    image_dir = 'image_files'
    os.makedirs(image_dir, exist_ok=True)
    
    filepath = os.path.join(image_dir, filename)
    with open(filepath, 'wb') as f:
        f.write(image_file.getbuffer())
    
    return filepath

def load_and_preprocess_image(image_path, target_size=(224, 224)):
    """Load and preprocess chest X-ray image"""
    if isinstance(image_path, str):
        image = Image.open(image_path).convert('RGB')
    else:
        # Handle Streamlit UploadedFile object
        image = Image.open(image_path).convert('RGB')
    image = image.resize(target_size)
    return image

def predictions_image(image_data, model_name="densenet"):
    """Predict image class - PLACEHOLDER (replace with actual lung cancer model)"""
    # TODO: Replace with actual lung cancer model
    # from lung_cancer_models import LungCancerClassifier
    # classifier = LungCancerClassifier(model_name=model_name, device='cpu')
    # result = classifier.predict(image_data)
    
    # Dummy prediction for now
    img_array = np.array(image_data) / 255.0
    img_array = img_array.astype(np.float32)
    img_batch = np.expand_dims(img_array, axis=0)
    
    # Fake prediction
    prediction = np.array([[0.3, 0.7]])  # [Benign, Malignant]
    class_label = np.argmax(prediction)
    
    return class_label, prediction

def lime_predict_image(image_data, model_name="densenet"):
    """LIME explanation for chest X-ray - PLACEHOLDER"""
    # TODO: Replace with actual LIME implementation for lung cancer
    # from lime_image_explainer import apply_lime_image
    # result = apply_lime_image(model, image_data, transform, image_class_names)
    
    class_label, prediction = predictions_image(image_data, model_name)
    
    # Create placeholder visualization
    fig, axs = plt.subplots(1, 2, figsize=(10, 25))
    axs[0].imshow(image_data)
    axs[0].set_title("Original X-ray")
    axs[0].axis('off')
    
    axs[1].imshow(image_data)
    axs[1].set_title(f"LIME Explanation\nPredicted: {image_class_names[class_label]}")
    axs[1].axis('off')
    
    plt.tight_layout()
    return fig

def grad_predict_image(image_data, model_name="densenet", preds=None, class_idx=None):
    """Grad-CAM explanation for chest X-ray - PLACEHOLDER"""
    # TODO: Replace with actual Grad-CAM implementation for lung cancer
    # from gradcam_implementation import apply_gradcam
    # visualization, heatmap = apply_gradcam(model, image_data, image_tensor, model_name)
    
    if class_idx is None:
        class_idx, preds = predictions_image(image_data, model_name)
    
    # Create placeholder visualization
    fig, ax = plt.subplots(1, 2, figsize=(10, 25))
    ax[0].imshow(image_data)
    ax[0].set_title("Original X-ray")
    ax[0].axis('off')
    
    ax[1].imshow(image_data)
    ax[1].set_title(f"Grad-CAM Heatmap\nPredicted: {image_class_names[class_idx]}")
    ax[1].axis('off')
    
    plt.tight_layout()
    return fig

# ============================================
# MAIN NAVIGATION
# ============================================

def main():
    # Sidebar navigation
    st.sidebar.title("🔬 Navigation")
    page = st.sidebar.selectbox(
        "App Sections",
        ["Homepage", "Classification", "Comparison", "About"]
    )
    
    if page == "Homepage":
        homepage()
    elif page == "Classification":
        classification_page()
    elif page == "Comparison":
        comparison_page()
    elif page == "About":
        about()

# ============================================
# PAGE: ABOUT (ORIGINAL)
# ============================================

def about():
    # st.set_page_config(layout="centered")
    st.title("About present work")
    st.markdown("**Deepfake audio refers to synthetically created audio by digital or manual means. An emerging field, it is used to not only create legal digital hoaxes, but also fool humans into believing it is a human speaking to them. Through this project, we create our own deep faked audio using Generative Adversarial Neural Networks (GANs) and objectively evaluate generator quality using Fréchet Audio Distance (FAD) metric. We augment a pre-existing dataset of real audio samples with our fake generated samples and classify data as real or fake using MobileNet, Inception, VGG and custom CNN models. MobileNet is the best performing model with an accuracy of 91.5% and precision of 0.507. We further convert our black box deep learning models into white box models, by using explainable AI (XAI) models. We quantitatively evaluate the classification of a MEL Spectrogram through LIME, SHAP and GradCAM models. We compare the features of a spectrogram that an XAI model focuses on to provide a qualitative analysis of frequency distribution in spectrograms.**")
    st.markdown("**The goal of this project is to study features of audio and bridge the gap of explain ability in deep fake audio detection, through our novel system pipeline. The findings of this study are applicable to the fields of phishing audio calls and digital mimicry detection on video streaming platforms. The use of XAI will provide end-users a clear picture of frequencies in audio that are flagged as fake, enabling them to make better decisions in generation of fake samples through GANs.**")
    
    st.markdown("---")
    st.markdown("### Lung Cancer Detection")
    st.markdown("**This module detects malignant tumors in chest X-rays using transfer learning with AlexNet and DenseNet architectures. The system uses Grad-CAM visualizations to explain model decisions and support radiologists in early and accurate diagnosis.**")

# ============================================
# PAGE: HOMEPAGE (MODIFIED)
# ============================================

def homepage():
    st.title("🔬 Unified Explainable AI Interface")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("🎵 Audio Analysis")
        st.markdown("""
        **Deepfake Audio Detection**
        - Upload `.wav` audio files
        - Detect real vs. fake audio
        - Models: MobileNet, VGG16, Custom CNN
        - XAI Methods: LIME, Grad-CAM
        """)
        if st.button("Go to Audio Classification"):
            st.session_state.page = "Classification"
            st.session_state.input_type = "audio"
    
    with col2:
        st.header("🫁 Image Analysis")
        st.markdown("""
        **Lung Cancer Detection**
        - Upload chest X-ray images
        - Detect malignant tumors
        - Models: AlexNet, DenseNet121
        - XAI Methods: LIME, Grad-CAM, SHAP
        """)
        if st.button("Go to Image Classification"):
            st.session_state.page = "Classification"
            st.session_state.input_type = "image"
    
    st.markdown("---")
    st.info("👈 Use the sidebar to navigate between sections")

# ============================================
# PAGE: CLASSIFICATION (NEW - UNIFIED)
# ============================================

def classification_page():
    st.title("📊 Multi-Modal Classification")
    st.markdown("---")
    
    # Input type selection
    input_type = st.radio(
        "Select Input Type:",
        ["Audio (.wav)", "Image (X-ray)"],
        horizontal=True
    )
    
    input_mode = "audio" if "Audio" in input_type else "image"
    
    # File uploader
    if input_mode == "audio":
        st.write('### Choose a wav file')
        uploaded_file = st.file_uploader(' ', type='wav', key="audio_upload")
        class_names = audio_class_names
    else:
        st.write('### Choose a chest X-ray image')
        uploaded_file = st.file_uploader(' ', type=['jpg', 'jpeg', 'png'], key="image_upload")
        class_names = image_class_names
    
    if uploaded_file is not None:
        # Get compatible models and XAI methods
        available_models = XAICompatibility.get_available_models(input_mode)
        available_xai = XAICompatibility.get_available_xai_methods(input_mode)
        
        # Configuration
        st.subheader("⚙️ Configuration")
        col1, col2 = st.columns(2)
        
        with col1:
            selected_model = st.selectbox(
                "Select Model",
                available_models,
                help=f"Models compatible with {input_mode} input"
            )
        
        with col2:
            selected_xai = st.selectbox(
                "Select XAI Method",
                available_xai,
                help=f"XAI methods available for {input_mode}"
            )
        
        st.write('___')
        
        # ==================== AUDIO PROCESSING ====================
        if input_mode == "audio":
            # view details
            # file_details = {'filename':uploaded_file.name, 'filetype':uploaded_file.type, 'filesize':uploaded_file.size}
            # st.write(file_details)
            
            # read and play the audio file
            st.write('### Play audio')
            audio_bytes = uploaded_file.read()
            st.audio(audio_bytes, format='audio/wav')

            st.write('### Spectrogram Image:')
            save_file(uploaded_file)
            # define the filename
            sound = uploaded_file.name
            
            with st.spinner('Fetching Results...'):
                spec = create_spectrogram(sound)
                st.image(spec, use_column_width=True)
                # Load SavedModel using tf.saved_model.load (Keras 3 compatible)
                model = tf.saved_model.load('saved_model/model')
            
            st.write('### Classification results:')
            class_label, prediction = predictions(spec, model)
            st.write("#### The uploaded audio file is " + class_names[class_label])
            
            # Show confidence
            confidence = float(np.max(prediction))
            st.metric("Confidence", f"{confidence*100:.2f}%")
            
            if st.button('Show XAI Metrics'):
                if selected_xai == 'LIME':
                    st.write('### XAI Metrics using LIME')
                    with st.spinner('Fetching Results...'):
                        fig2 = lime_predict(spec, model)
                        st.pyplot(fig2)
                
                elif selected_xai == 'Grad-CAM':
                    st.write('### XAI Metrics using Grad-CAM')
                    with st.spinner('Fetching Results...'):
                        grad_img = grad_predict(spec, model, prediction, class_label)
                        st.pyplot(grad_img)
        
        # ==================== IMAGE PROCESSING ====================
        else:  # image
            # Display uploaded image
            st.write('### Uploaded X-ray:')
            image = load_and_preprocess_image(uploaded_file)
            st.image(image, use_column_width=True)
            
            # Save file
            image_path = save_image_file(uploaded_file)
            
            with st.spinner('Fetching Results...'):
                pass  # Model loading will go here
            
            st.write('### Classification results:')
            class_label, prediction = predictions_image(image, selected_model)
            st.write("#### The uploaded X-ray is " + class_names[class_label])
            
            # Show confidence
            confidence = float(np.max(prediction))
            st.metric("Confidence", f"{confidence*100:.2f}%")
            
            if st.button('Show XAI Metrics'):
                if selected_xai == 'LIME':
                    st.write('### XAI Metrics using LIME')
                    with st.spinner('Fetching Results...'):
                        fig2 = lime_predict_image(image, selected_model)
                        st.pyplot(fig2)
                
                elif selected_xai == 'Grad-CAM':
                    st.write('### XAI Metrics using Grad-CAM')
                    with st.spinner('Fetching Results...'):
                        grad_img = grad_predict_image(image, selected_model, prediction, class_label)
                        st.pyplot(grad_img)
                
                elif selected_xai == 'SHAP':
                    st.write('### XAI Metrics using SHAP')
                    st.info('SHAP implementation coming soon for images')
    
    elif uploaded_file is None:
        if input_mode == "audio":
            st.info("Please upload a .wav file")
        else:
            st.info("Please upload an image file (jpg, jpeg, png)")

# ============================================
# PAGE: COMPARISON (NEW)
# ============================================

def comparison_page():
    st.title("🔄 XAI Methods Comparison")
    st.markdown("---")
    
    st.info("Upload a file and compare multiple XAI methods side-by-side")
    
    # Input type selection
    input_type = st.radio(
        "Select Input Type:",
        ["Audio (.wav)", "Image (X-ray)"],
        horizontal=True,
        key="comparison_input_type"
    )
    
    input_mode = "audio" if "Audio" in input_type else "image"
    
    # File uploader
    if input_mode == "audio":
        uploaded_file = st.file_uploader("Upload Audio", type=['wav'], key="compare_audio")
        class_names = audio_class_names
    else:
        uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'], key="compare_image")
        class_names = image_class_names
    
    if uploaded_file is not None:
        available_xai = XAICompatibility.get_available_xai_methods(input_mode)
        
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
            st.image(image, use_column_width=True)
        
        if st.button("🔍 Compare Methods", type="primary"):
            st.subheader("📊 Comparison Results")
            
            # Create columns for comparison
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
                                fig = lime_predict_image(image)
                                st.pyplot(fig)
                            elif method == "Grad-CAM":
                                class_label, prediction = predictions_image(image)
                                fig = grad_predict_image(image, "densenet", prediction, class_label)
                                st.pyplot(fig)
                            elif method == "SHAP":
                                st.info("SHAP visualization coming soon")

if __name__ == "__main__":
    main()