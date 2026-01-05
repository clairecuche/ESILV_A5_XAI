# Unified Explainable AI Interface for Multi-Modal Classification

## Project Overview

This project provides a **unified, multi-modal interface** for classification and explainability across two domains:
1. **Audio Classification**: Deepfake audio detection using spectrogram-based models
2. **Image Classification**: Lung cancer detection from chest X-ray images

The interface integrates multiple pre-trained models and Explainable AI (XAI) techniques, automatically filtering compatible XAI methods based on input type. This ensures a user-friendly experience where only applicable techniques are presented for each data modality.

### Key Features

**Multi-modal input support**: Audio (.wav) and medical images (chest X-rays)  
**Multiple pre-trained models**: 
  - Audio: SavedModel with MobileNet/VGG16/ResNet architectures
  - Image: DenseNet121, AlexNet   
**Comprehensive XAI methods**: LIME, Grad-CAM, SHAP for both modalities  
**Automatic compatibility filtering**: XAI methods are automatically filtered based on input type  
**Side-by-side comparison**: Compare multiple XAI techniques simultaneously  
**Interactive Streamlit interface**: User-friendly GUI with real-time visualization  

---

## Datasets

### Audio: Deepfake Detection
The 'Fake or Real' dataset from York University containing authentic and deepfake audio recordings. Audio files are converted to mel-spectrograms for classification.

### Image: Lung Cancer Detection
Chest X-ray images for lung cancer classification (benign/malignant/normal). Models use transfer learning on medical imaging data.

---

## Project Structure

```
ESILV_A5_XAI/
├── Code/
│   ├── Deepfake_Audio/
│   │   ├── Audio_classifier.ipynb
│   │   ├── InceptionV3-MobileNet.ipynb
│   │   ├── Spectrogram-converter.ipynb
│   │   └── VGG16-Custom CNN-ResNet.ipynb
│   └── Lung_Cancer_Detection/
│       ├── Lung_Cancer_Model.py
│       ├── Lung_Cancer_Test.py
│       ├── gradcam.py
│       ├── lime_explainer.py
│       └── shap_explainer.py
├── Streamlit/
│   ├── new_app.py              # Main unified interface
│   ├── app.py                  # Legacy audio interface
│   └── image_files/
├── saved_model/                # Pre-trained audio models
├── img/                        # Sample outputs and documentation
├── requirements.txt
└── README.md
```

---

## Setup and Installation

### Prerequisites
- Python 3.11 or higher
- Virtual environment (recommended)

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd ESILV_A5_XAI
   ```

2. **Create and activate virtual environment**
   ```bash
   # Windows
   python -m venv .venv
   .venv\Scripts\activate

   # Linux/Mac
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
---

## Running the Application

### Start the Streamlit Interface

```bash
streamlit run ./Streamlit/new_app.py
```

The application will open in your default browser at `http://localhost:8501`

### Interface Workflow

1. **Select Input Type**: Choose between Audio or Image classification
2. **Upload Data**: 
   - Audio: Upload `.wav` file
   - Image: Upload chest X-ray image (`.jpg`, `.png`, `.jpeg`)
3. **Select Model**: Choose from available pre-trained models 
4. **Choose XAI Method**: Select from compatible explainability techniques
5. **View Results**: Classification results with XAI visualization
6. **Compare (Optional)**: Use comparison mode to evaluate multiple XAI methods side-by-side

---

## Technical Details

### Models

#### Audio Models (TensorFlow/Keras)
- **SavedModel**: Spectrogram-based CNN with MobileNet/VGG16/ResNet backbones
- Input: Mel-spectrogram (converted from .wav audio)
- Output: Binary classification (Real/Fake)

#### Image Models (PyTorch)

**Medical-Specific Models** :
- **DenseNet121 (Medical)**: Uses TorchXRayVision pretrained weights
  - Pretrained on: ChestX-ray14, CheXpert, MIMIC-CXR, PadChest
  - Input: Grayscale chest X-rays (224×224, [-1024, 1024] HU range)

**ImageNet Models** :
- **DenseNet121 (ImageNet)**: Dense Convolutional Network with ImageNet weights
- **AlexNet (ImageNet)**: Classic CNN architecture with ImageNet weights
- Input: RGB images (224×224)
- Output: Binary classification (No Finding/Malignant)


### XAI Techniques

| XAI Method | Audio | Image | Description |
|------------|-------|-------|-------------|
| **LIME** | ✅ | ✅ | Local Interpretable Model-agnostic Explanations |
| **Grad-CAM** | ✅ | ✅ | Gradient-weighted Class Activation Mapping |
| **SHAP** | ✅ | ✅ | SHapley Additive exPlanations |

### Automatic Compatibility System

The `XAICompatibility` class automatically:
- Filters available models based on input type
- Disables incompatible XAI methods (e.g., no Grad-CAM for non-CNN models)
- Prevents user errors by hiding inapplicable options
- Provides clear feedback on method availability

---

## Improvements Over Original Repositories

### Integration & Refactoring
1. **Unified Interface**: Merged two separate projects into single Streamlit application
2. **Modular Architecture**: Separated model loading, XAI computation, and visualization
3. **Smart Caching**: Implemented `@st.cache` to avoid reloading models on each interaction
4. **Error Handling**: Comprehensive try-catch blocks with user-friendly error messages

### Enhanced Functionality
1. **Medical-Specific Models**: Integrated TorchXRayVision for superior medical imaging performance
2. **Model Selection**: Users can choose between medical-pretrained and ImageNet models
3. **Automatic Compatibility Checking**: Dynamic filtering of models and XAI methods
4. **Side-by-Side Comparison**: Compare multiple XAI outputs simultaneously
5. **Session State Management**: Persistent results across page interactions
6. **File Upload Validation**: Checks file format and size before processing

### Technical Optimizations
1. **Medical Imaging Integration**: TorchXRayVision with proper DICOM normalization [-1024, 1024]
2. **Cross-Framework Support**: Seamlessly integrates TensorFlow (audio) and PyTorch (image) models
3. **Memory Management**: Efficient handling of large spectrogram and image tensors
4. **Device Selection**: Automatic CPU/GPU selection for PyTorch models

---


## Known Issues & Limitations

- Large audio files (>10MB) may take longer to process
- GPU acceleration requires CUDA-compatible hardware
- Some XAI methods may be computationally expensive on CPU

---

## Future Enhancements

- Support for additional input formats (CSV, video)
- More pre-trained models (InceptionV3, ResNet50)
- Additional XAI techniques (Integrated Gradients, Attention Maps)
- Batch processing for multiple files
- Export functionality for XAI visualizations

---

## Generative AI Usage Statement

### Declaration of AI Tool Usage

This project used **Generative AI tools** during development. Below is a complete declaration of usage:

#### Tools Used
- **GitHub Copilot** (powered by GPT-4/Claude models)

#### Purpose of Usage

1. **Code Refactoring & Debugging** (70% of AI usage):
   - Fixed NumPy compatibility issues 
   - Resolved TensorFlow/PyTorch version conflicts
   - Debugged SHAP integration
   - Optimized Streamlit caching and session state management

2. **Documentation** (20% of AI usage):
   - Generated this README structure
   - Wrote inline code comments and docstrings
   - Created setup instructions and troubleshooting guides

3. **Code Generation** (10% of AI usage):
   - Implemented `XAICompatibility` filtering system
   - Created comparison tab layout structure
   - Generated error handling wrappers

#### Human Contributions
All core functionality, model integration, XAI implementation logic, interface design decisions, and testing were performed by the team. AI tools were used as assistants for specific technical challenges and documentation.

#### Verification
All AI-generated code suggestions were:
- Reviewed and validated by team members
- Tested thoroughly before integration
- Modified to fit project-specific requirements

---

## References

### Original Repositories
- Deepfake Audio Detection: https://github.com/Aamir-Hullur/Deepfake-Audio-detection-using-XAI
- Lung Cancer Detection: https://github.com/schaudhuri16/LungCancerDetection

---

## Contributors
Claire CUCHE CDOF2
Inès DARDE CDOF2
