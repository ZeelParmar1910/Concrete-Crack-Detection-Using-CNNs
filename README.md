# Concrete Crack Detection Using CNNs

This project uses Convolutional Neural Networks to classify concrete surface images as Cracked or Non-Cracked. Built as part of my Pattern Recognition course project (SS26).

## Dataset

- **Source**: [Surface Crack Detection — Kaggle](https://www.kaggle.com/datasets/arunrk7/surface-crack-detection)
- **Images**: 40,000 concrete surface images
- **Classes**: Positive (Crack), Negative (No Crack)
- **Balance**: Perfectly balanced — 20,000 per class
- **Format**: JPEG, 227×227 pixels (resized to 224×224 for training)

## Models Used

1. **Custom CNN** — 4 convolutional blocks built from scratch (32→64→128→256 filters)
2. **MobileNetV2** — Transfer learning with two-phase fine-tuning
3. **ResNet50** — Transfer learning with two-phase fine-tuning

## Results

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC | Parameters | Inference (ms/img) | Training Time |
|-------|----------|-----------|--------|----------|---------|------------|--------------------|---------------|
| Custom CNN | **99.90%** | 99.87% | 99.93% | 99.90% | 0.9999 | 456K | 20.3 | 17 min 12 s |
| MobileNetV2 | **99.93%** | 99.93% | 99.93% | 99.93% | 1.0000 | 2.6M | 127.1 | 11 min 54 s |
| ResNet50 | 99.48% | 99.73% | 99.23% | 99.48% | 0.9999 | 24.1M | 115.4 | 41 min 0 s |

**Best model: MobileNetV2** (highest accuracy + perfect AUC-ROC, fastest training)  
**Most efficient: Custom CNN** (99.9% accuracy with only 456K parameters, 6.3× faster inference than transfer learning models)

*Results on 6,000-image held-out test set (15% of total data). Training times measured on Kaggle GPU (T4).*

## How to Run

### On Kaggle (recommended)
1. Go to the [Kaggle Notebook](YOUR_KAGGLE_NOTEBOOK_LINK) — update this after uploading
2. Click **Copy & Edit**
3. Go to **Settings → Accelerator → GPU T4 x2** (or P100)
4. Run all cells

### On Google Colab
1. Open [Google Colab](https://colab.research.google.com/) and upload `concrete_crack_detection_cnn.ipynb`
2. Set runtime: **Runtime → Change runtime type → GPU (T4)**
3. Mount your Google Drive or install Kaggle API to download the dataset:
```python
# Install Kaggle API
!pip install kaggle

# Upload your kaggle.json API token, then:
!kaggle datasets download -d arunrk7/surface-crack-detection
!unzip surface-crack-detection.zip -d surface_crack_data/
```
4. Update `data_dir` in the notebook to point to `surface_crack_data/`
5. Run all cells

### Locally
```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/concrete-crack-detection-cnn
cd concrete-crack-detection-cnn

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download dataset from Kaggle
# https://www.kaggle.com/datasets/arunrk7/surface-crack-detection
# Extract to a folder, then update data_dir in the notebook

# 4. Run the notebook
jupyter notebook concrete_crack_detection_cnn.ipynb
```

## Project Structure

```
Phase 2/
├── concrete_crack_detection_cnn.ipynb    ← main Kaggle notebook
├── README.md
├── requirements.txt
├── figures/                              ← saved output figures (after running)
│   ├── sample_images.png
│   ├── class_distribution.png
│   ├── augmentation_examples.png
│   ├── custom_cnn_curves.png
│   ├── mobilenetv2_curves.png
│   ├── resnet50_curves.png
│   ├── custom_cnn_cm.png
│   ├── mobilenetv2_cm.png
│   ├── resnet50_cm.png
│   ├── roc_comparison.png
│   ├── model_comparison.png
│   ├── gradcam_results.png
│   ├── shap_analysis.png
│   ├── workflow_diagram.png
│   ├── error_analysis.png
│   └── pdf/                              ← PDF versions of all figures
│       ├── sample_images.pdf
│       ├── class_distribution.pdf
│       ├── augmentation_examples.pdf
│       ├── custom_cnn_curves.pdf
│       ├── mobilenetv2_curves.pdf
│       ├── resnet50_curves.pdf
│       ├── custom_cnn_cm.pdf
│       ├── mobilenetv2_cm.pdf
│       ├── resnet50_cm.pdf
│       ├── roc_comparison.pdf
│       ├── model_comparison.pdf
│       ├── gradcam_results.pdf
│       ├── shap_analysis.pdf
│       ├── workflow_diagram.pdf
│       └── error_analysis.pdf
└── results/
    └── model_comparison.csv              ← saved after running notebook
```

## Methodology

### Preprocessing
- Resize images: 227×227 → 224×224
- Normalize: 0-255 → 0.0-1.0
- Custom split: 70% train / 15% val / 15% test (stratified)

### Data Augmentation (training only)
- Random rotation ±20°
- Horizontal + vertical flip (valid — cracks appear in any orientation)
- Zoom up to 15%
- Brightness variation 0.8–1.2

### Training
- Batch size: 32, Adam optimizer
- Early stopping (patience=5), ReduceLROnPlateau
- Custom CNN: up to 30 epochs, lr=0.001
- Transfer learning: Phase 1 (frozen, 10 epochs, lr=0.001) → Phase 2 (fine-tune last 20 layers, 15 epochs, lr=1e-5)

### Evaluation
- Accuracy, Precision, Recall, F1-Score, AUC-ROC
- Confusion Matrix, ROC curves
- Grad-CAM visualizations
- Error analysis (false positives and false negatives)
- Inference time comparison

## Requirements

- Python 3.8+
- TensorFlow 2.x
- OpenCV
- NumPy, Pandas, Matplotlib, Seaborn
- scikit-learn
- python-docx (for proposal generation)

## Author

[Your Full Name] — Pattern Recognition Course Project (SS26)

---

*Concrete crack detection dataset: Özgenel, Ç.F., Gönenç Sorguç, A. "Performance Comparison of Pretrained Convolutional Neural Networks on Crack Detection in Buildings", ISARC 2018*
