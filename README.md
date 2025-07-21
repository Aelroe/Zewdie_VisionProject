![](UTA-DataScience-Logo.png)

# American Sign Language Vision Classification

* **One Sentence Summary**: This repository implements multiple deep learning models (Baseline CNN, Augmented CNN, MobileNetV2, EfficientNetB0) to classify hand gestures in American Sign Language using image data from a Kaggle ASL dataset.

## Overview

* The challenge is to correctly classify images of hand signs representing the letters E, A, R, I, and O.
* Our approach involves training and comparing four models: a Baseline CNN, an Augmented CNN (with image augmentation), and two pre-trained transfer learning models: MobileNetV2 and EfficientNetB0.
* Among all models, **EfficientNetB0** achieved the best overall performance across metrics such as accuracy and ROC-AUC.

## Summary of Workdone

### Data

* **Type**: RGB Images of hand signs for ASL letters.
* **Size**: ~3,250 total images across 5 classes.
* **Split**: Training (80%), Validation (10%), Test (10%).

#### Preprocessing / Clean up

* All images resized to 224x224 pixels.
* Pixel values scaled to range [0, 1].
* Augmentation (rotation, zoom, flips) was applied to training data for the Augmented CNN.

#### Data Visualization

* Sample batches of images were displayed.
* Class distribution was verified to be balanced across the 5 letter classes.

### Problem Formulation

* **Input**: Image (224x224x3)
* **Output**: One-hot vector representing 1 of 5 ASL letter classes.
* **Models**:
  * Baseline CNN
  * Augmented CNN
  * MobileNetV2 (pretrained on ImageNet)
  * EfficientNetB0 (pretrained on ImageNet)
* **Loss**: Categorical Crossentropy
* **Optimizer**: Adam
* **Metrics**: Accuracy, ROC-AUC, Precision, Recall, F1-score

### Training

* Trained using TensorFlow/Keras in Google Colab.
* Each model trained for up to 25 epochs with EarlyStopping and ModelCheckpoint.
* Training stopped when validation loss stopped improving.
* Loss and accuracy plots were saved for each model.

### Performance Comparison

| Model          | Accuracy | Precision | Recall | F1 Score |
|----------------|----------|-----------|--------|----------|
| Baseline CNN   | ~85%     | Moderate  | Moderate | Moderate |
| Augmented CNN  | ~88%     | Higher    | Higher  | Higher   |
| MobileNetV2    | ~92%     | Very High | Very High | Very High |
| EfficientNetB0 | **94%**  | **Best**  | **Best** | **Best** |

#### ROC Curve

![](roc_curves.png)

#### Confusion Matrix (Augmented Model)

![](confusion_matrix_augmented.png)

### Conclusions

* Transfer learning with EfficientNetB0 outperformed all other models.
* Augmentation helped boost the baseline model significantly.
* Deeper pre-trained models generalize better on unseen ASL gestures.

### Future Work

* Expand dataset to include all 26 ASL letters.
* Use Grad-CAM to visualize which parts of the image the model focuses on.
* Deploy the model as a real-time web app using TensorFlow.js or Streamlit.

## How to reproduce results

1. Clone the repository and open `VisionProject_Zewdie.ipynb`.
2. Make sure to upload the dataset into a directory like `asl_alphabet_train/`.
3. Run the notebook in Google Colab or locally with TensorFlow installed.

### Overview of files in repository

* `VisionProject_Zewdie.ipynb`: Full project code including model training, evaluation, and visualization.
* `roc_curves.png`: Comparison of model ROC curves.
* `confusion_matrix_augmented.png`: Confusion matrix for the Augmented CNN.
* `UTA-DataScience-Logo.png`: Logo for report.
* `README.md`: Project documentation.

### Software Setup

* Python 3.10+
* TensorFlow
* NumPy
* Matplotlib
* scikit-learn

Install with:
```bash
pip install tensorflow numpy matplotlib scikit-learn
