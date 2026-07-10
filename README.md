# Cat vs Dog Image Classification using CNN and MobileNetV2

## Course Information

**Course:** Introduction to Machine Learning

**Project Type:** Mini Project

**Team Members:** 3

---

## Project Overview

This project aims to classify images of cats and dogs using deep learning techniques. We developed and evaluated multiple image classification models, beginning with a basic Convolutional Neural Network (CNN) and progressively improving performance through data augmentation and transfer learning with MobileNetV2.

The project demonstrates the complete machine learning workflow, including data preprocessing, model development, evaluation, and prediction on unseen images.

---

## Objectives

- Build an image classification model using CNN.
- Improve model performance through data augmentation.
- Apply transfer learning using MobileNetV2.
- Compare different deep learning approaches.
- Evaluate the models using standard classification metrics.

---

## Dataset

The dataset consists of labeled images divided into two classes:

- Cat
- Dog

Each image is resized before training.

---

## Technologies Used

- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Matplotlib
- Scikit-learn

---

## Project Workflow

1. Load the image dataset.
2. Resize and normalize images.
3. Split the dataset into training and testing sets.
4. Train a baseline CNN model.
5. Improve the CNN using data augmentation.
6. Implement transfer learning with MobileNetV2.
7. Fine-tune the pretrained model.
8. Evaluate model performance.
9. Perform predictions on unseen images.

---

## Model Development

### Phase 1 — Baseline CNN

The first model was a simple Convolutional Neural Network consisting of:

- Convolution layers
- Max Pooling layers
- Fully Connected layers
- Sigmoid output layer

This model served as the baseline for comparison.

---

### Phase 2 — Improved CNN

The second version enhanced the baseline model by introducing:

- Data augmentation
- Dropout layer
- Additional convolution layer
- Model checkpointing

These improvements helped reduce overfitting and improve generalization.

---

### Phase 3 — Transfer Learning with MobileNetV2

To further improve classification accuracy, a pretrained MobileNetV2 model was used.

The ImageNet pretrained weights were utilized while replacing the classifier with a custom classification head.

The pretrained feature extractor was initially frozen before fine-tuning the final layers.

---

### Phase 4 — Fine-Tuning

The final model included:

- MobileNetV2
- Batch Normalization
- Dropout
- Fine-tuning of the final pretrained layers
- Early Stopping
- Model Checkpointing

This produced the best overall performance.

---

## Evaluation

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)
- Confusion Matrix
- Classification Report
- ROC Curve
- Area Under the Curve (AUC)

---

## Output

The project generates:

- Trained model (.h5)
- Accuracy curves
- Loss curves
- Confusion Matrix
- ROC Curve
- Classification Report
- Demo predictions on unseen images

---

## Repository Structure

```
cat-dog-classification/
│
├── Cat_Dog_Classification.ipynb
├── README.md
├── requirements.txt
├── .gitignore
└── dataset/
```

---

## How to Run

Clone the repository:

```bash
git clone https://github.com/solinesong/cat-dog-classification.git
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Launch Jupyter Notebook and run all cells.

---

## Team Project

This project was completed as a mini project for the Introduction to Machine Learning course by a team of three members.

### My Contributions

- Assisted with data preprocessing.
- Developed and improved CNN architectures.
- Implemented transfer learning using MobileNetV2.
- Evaluated model performance using multiple metrics.
- Generated visualizations and prediction results.
- Participated in debugging, testing, and project documentation.

---

## Future Improvements

- Increase dataset size.
- Experiment with EfficientNet and ResNet.
- Hyperparameter optimization.
- Multi-class animal classification.
- Deploy the model as a web application.