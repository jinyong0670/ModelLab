# Softmax Regression from Scratch

## Overview
Implementation of multi-class Softmax Regression using NumPy, including manual forward propagation, backpropagation, and gradient descent.

## Dataset
- Iris Dataset
- 150 samples
- 4 features
- 3 classes: Setosa, Versicolor, Virginica
- Train/Test Split: 80/20

## Implementation
- Softmax Activation
- One-hot Encoding
- Categorical Cross Entropy
- Forward Propagation
- Backpropagation
- Gradient Descent
- Multi-class Confusion Matrix
- Precision, Recall, F1-score
- Macro and Micro Averaging

## Training Configuration
- Learning Rate: 0.1
- Epochs: 2000
- Feature Scaling: Standardization

## Results
Record the final values from `main.py`:
- Train Accuracy: 97.50%
- Test Accuracy: 96.67%
- Macro F1: 0.9666
- Micro F1: 0.9667

## Key Learnings
- Softmax converts logits into class probabilities.
- Cross Entropy measures the discrepancy between the predicted probabilities and the target distribution.
- Softmax Regression generalizes binary logistic regression to multiple classes.
- Micro F1 equals accuracy for single-label multi-class classification.