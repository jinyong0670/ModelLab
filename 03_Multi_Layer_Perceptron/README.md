# Multi-Layer Perceptron (MLP) Implementation

## 1. Overview

This project implements a **Multi-Layer Perceptron (MLP) using NumPy**, without relying on deep learning frameworks such as PyTorch or TensorFlow.

The model is trained to classify handwritten digits (0–9) using the Scikit-learn Digits dataset.

The main goal is to understand the mathematical foundations of neural networks, including forward propagation, backpropagation, gradient descent, and model evaluation.

### Key Features

- Implementation of a two-layer MLP using NumPy
- ReLU activation and Softmax output
- Backpropagation derived using the Chain Rule
- Numerical Gradient Checking
- Full-batch and Mini-batch Gradient Descent
- Validation-based Early Stopping
- Hyperparameter Tuning
- Experiments with Multiple Random Seeds

---

## 2. Dataset

**Dataset:** Scikit-learn Digits Dataset

| Property | Value |
|---|---|
| Samples | 1,797 |
| Image Resolution | 8 × 8 |
| Input Features | 64 |
| Classes | 10 (digits 0–9) |
| Pixel Value Range | 0–16 |

### Preprocessing

Pixel values are normalized to the range [0, 1]:

\[
X_{\text{normalized}} = \frac{X}{16}
\]

For validation experiments, the dataset is divided into:

| Split | Samples |
|---|---:|
| Training | 1,149 |
| Validation | 288 |
| Test | 360 |

Stratified splitting is used to preserve the class distribution.

---

## 3. Model Architecture

The final MLP has the following architecture:

**Input (64) → Hidden (64, ReLU) → Output (10, Softmax)**

### Forward Propagation

\[
Z_1 = XW_1 + b_1
\]

\[
A_1 = \operatorname{ReLU}(Z_1)
\]

\[
Z_2 = A_1W_2 + b_2
\]

\[
\hat{Y} = \operatorname{Softmax}(Z_2)
\]

### Loss Function

Categorical Cross Entropy is used for multi-class classification:

\[
L = -\frac{1}{m}\sum_{i=1}^{m}\sum_{c=1}^{10} y_{ic}\log(\hat{y}_{ic})
\]

### Parameter Initialization

- First layer: He Initialization
- Output layer: Random Normal Initialization scaled by input dimension
- Biases: Initialized to zero

---

## 4. Backpropagation

All gradients are implemented manually using the Chain Rule.

### Output Layer

\[
dZ_2 = \hat{Y} - Y
\]

\[
dW_2 = \frac{1}{m}A_1^T dZ_2
\]

\[
db_2 = \frac{1}{m}\sum_i dZ_{2,i}
\]

### Hidden Layer

\[
dA_1 = dZ_2 W_2^T
\]

\[
dZ_1 = dA_1 \odot \operatorname{ReLU}'(Z_1)
\]

\[
dW_1 = \frac{1}{m}X^T dZ_1
\]

\[
db_1 = \frac{1}{m}\sum_i dZ_{1,i}
\]

The parameters are updated using Gradient Descent:

\[
\theta \leftarrow \theta - \eta \nabla_\theta L
\]

---

## 5. Gradient Checking

To verify the correctness of backpropagation, analytical gradients are compared with numerical gradients using Central Difference:

\[
g_{\text{numerical}} \approx
\frac{L(\theta+\epsilon)-L(\theta-\epsilon)}{2\epsilon}
\]

with \(\epsilon = 10^{-5}\).

### Results

| Parameter | Relative Error |
|---|---:|
| W1[10,0] | 2.34e-10 |
| W1[20,0] | 0.00 |
| W1[30,5] | 1.66e-09 |
| W1[40,10] | 0.00 |

The selected parameters showed close agreement between numerical and analytical gradients. Two checked positions had zero gradients; the other two had very small relative errors.

---

## 6. Training Experiments

### Full-batch vs Mini-batch

| Method | Epochs | Test Accuracy |
|---|---:|---:|
| Full-batch Gradient Descent | 1,000 | 96.39% |
| Mini-batch Gradient Descent (Batch Size 32) | 100 | 97.50% |

These preliminary experiments used a different training setup from the later validation experiments, so the results are not a controlled comparison of optimization methods.

### Early Stopping

Early Stopping monitors validation loss and restores the parameters from the epoch with the lowest observed validation loss.

- Maximum Epochs: 150
- Patience: 15
- Best Epoch: 96
- Stopping Epoch: 111
- Best Validation Loss: 0.080993

---

## 7. Hyperparameter Tuning

### Hidden Size Comparison

| Hidden Size | Best Validation Loss | Validation Accuracy |
|---|---:|---:|
| 32 | 0.093518 | 97.92% |
| 64 | **0.070696** | 97.57% |
| 128 | 0.080993 | 97.57% |
| 256 | 0.078126 | 97.57% |

### Learning Rate Comparison

| Learning Rate | Best Validation Loss | Validation Accuracy |
|---|---:|---:|
| 0.01 | 0.150797 | 97.22% |
| 0.05 | 0.080737 | 97.92% |
| 0.1 | **0.070696** | 97.57% |

### Batch Size Comparison

| Batch Size | Best Validation Loss | Validation Accuracy |
|---|---:|---:|
| 16 | **0.068580** | 97.92% |
| 32 | 0.070696 | 97.57% |
| 64 | 0.080853 | 97.92% |

Validation loss was used as the primary model-selection criterion.

---

## 8. Multiple Random Seeds

To evaluate the sensitivity to initialization and data shuffling, experiments were repeated using seeds 42, 123, and 2026.

| Batch Size | Mean Val Loss ± Std | Mean Val Accuracy ± Std |
|---|---|---|
| 16 | **0.071411 ± 0.004064** | 97.57% ± 0.49%p |
| 32 | 0.075888 ± 0.005840 | 97.57% ± 0.57%p |

Batch Size 16 achieved a lower mean validation loss in this limited experiment. More random seeds would be required for a stronger conclusion.

---

## 9. Final Evaluation

The final configuration was selected based on validation loss.

| Hyperparameter | Value |
|---|---|
| Input Size | 64 |
| Hidden Size | 64 |
| Output Size | 10 |
| Learning Rate | 0.1 |
| Batch Size | 16 |
| Patience | 15 |
| Random Seed | 42 |
| Best Epoch | 75 |
| Best Validation Loss | 0.068580 |

### Final Test Result

**Test Accuracy: 96.94% (349/360 correct predictions)**

The selected model achieved 96.94% accuracy on the held-out test set.

---

## 10. Project Structure

```text
03_Multi_Layer_Perceptron/
├── model.py
├── main.py
├── validation.py
├── hyperparameter.py
├── final_eval.py
└── README.md
```

| File | Description |
|---|---|
| `model.py` | MLP layers, activation functions, loss, backpropagation, and parameter updates |
| `main.py` | Baseline training, mini-batch training, and gradient checking |
| `validation.py` | Validation monitoring and early stopping |
| `hyperparameter.py` | Hyperparameter tuning and multiple-seed experiments |
| `final_eval.py` | Final model training and test evaluation |

---

## 11. How to Run

Run the commands from the ModelLab repository root.

Install dependencies:

```bash
pip install numpy matplotlib scikit-learn
```

Run the main experiments:

```bash
python 03_Multi_Layer_Perceptron/main.py
```

Run validation experiments:

```bash
python 03_Multi_Layer_Perceptron/validation.py
```

Run hyperparameter experiments:

```bash
python 03_Multi_Layer_Perceptron/hyperparameter.py
```

Run the final evaluation:

```bash
python 03_Multi_Layer_Perceptron/final_eval.py
```

---

## 12. Key Takeaways

Through this project, I learned:

- How fully connected neural networks transform input features through multiple layers.
- How ReLU introduces nonlinearity into a neural network.
- How to derive and implement backpropagation using the Chain Rule.
- How numerical gradient checking can help verify custom gradient implementations.
- How mini-batch training changes the number and frequency of parameter updates.
- How validation loss and early stopping support model selection.
- Why hyperparameter tuning and repeated experiments are important for reproducibility.
- Why increased model complexity does not necessarily improve generalization.

### Future Improvements

- Extend the MLP to multiple hidden layers.
- Compare different activation functions and optimizers.
- Implement Adam Optimization from scratch.
- Compare MLP with CNN on handwritten digit classification.