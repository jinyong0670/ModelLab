# Multi-Layer Perceptron (MLP)

## 1. Overview

This project implements a **Multi-Layer Perceptron (MLP) using NumPy**, without relying on deep learning frameworks such as PyTorch or TensorFlow.

The model is trained to classify handwritten digits (0–9) using the Scikit-learn Digits dataset.

The main goal is to understand the mathematical foundations of neural networks, including forward propagation, backpropagation, gradient descent, and model evaluation.

### Key Features

- Implementation of a two-layer MLP using NumPy
- ReLU activation and Softmax output
- Backpropagation implemented using the Chain Rule
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

$$
X_{\text{normalized}} = \frac{X}{16}
$$

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

The input is transformed through two fully connected layers.

$$
Z_1 = XW_1 + b_1
$$

$$
A_1 = \operatorname{ReLU}(Z_1)
$$

$$
Z_2 = A_1W_2 + b_2
$$

$$
\hat{Y} = \operatorname{Softmax}(Z_2)
$$

### Activation Functions

ReLU introduces nonlinearity into the hidden layer.

$$
\operatorname{ReLU}(x) = \max(0,x)
$$

Softmax converts the output logits into class probabilities.

$$
\operatorname{Softmax}(z)_i =
\frac{e^{z_i}}{\sum_{j=1}^{C}e^{z_j}}
$$

### Loss Function

Categorical Cross Entropy is used for multi-class classification:

$$
L = -\frac{1}{m}\sum_{i=1}^{m}\sum_{c=1}^{10} y_{ic}\log(\hat{y}_{ic})
$$

Here, \(m\) represents the number of training samples, and \(y_{ic}\) denotes the one-hot encoded target.

### Parameter Initialization

- **First layer:** He Initialization
- **Output layer:** Random Normal Initialization scaled by the input dimension
- **Biases:** Initialized to zero

For the first layer:

$$
W_1 \sim \mathcal{N}\left(0,\frac{2}{n_{\text{in}}}\right)
$$

For the output layer:

$$
W_2 \sim \mathcal{N}\left(0,\frac{1}{n_{\text{hidden}}}\right)
$$

The second argument of \(\mathcal{N}\) represents the variance.

---

## 4. Backpropagation

All gradients are implemented manually using the Chain Rule.

### Output Layer

The gradient of Softmax combined with Categorical Cross Entropy simplifies to:

$$
dZ_2 = \hat{Y} - Y
$$

The gradients with respect to the output layer parameters are:

$$
dW_2 = \frac{1}{m}A_1^T dZ_2
$$

$$
db_2 = \frac{1}{m}\sum_{i=1}^{m} dZ_{2,i}
$$

### Hidden Layer

The gradient is propagated backward through the second weight matrix.

$$
dA_1 = dZ_2 W_2^T
$$

The element-wise derivative of ReLU is applied:

$$
dZ_1 = dA_1 \odot \operatorname{ReLU}'(Z_1)
$$

where \(\odot\) denotes element-wise multiplication.

The gradients of the first layer are:

$$
dW_1 = \frac{1}{m}X^T dZ_1
$$

$$
db_1 = \frac{1}{m}\sum_{i=1}^{m}dZ_{1,i}
$$

### Gradient Descent

All parameters are updated using Gradient Descent:

$$
\theta \leftarrow \theta - \eta \nabla_\theta L
$$

where \(\eta\) is the learning rate.

---

## 5. Gradient Checking

To verify the correctness of backpropagation, analytical gradients are compared with numerical gradients using the Central Difference method.

$$
g_{\text{numerical}}
\approx
\frac{L(\theta+\epsilon)-L(\theta-\epsilon)}
{2\epsilon}
$$

The perturbation value is:

$$
\epsilon = 10^{-5}
$$

### Relative Error

The relative difference between analytical and numerical gradients is calculated as:

$$
\text{Relative Error}
=
\frac{
|g_{\text{numerical}}-g_{\text{analytical}}|
}{
\max(10^{-8},|g_{\text{numerical}}|+|g_{\text{analytical}}|)
}
$$

### Results

| Parameter | Numerical Gradient | Analytical Gradient | Relative Error |
|---|---:|---:|---:|
| W1[10,0] | 0.0123622822 | 0.0123622822 | 2.34e-10 |
| W1[20,0] | 0.0000000000 | 0.0000000000 | 0.00 |
| W1[30,5] | 0.0029052347 | 0.0029052347 | 1.66e-09 |
| W1[40,10] | 0.0000000000 | 0.0000000000 | 0.00 |

The nonzero gradients closely matched their numerical approximations.

This provides strong evidence that the implemented backpropagation is correct for the tested parameters.

---

## 6. Training Experiments

### Full-batch vs Mini-batch Gradient Descent

Two optimization strategies were implemented and evaluated.

| Method | Epochs | Test Accuracy |
|---|---:|---:|
| Full-batch Gradient Descent | 1,000 | 96.39% |
| Mini-batch Gradient Descent (Batch Size 32) | 100 | 97.50% |

Mini-batch Gradient Descent achieved higher test accuracy in this preliminary experiment.

However, the experiments used different numbers of parameter updates, so this result alone does not establish that mini-batch training is always superior.

### Validation and Early Stopping

The training dataset was further divided into training and validation subsets.

Validation loss was monitored after each epoch, and training was terminated when validation loss did not improve for a specified number of epochs.

**Early Stopping Experiment:**

| Property | Value |
|---|---:|
| Maximum Epochs | 150 |
| Patience | 15 |
| Best Epoch (0-based) | 96 |
| Stopping Epoch (0-based) | 111 |
| Best Validation Loss | 0.080993 |
| Test Accuracy | 97.50% |

The model parameters from the best validation epoch were restored before final evaluation.

---

## 7. Hyperparameter Tuning

Hyperparameter tuning was conducted using the validation dataset.

The primary model selection criterion was **Validation Loss**.

### Hidden Size Comparison

| Hidden Size | Best Epoch | Best Validation Loss | Validation Accuracy |
|---|---:|---:|---:|
| 32 | 94 | 0.093518 | 97.92% |
| 64 | 142 | **0.070696** | 97.57% |
| 128 | 96 | 0.080993 | 97.57% |
| 256 | 146 | 0.078126 | 97.57% |

Hidden Size 64 achieved the lowest validation loss in this experiment.

### Learning Rate Comparison

Hidden Size was fixed at 64.

| Learning Rate | Best Epoch | Best Validation Loss | Validation Accuracy |
|---|---:|---:|---:|
| 0.01 | 149 | 0.150797 | 97.22% |
| 0.05 | 148 | 0.080737 | 97.92% |
| 0.1 | 142 | **0.070696** | 97.57% |

Learning Rate 0.1 achieved the lowest validation loss within the tested range and training budget.

### Batch Size Comparison

Hidden Size and Learning Rate were fixed at 64 and 0.1, respectively.

| Batch Size | Best Epoch | Best Validation Loss | Validation Accuracy |
|---|---:|---:|---:|
| 16 | 75 | **0.068580** | 97.92% |
| 32 | 142 | 0.070696 | 97.57% |
| 64 | 148 | 0.080853 | 97.92% |

Batch Size 16 achieved the lowest validation loss.

Increasing model size or batch size did not consistently improve performance.

---

## 8. Multiple Random Seeds

To evaluate sensitivity to weight initialization and data shuffling, the experiments were repeated using three random seeds:

**42, 123, and 2026**

### Individual Results

| Batch Size | Seed | Best Validation Loss | Validation Accuracy |
|---|---:|---:|---:|
| 16 | 42 | 0.068580 | 97.92% |
| 16 | 123 | 0.077159 | 96.88% |
| 16 | 2026 | 0.068496 | 97.92% |
| 32 | 42 | 0.070696 | 97.57% |
| 32 | 123 | 0.084047 | 96.88% |
| 32 | 2026 | 0.072922 | 98.26% |

### Mean and Standard Deviation

| Batch Size | Mean Val Loss ± Std | Mean Val Accuracy ± Std |
|---|---|---|
| 16 | **0.071411 ± 0.004064** | 97.57% ± 0.49%p |
| 32 | 0.075888 ± 0.005840 | 97.57% ± 0.57%p |

Batch Size 16 achieved a lower mean validation loss across the three random seeds.

However, more repetitions would be needed for a statistically reliable comparison.

---

## 9. Final Model Evaluation

The final hyperparameters were selected based on validation loss.

### Final Configuration

| Hyperparameter | Value |
|---|---|
| Input Size | 64 |
| Hidden Size | 64 |
| Output Size | 10 |
| Hidden Activation | ReLU |
| Output Activation | Softmax |
| Loss | Categorical Cross Entropy |
| Optimizer | Mini-batch Gradient Descent |
| Learning Rate | 0.1 |
| Batch Size | 16 |
| Maximum Epochs | 150 |
| Patience | 15 |
| Random Seed | 42 |

### Final Training Results

| Metric | Result |
|---|---:|
| Best Epoch (0-based) | 75 |
| Early Stopping Epoch (0-based) | 90 |
| Best Validation Loss | 0.068580 |
| Final Test Accuracy | **96.94%** |
| Correct Predictions | 349 / 360 |

The final model achieved **96.94% accuracy** on the held-out test set.

Hyperparameters were selected using validation performance rather than test accuracy.

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
| `model.py` | Neural network operations, activation functions, loss, backpropagation, and gradient descent |
| `main.py` | Baseline training, mini-batch training, and gradient checking |
| `validation.py` | Training with validation monitoring and early stopping |
| `hyperparameter.py` | Hyperparameter tuning and multiple random seed experiments |
| `final_eval.py` | Final model training and test evaluation |

---

## 11. How to Run

Run all commands from the root directory of the ModelLab repository.

### Install Dependencies

```bash
pip install numpy matplotlib scikit-learn
```

### Basic MLP Training

```bash
python 03_Multi_Layer_Perceptron/main.py
```

### Validation and Early Stopping

```bash
python 03_Multi_Layer_Perceptron/validation.py
```

### Hyperparameter Tuning

```bash
python 03_Multi_Layer_Perceptron/hyperparameter.py
```

### Final Model Evaluation

```bash
python 03_Multi_Layer_Perceptron/final_eval.py
```

---

## 12. Key Takeaways

Through this project, I learned:

- How fully connected neural networks transform input features through multiple layers.
- Why nonlinear activation functions such as ReLU are essential for learning nonlinear patterns.
- How to derive and implement backpropagation using the Chain Rule.
- How numerical gradient checking verifies manually implemented gradients.
- How Full-batch and Mini-batch Gradient Descent differ in their update frequency.
- How validation data and early stopping support model selection.
- How hyperparameters affect convergence and generalization.
- Why multiple random seeds are important for reproducible machine learning experiments.
- Why a more complex model does not necessarily achieve better generalization performance.

### Future Improvements

- Extend the MLP to multiple hidden layers.
- Implement different activation functions.
- Implement Momentum and Adam optimizers from scratch.
- Compare MLP and CNN on handwritten digit classification.
- Evaluate model performance across more random seeds.

---

## 13. Conclusion

This project provided hands-on experience in implementing and evaluating a neural network entirely with NumPy.

By manually implementing forward propagation, backpropagation, and gradient descent, I gained a deeper understanding of how neural networks learn.

Additionally, numerical gradient checking, early stopping, hyperparameter tuning, and repeated experiments helped establish a more systematic approach to model development and evaluation.

The final MLP achieved **96.94% test accuracy** on the Scikit-learn Digits dataset.