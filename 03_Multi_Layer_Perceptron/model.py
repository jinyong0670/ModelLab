import numpy as np


# ==================================
# Parameter Initialization
# ==================================

def initialize_parameters(input_size, hidden_size, output_size):
    #He initialization(because of ReLU activation func)
    W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2 / input_size)

    b1 = np.zeros((1, hidden_size))

    W2 = np.random.randn(hidden_size, output_size) * np.sqrt(1 / hidden_size)

    b2 = np.zeros((1, output_size))

    return W1, b1, W2, b2


# ==================================
# Activation Functions
# ==================================

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)

    exp_z = np.exp(z)

    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


# ==================================
# Forward Propagation
# ==================================

def forward(X, W1, b1, W2, b2):

    # Input -> Hidden Layer
    Z1 = X @ W1 + b1

    # ReLU Activation
    A1 = relu(Z1)

    # Hidden -> Output Layer
    Z2 = A1 @ W2 + b2

    # Softmax Activation
    pred = softmax(Z2)

    return pred, (Z1, A1, Z2)


# ==================================
# One-hot Encoding
# ==================================

def one_hot_encoding(y, num_classes):
    m = len(y)

    one_hot = np.zeros((m, num_classes))
    one_hot[np.arange(m), y] = 1

    return one_hot


# ==================================
# Loss Function
# ==================================

def categorical_cross_entropy(y, y_pred):
    eps = 1e-8
    y_pred = np.clip(y_pred, eps, 1 - eps)

    loss = -y * np.log(y_pred)

    return np.mean(np.sum(loss, axis=1))

# ==================================
# Backpropagation
# ==================================

def backward(X, y, pred, cache, W2):

    Z1, A1, Z2 = cache
    m = X.shape[0]

    # Output Layer
    dZ2 = pred - y

    dW2 = A1.T @ dZ2 / m
    db2 = np.mean(dZ2, axis = 0, keepdims=True)

    # Hidden Layer
    dA1 = dZ2 @ W2.T

    dZ1 = dA1 * relu_derivative(Z1)

    dW1 = X.T @ dZ1 / m
    db1 = np.mean(dZ1, axis = 0, keepdims=True)

    return dW1, db1, dW2, db2


# ==================================
# Gradient Descent
# ==================================

def update_parameters(W1, b1, W2, b2,
                      dW1, db1, dW2, db2,
                      learning_rate):

    W1 = W1 - learning_rate * dW1

    b1 = b1 - learning_rate * db1

    W2 = W2 - learning_rate * dW2

    b2 = b2 - learning_rate * db2

    return W1, b1, W2, b2