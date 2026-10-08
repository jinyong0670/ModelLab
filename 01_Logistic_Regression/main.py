
import numpy as np

# Training data
X = np.array([
    [1.0, 1.0],
    [1.0, 2.0],
    [2.0, 1.0],
    [3.0, 3.0],
    [3.0, 4.0],
    [4.0, 3.0]
])

y = np.array([
    [0],
    [0],
    [0],
    [1],
    [1],
    [1]
])

# Initialize parameters
np.random.seed(42)
W = np.random.randn(2, 1) * 0.01
b = np.zeros((1,))

# TODO 1
def sigmoid(z):
    return 1/ (1 + np.exp(-z))

# TODO 2
def forward(X, W, b):
    return sigmoid(X @ W + b)

print("X shape:", X.shape)
print("W shape:", W.shape)

predictions = forward(X, W, b)
print("Predictions:\n", predictions)