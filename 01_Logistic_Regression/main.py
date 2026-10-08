import numpy as np
import matplotlib.pyplot as plt

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

def sigmoid(z):
    return 1/ (1 + np.exp(-z))

def forward(X, W, b):
    return sigmoid(X @ W + b)

def binary_cross_entropy(y, y_pred):
    eps = 1e-8
    y_pred = np.clip(y_pred, eps, 1 - eps) #if y_pred = 0이면 log(0) 연산 -> 방지하기 위해 아주 작은 값으로 제한
    loss = -(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred)) #loss 계산식

    return np.mean(loss) #loss들의 평균을 return

def backward(X, y, y_pred):
    m = X.shape[0]

    dz = y_pred - y

    dW = (X.T @ dz) / m

    db = np.mean(dz)

    return dW, db

def update_parameters(W, b, dW, db, learning_rate):
    W = W - learning_rate * dW
    b = b - learning_rate * db

    return W, b


learning_rate = 0.1
epochs = 1000

loss_history = []

for epoch in range(epochs):

    # Forward Propagation
    predictions = forward(X, W, b)

    # Calculate Loss
    loss = binary_cross_entropy(y, predictions)
    loss_history.append(loss)

    # Backpropagation
    dW, db = backward(X, y, predictions)

    # Gradient Descent
    W, b = update_parameters(W, b, dW, db, learning_rate)

    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss = {loss:.6f}")

final_predictions = forward(X, W, b)

predicted_labels = (final_predictions >= 0.5).astype(int)

accuracy = np.mean(predicted_labels == y)

print("\nFinal Predictions:")
print(final_predictions)

print("\nTraining Accuracy:", accuracy * 100, "%")


# Plot Loss Curve
plt.figure(figsize=(8, 5))

plt.plot(loss_history, label="Training Loss")

plt.xlabel("Epoch")
plt.ylabel("Binary Cross Entropy Loss")
plt.title("Logistic Regression Training Loss")

plt.legend()
plt.grid(True)

plt.savefig(
    "01_Logistic_Regression/loss_curve.png",
    dpi=150,
    bbox_inches="tight"
)

plt.show()


# Separate samples by class
class_0 = X[y.flatten() == 0]
class_1 = X[y.flatten() == 1]

plt.figure(figsize=(7, 6))

plt.scatter(
    class_0[:, 0], class_0[:, 1],
    label="Class 0", s=100
)

plt.scatter(
    class_1[:, 0], class_1[:, 1],
    label="Class 1", s=100
)

# Generate decision boundary
x1_values = np.linspace(0, 5, 100)

w1 = W[0, 0]
w2 = W[1, 0]
bias = b.item()

x2_values = -(w1 * x1_values + b) / w2

plt.plot(
    x1_values, x2_values,
    "k--", label="Decision Boundary"
)

plt.xlim(0, 5)
plt.ylim(0, 5)

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Logistic Regression Decision Boundary")

plt.legend()
plt.grid(True)

plt.savefig(
    "01_Logistic_Regression/decision_boundary.png",
    dpi=150,
    bbox_inches="tight"
)

plt.show()
