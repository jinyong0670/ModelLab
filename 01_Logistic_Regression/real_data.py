import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# ==================================
# Step 1. Load Dataset
# ==================================

data = load_breast_cancer()

X = data.data
y = data.target.reshape(-1, 1)

print("X shape:", X.shape)
print("y shape:", y.shape)


# ==================================
# Step 2. Train/Test Split
# ==================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)


# ==================================
# Step 3. Feature Scaling
# ==================================

mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)

std = np.where(std == 0, 1, std)

X_train_scaled = (X_train - mean) / std
X_test_scaled = (X_test - mean) / std


# ==================================
# Step 4. Model Functions
# ==================================

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

    dz = y_pred - y #BCE 기준

    dW = (X.T @ dz) / m

    db = np.mean(dz)

    return dW, db

def update_parameters(W, b, dW, db, learning_rate):
    W = W - learning_rate * dW
    b = b - learning_rate * db

    return W, b

# ==================================
# Step 5. Initialize Parameters
# ==================================

np.random.seed(42)

num_features = X_train_scaled.shape[1]

W = np.random.randn(num_features, 1) * 0.01
b = np.zeros((1,))

print("W shape:", W.shape)
print("b shape:", b.shape)


# ==================================
# Step 6. Training Loop
# ==================================

learning_rate = 0.1
epochs = 2000

loss_history = []

for epoch in range(epochs):

    # Forward Propagation
    pred = forward(X_train_scaled, W, b)

    # Calculate Loss
    loss = binary_cross_entropy(y_train, pred)
    loss_history.append(loss)

    # Backpropagation
    dW, db = backward(X_train_scaled, y_train, pred)

    # Gradient Descent
    W, b = update_parameters(W, b, dW, db, learning_rate)

    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss = {loss:.6f}")

# ==================================
# Step 7. Evaluation
# ==================================

train_probs = forward(X_train_scaled, W, b)
train_labels = (train_probs >= 0.5).astype(int)

train_accuracy = np.mean(train_labels == y_train)

# Test predictions
test_probs = forward(X_test_scaled, W, b)
test_labels = (test_probs >= 0.5).astype(int)

test_accuracy = np.mean(test_labels == y_test)

print("\n===== Final Results =====")
print(f"Train Accuracy: {train_accuracy * 100:.2f}%")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")


# ==================================
# Step 8. Confusion Matrix
# ==================================

# Class 0 (malignant) is the positive class

# True Positive:
# Actual = 0, Predicted = 0
TP = np.sum((y_test == 0) & (test_labels == 0))

# True Negative:
# Actual = 1, Predicted = 1
TN = np.sum((y_test == 1) & (test_labels == 1))

# False Positive:
# Actual = 1, Predicted = 0
FP = np.sum((y_test == 0) & (test_labels == 1))

# False Negative:
# Actual = 0, Predicted = 1
FN = np.sum((y_test == 1) & (test_labels == 0))

print("\n===== Confusion Matrix =====")
print("TP:", TP)
print("TN:", TN)
print("FP:", FP)
print("FN:", FN)
