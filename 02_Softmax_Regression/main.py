import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ==================================
# Step 1. Load Dataset
# ==================================

data = load_iris()

X = data.data
y = data.target

print("X shape:", X.shape)
print("y shape:", y.shape)
print("Classes:", data.target_names)

# ==================================
# Step 2. Train/Test Split
# ==================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==================================
# Step 3. Feature Scaling
# ==================================

mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)

std = np.where(std == 0, 1, std)

X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

print("Train:", X_train.shape)
print("Test:", X_test.shape)

# ==================================
# Step 4. Softmax Activation
# ==================================

def softmax(z):
    z = z - np.max(z, axis = 1, keepdims = True) #np.exp 할 때 너무 크면 overflow 가능 --> 최댓값 빼주기
    exp_z = np.exp(z)
    prob = exp_z / np.sum(exp_z, axis = 1, keepdims = True)

    return prob

# ==================================
# Step 5. One-hot Encoding
# ==================================

def one_hot_encoding(y, num_classes):

    m = len(y)

    # Create a zero matrix
    one_hot = np.zeros((m, num_classes))

    #1: for문 이용한 구현
    #for i in range(m):
    #    label = y[i]
    #    one_hot[i, label] = 1

    #2: numpy array 이용
    one_hot[np.arange(m), y] = 1

    return one_hot


# ==================================
# Step 6. Categorical Cross Entropy
# ==================================

def categorical_cross_entropy(y, y_pred):

    eps = 1e-8
    y_pred = np.clip(y_pred, eps, 1 - eps)

    loss = - y * np.log(y_pred)

    mean_loss = np.mean(np.sum(loss, axis = 1))

    return mean_loss


# ==================================
# Step 7. Initialize Parameters
# ==================================

np.random.seed(42)

num_features = X_train.shape[1]
num_classes = 3

W = np.random.randn(num_features, num_classes) * 0.01
b = np.zeros((1, num_classes))


# ==================================
# Step 8. Forward Propagation
# ==================================

def forward(X, W, b):
    z = X @ W + b

    pred = softmax(z)

    return pred


# ==================================
# Step 9. Backpropagation
# ==================================

def backward(X, y, pred):

    m = X.shape[0]

    dz = pred - y

    dW = (X.T @ dz) / m

    db = np.mean(dz, axis = 0, keepdims = True)

    return dW, db


# ==================================
# Step 10. Parameter Update
# ==================================

def update_parameters(W, b, dW, db, learning_rate):

    W = W - learning_rate * dW

    b = b - learning_rate * db

    return W, b


# ==================================
# Step 11. Training Loop
# ==================================

learning_rate = 0.1
epochs = 2000

y_train_onehot = one_hot_encoding(y_train, num_classes)

loss_history = []

for epoch in range(epochs):

    # 1. Forward Propagation
    predictions = forward(X_train, W, b)

    # 2. Calculate Categorical Cross Entropy
    loss = categorical_cross_entropy(y_train_onehot, predictions)

    # 3. Backpropagation
    dW, db = backward(X_train, y_train_onehot, predictions)

    W, b = update_parameters(W, b, dW, db, learning_rate)

    # 5. Record Loss
    loss_history.append(loss)

    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss = {loss:.6f}")


# ==================================
# Step 12. Evaluation
# ==================================

train_probs = forward(X_train, W, b)
test_probs = forward(X_test, W, b)

train_labels = np.argmax(train_probs, axis = 1)
test_labels = np.argmax(test_probs, axis = 1)

train_accuracy = np.mean(train_labels == y_train)
test_accuracy = np.mean(test_labels == y_test)

print("\n===== Final Results =====")
print(f"Train Accuracy: {train_accuracy * 100:.2f}%")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

plt.figure(figsize=(8, 5))
plt.plot(loss_history, label="Training Loss")

plt.xlabel("Epoch")
plt.ylabel("Categorical Cross Entropy")
plt.title("Softmax Regression - Iris Dataset")

plt.legend()
plt.grid(True)

plt.savefig(
    "02_Softmax_Regression/loss_curve.png",
    dpi=150,
    bbox_inches="tight"
)

plt.show()


# ==================================
# Step 13. Confusion Matrix
# ==================================

confusion_matrix = np.zeros(
    (num_classes, num_classes),
    dtype=int
)

for i in range(len(y_test)):

    actual = y_test[i]
    predicted = test_labels[i]

    confusion_matrix[actual, predicted] += 1


print("\nConfusion Matrix:")
print(confusion_matrix)

print("Total samples:", np.sum(confusion_matrix))


# ==================================
# Step 14. Multi-class Metrics
# ==================================

precision_list = []
recall_list = []
f1_list = []

for c in range(num_classes):

    # True Positive
    TP = confusion_matrix[c, c]

    predicted_positive = np.sum(confusion_matrix[:, c])

    actual_positive = np.sum(confusion_matrix[c, :])

    # Precision, Recall, F1
    precision = TP / predicted_positive if predicted_positive > 0 else 0.0
    recall = TP / actual_positive if actual_positive > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    precision_list.append(precision)
    recall_list.append(recall)
    f1_list.append(f1)

    print(
        f"Class {c}: "
        f"Precision={precision:.4f}, "
        f"Recall={recall:.4f}, "
        f"F1={f1:.4f}"
    )


# ==================================
# Step 15. Macro & Micro Average
# ==================================

# Macro Average
macro_precision = np.mean(precision_list)
macro_recall = np.mean(recall_list)
macro_f1 = np.mean(f1_list)

# Micro Average
total_TP = np.trace(confusion_matrix)
total_samples = np.sum(confusion_matrix)

micro_precision = total_TP / total_samples
micro_recall = total_TP / total_samples
micro_f1 = total_TP / total_samples

print("\n===== Overall Metrics =====")
print(f"Macro Precision: {macro_precision:.4f}")
print(f"Macro Recall: {macro_recall:.4f}")
print(f"Macro F1: {macro_f1:.4f}")

print(f"Micro Precision: {micro_precision:.4f}")
print(f"Micro Recall: {micro_recall:.4f}")
print(f"Micro F1: {micro_f1:.4f}")

print(f"Test Accuracy: {test_accuracy:.4f}")
