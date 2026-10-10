#MLP using validation set and early-stopping

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from model import (
    initialize_parameters,
    forward,
    backward,
    update_parameters,
    one_hot_encoding,
    categorical_cross_entropy
)

# ==================================
# Step 1. Load Dataset
# ==================================

digits = load_digits()

X = digits.data / 16.0
y = digits.target

# ==================================
# Step 2. Train / Validation / Test
# ==================================

X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval,
    test_size=0.2,
    random_state=42,
    stratify=y_trainval
)

print("Train:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)

# ==================================
# Step 3. One-hot Encoding
# ==================================

y_train_onehot = one_hot_encoding(y_train, 10)
y_val_onehot = one_hot_encoding(y_val, 10)

# ==================================
# Step 4. Initialize Parameters
# ==================================

np.random.seed(42)

W1, b1, W2, b2 = initialize_parameters(
    input_size=64,
    hidden_size=128,
    output_size=10
)


# ==================================
# Step 5. Training with Validation
# ==================================

learning_rate = 0.1
epochs = 150
batch_size = 32

train_losses = []
val_losses = []

# Early Stopping settings
patience = 15

best_val_loss = float("inf")
best_epoch = -1
best_params = None

wait = 0

for epoch in range(epochs):

    indices = np.random.permutation(X_train.shape[0])

    X_shuffled = X_train[indices]
    y_shuffled = y_train_onehot[indices]

    for start in range(0, X_train.shape[0], batch_size):

        X_batch = X_shuffled[start:start + batch_size]
        y_batch = y_shuffled[start:start + batch_size]

        pred, cache = forward(X_batch, W1, b1, W2, b2)

        dW1, db1, dW2, db2 = backward(X_batch, y_batch, pred, cache, W2)

        W1, b1, W2, b2 = update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate)

    # Evaluate the updated model
    train_probs, _ = forward(X_train, W1, b1, W2, b2)
    val_probs, _ = forward(X_val, W1, b1, W2, b2)

    train_loss = categorical_cross_entropy(y_train_onehot, train_probs)

    val_loss = categorical_cross_entropy(y_val_onehot, val_probs)

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    if epoch % 10 == 0:
            print(
                f"Epoch {epoch:3d} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f}"
            )

    
    # ==================================
    # Step 6. Early Stopping
    # ==================================

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        best_epoch = epoch

        best_params = (
            W1.copy(),
            b1.copy(),
            W2.copy(),
            b2.copy()
        )

        wait = 0

    else:

        wait += 1

        if wait >= patience:
            print(f"Early Stopping at Epoch {epoch}")
            break



plt.figure(figsize=(8, 5))

plt.plot(train_losses, label="Training Loss")
plt.plot(val_losses, label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Categorical Cross Entropy")
plt.title("MLP Training vs Validation Loss")
plt.legend()
plt.grid(True)
plt.show()

# ==================================
# Step 7. Restore Best Model
# ==================================

W1, b1, W2, b2 = best_params

print(f"Best Epoch: {best_epoch}")
print(f"Best Validation Loss: {best_val_loss:.6f}")

# ==================================
# Step 8. Final Test Evaluation
# ==================================

test_probs, _ = forward(X_test, W1, b1, W2, b2)

test_predictions = np.argmax(test_probs, axis=1)

test_accuracy = np.mean(test_predictions == y_test)

print(f"Final Test Accuracy: {test_accuracy * 100:.2f}%")