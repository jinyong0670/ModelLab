import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from model import (
    initialize_parameters,
    relu,
    relu_derivative,
    softmax,
    forward,
    one_hot_encoding,
    categorical_cross_entropy,
    backward,
    update_parameters
)

# ==================================
# Step 1. Load Dataset
# ==================================

digits = load_digits()

X = digits.data
y = digits.target
images = digits.images

print("X shape:", X.shape)
print("y shape:", y.shape)
print("Images shape:", images.shape)

print("First label:", y[0])
print("First feature vector:", X[0])

plt.figure(figsize=(8, 3))

for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(images[i], cmap="gray")
    plt.title(f"Label: {y[i]}")
    plt.axis("off")

plt.tight_layout()
plt.show()

# ==================================
# Step 2. Preprocessing
# ==================================

#normalization
X = X / 16.0

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train:", X_train.shape)
print("Test:", X_test.shape)

# ==================================
# Step 3. Parameter Initialization
# ==================================

np.random.seed(42)

W1, b1, W2, b2 = initialize_parameters(
    input_size=64,
    hidden_size=128,
    output_size=10
)

# ==================================
# Step 4. Training Loop
# ==================================

learning_rate = 0.1
epochs = 1000

y_train_onehot = one_hot_encoding(y_train, 10)

loss_history = []

for epoch in range(epochs):

    pred, cache = forward(X_train, W1, b1, W2, b2)

    loss = categorical_cross_entropy(y_train_onehot, pred)

    dW1, db1, dW2, db2 = backward(X_train, y_train_onehot, pred, cache, W2)

    W1, b1, W2, b2 = update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate)

    loss_history.append(loss)

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.6f}")

plt.figure(figsize=(8, 5))
plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("Categorical Cross Entropy")
plt.title("MLP Training Loss")
plt.grid(True)
plt.show()

train_probs, _ = forward(X_train, W1, b1, W2, b2)
test_probs, _ = forward(X_test, W1, b1, W2, b2)

train_predictions = np.argmax(train_probs, axis=1)
test_predictions = np.argmax(test_probs, axis=1)

train_accuracy = np.mean(train_predictions == y_train)
test_accuracy = np.mean(test_predictions == y_test)

print(f"Train Accuracy: {train_accuracy * 100:.2f}%")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")



# ==================================
# Step 5. Gradient Checking
# ==================================

epsilon = 1e-5

# 1. Small dataset
X_check = X_train[:8]
y_check = one_hot_encoding(y_train[:8], 10)

# 2. Fresh parameters (before training)
np.random.seed(123)

W1_check, b1_check, W2_check, b2_check = initialize_parameters(
    input_size=64,
    hidden_size=128,
    output_size=10
)

# 3. Analytical Gradient
pred_check, cache_check = forward(
    X_check, W1_check, b1_check, W2_check, b2_check
)

dW1_check, db1_check, dW2_check, db2_check = backward(
    X_check,
    y_check,
    pred_check,
    cache_check,
    W2_check
)

# 4. Numerical Gradient
for row, col in [(10, 0), (20, 0), (30, 5), (40, 10)]:

    original = W1_check[row, col]

    W1_check[row, col] = original + epsilon
    pred_plus, _ = forward(
        X_check, W1_check, b1_check, W2_check, b2_check
    )
    loss_plus = categorical_cross_entropy(y_check, pred_plus)

    W1_check[row, col] = original - epsilon
    pred_minus, _ = forward(
        X_check, W1_check, b1_check, W2_check, b2_check
    )
    loss_minus = categorical_cross_entropy(y_check, pred_minus)

    W1_check[row, col] = original

    numerical = (loss_plus - loss_minus) / (2 * epsilon)
    analytical = dW1_check[row, col]

    error = abs(numerical - analytical) / max(
        1e-8,
        abs(numerical) + abs(analytical)
    )

    print(f"W1[{row},{col}]")
    print(f"  Numerical:  {numerical:.10f}")
    print(f"  Analytical: {analytical:.10f}")
    print(f"  Error:      {error:.2e}")


# ==================================
# Step 6. Mini-batch Gradient Descent
# ==================================

np.random.seed(42)

W1_mb, b1_mb, W2_mb, b2_mb = initialize_parameters(
    input_size=64,
    hidden_size=128,
    output_size=10
)

learning_rate = 0.1
epochs = 100
batch_size = 32

mini_batch_losses = []

for epoch in range(epochs):

    # Shuffle training dataset
    indices = np.random.permutation(X_train.shape[0])

    X_shuffled = X_train[indices]
    y_shuffled = y_train_onehot[indices]

    # Process mini-batches
    for start in range(0, X_train.shape[0], batch_size):

        end = start + batch_size

        X_batch = X_shuffled[start:end]
        y_batch = y_shuffled[start:end]

        # TODO 1: Forward
        pred, cache = forward(X_batch, W1_mb, b1_mb, W2_mb, b2_mb)

        # TODO 2: Backward
        dW1, db1, dW2, db2 = backward(X_batch, y_batch, pred, cache, W2_mb)

        # TODO 3: Update parameters
        W1_mb, b1_mb, W2_mb, b2_mb = update_parameters(W1_mb, b1_mb, W2_mb, b2_mb, dW1, db1, dW2, db2, learning_rate)

    # Calculate full training loss after each epoch
    train_probs, _ = forward(
        X_train, W1_mb, b1_mb, W2_mb, b2_mb
    )

    epoch_loss = categorical_cross_entropy(
        y_train_onehot, train_probs
    )

    mini_batch_losses.append(epoch_loss)

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {epoch_loss:.6f}")

test_probs_mb, _ = forward(
    X_test, W1_mb, b1_mb, W2_mb, b2_mb
)

test_predictions_mb = np.argmax(test_probs_mb, axis=1)
test_accuracy_mb = np.mean(test_predictions_mb == y_test)

print(f"Mini-batch Test Accuracy: {test_accuracy_mb * 100:.2f}%")
