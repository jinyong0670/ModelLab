import numpy as np

from model import forward
from hyperparameter import train_model, X_test, y_test


# ==================================
# Step 1. Final Hyperparameters
# ==================================

hidden_size = 64
learning_rate = 0.1
batch_size = 16


# ==================================
# Step 2. Train Final Model
# ==================================

result = train_model(
    hidden_size=hidden_size,
    learning_rate=learning_rate,
    batch_size=batch_size,
    epochs=150,
    patience=15,
    seed=42
)


# ==================================
# Step 3. Get Best Parameters
# ==================================

W1, b1, W2, b2 = result["parameters"]


# ==================================
# Step 4. Predict Test Data
# ==================================

test_probs, _ = forward(
    X_test, W1, b1, W2, b2
)

test_predictions = np.argmax(test_probs, axis=1)


# ==================================
# Step 5. Calculate Accuracy
# ==================================

test_accuracy = np.mean(test_predictions == y_test)


# ==================================
# Step 6. Print Results
# ==================================

print(f"Best Epoch: {result['best_epoch']}")
print(f"Best Validation Loss: {result['best_val_loss']:.6f}")
print(f"Final Test Accuracy: {test_accuracy * 100:.2f}%")
