#checking hyperparameters

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

y_train_onehot = one_hot_encoding(y_train, 10)
y_val_onehot = one_hot_encoding(y_val, 10)

print("Train:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)


# ==================================
# Step 2. Train One MLP Model
# ==================================

def train_model(hidden_size,
                learning_rate=0.1,
                batch_size=32,
                epochs=150,
                patience=15,
                seed=42):

    np.random.seed(seed)

    W1, b1, W2, b2 = initialize_parameters(
        input_size=64,
        hidden_size=hidden_size,
        output_size=10
    )

    train_losses = []
    val_losses = []

    best_val_loss = float("inf")
    best_epoch = -1
    best_params = None
    wait = 0

    for epoch in range(epochs):

        # Shuffle
        indices = np.random.permutation(X_train.shape[0])

        X_shuffled = X_train[indices]
        y_shuffled = y_train_onehot[indices]

        for start in range(0, X_train.shape[0], batch_size):

            X_batch = X_shuffled[start:start + batch_size]
            y_batch = y_shuffled[start:start + batch_size]

            pred, cache = forward(X_batch, W1, b1, W2, b2)

            dW1, db1, dW2, db2 = backward(X_batch, y_batch, pred, cache, W2)

            W1, b1, W2, b2 = update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate)

        # Evaluate after each epoch
        train_probs, _ = forward(X_train, W1, b1, W2, b2)
        val_probs, _ = forward(X_val, W1, b1, W2, b2)

        train_loss = categorical_cross_entropy(y_train_onehot, train_probs)
        val_loss = categorical_cross_entropy(y_val_onehot, val_probs)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

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

    # Restore the best model
    W1, b1, W2, b2 = best_params

    # Validation accuracy
    val_probs, _ = forward(X_val, W1, b1, W2, b2)
    val_predictions = np.argmax(val_probs, axis=1)
    val_accuracy = np.mean(val_predictions == y_val)

    return {
        "hidden_size": hidden_size,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "val_accuracy": val_accuracy,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "parameters": (W1, b1, W2, b2)
    }

if __name__ == "__main__":
    # ==================================
    # Step 3. Hyperparameter Experiment
    # ==================================

    hidden_sizes = [32, 64, 128, 256]

    results = []

    for hidden_size in hidden_sizes:

        print(f"\nTraining MLP with Hidden Size = {hidden_size}")

        result = train_model(hidden_size)

        results.append(result)

        print(f"Best Epoch: {result['best_epoch']}")
        print(f"Best Val Loss: {result['best_val_loss']:.6f}")
        print(f"Val Accuracy: {result['val_accuracy'] * 100:.2f}%")


    sizes = [r["hidden_size"] for r in results]
    accuracies = [r["val_accuracy"] * 100 for r in results]

    plt.figure(figsize=(8, 5))
    plt.plot(sizes, accuracies, marker="o")

    plt.xlabel("Hidden Size")
    plt.ylabel("Validation Accuracy (%)")
    plt.title("MLP Hidden Size Comparison")
    plt.xticks(sizes)
    plt.grid(True)
    plt.show()


    # ==================================
    # Step 4. Learning Rate Experiment
    # ==================================

    learning_rates = [0.01, 0.05, 0.1]

    lr_results = []

    for lr in learning_rates:

        print(f"\nTraining MLP with Learning Rate = {lr}")

        # Keep hidden_size = 64
        result = train_model(hidden_size=64, learning_rate=lr)

        lr_results.append(result)

        print(f"Best Epoch: {result['best_epoch']}")
        print(f"Best Val Loss: {result['best_val_loss']:.6f}")
        print(f"Val Accuracy: {result['val_accuracy'] * 100:.2f}%")


    plt.figure(figsize=(8, 5))

    for lr, result in zip(learning_rates, lr_results):
        plt.plot(
            result["val_losses"],
            label=f"Learning Rate = {lr}"
        )

    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.title("Learning Rate Comparison")
    plt.legend()
    plt.grid(True)
    plt.show()


    # ==================================
    # Step 5. Batch Size Experiment
    # ==================================

    batch_sizes = [16, 32, 64]

    batch_results = []

    for batch_size in batch_sizes:

        print(f"\nTraining MLP with Batch Size = {batch_size}")

        # hidden_size = 64
        # learning_rate = 0.1
        # batch_size = current batch_size
        result = train_model(hidden_size=64, learning_rate=0.1, batch_size= batch_size)

        batch_results.append(result)

        print(f"Best Epoch: {result['best_epoch']}")
        print(f"Best Val Loss: {result['best_val_loss']:.6f}")
        print(f"Val Accuracy: {result['val_accuracy'] * 100:.2f}%")


    plt.figure(figsize=(8, 5))

    for batch_size, result in zip(batch_sizes, batch_results):
        plt.plot(
            result["val_losses"],
            label=f"Batch Size = {batch_size}"
        )

    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.title("Batch Size Comparison")
    plt.legend()
    plt.grid(True)
    plt.show()


    # ==================================
    # Step 6. Multiple Random Seeds
    # ==================================

    seeds = [42, 123, 2026]
    batch_sizes_to_compare = [16, 32]

    seed_results = {}

    for batch_size in batch_sizes_to_compare:

        val_losses = []
        val_accuracies = []

        for seed in seeds:

            result = train_model(
                hidden_size=64,
                learning_rate=0.1,
                batch_size=batch_size,
                epochs=150,
                patience=15,
                seed=seed
            )

            val_losses.append(result["best_val_loss"])

            val_accuracies.append(result["val_accuracy"])

            print(
                f"Batch Size: {batch_size}, "
                f"Seed: {seed}, "
                f"Val Loss: {result['best_val_loss']:.6f}, "
                f"Val Accuracy: {result['val_accuracy'] * 100:.2f}%"
            )

        mean_loss = np.mean(val_losses)
        std_loss = np.std(val_losses)

        mean_acc = np.mean(val_accuracies)
        std_acc = np.std(val_accuracies)

        seed_results[batch_size] = {
            "mean_loss": mean_loss,
            "std_loss": std_loss,
            "mean_acc": mean_acc,
            "std_acc": std_acc
        }

        print(f"\nBatch Size = {batch_size}")
        print(f"Val Loss: {mean_loss:.6f} ± {std_loss:.6f}")
        print(f"Val Accuracy: {mean_acc * 100:.2f}% ± {std_acc * 100:.2f}%p")
