# src/neural_network.py

import os
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score


def build_model(input_dim):
    """
    Builds small neural network model.
    """

    model = Sequential([
        Dense(32, activation="relu", input_shape=(input_dim,)),
        Dense(16, activation="relu"),
        Dense(3, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def train_neural_network(
    X_train_scaled,
    X_val_scaled,
    X_test_scaled,
    y_train_cls,
    y_val_cls,
    y_test_cls
):
    """
    Trains neural network using training set.
    Uses validation set for monitoring.
    Test set used once for final evaluation.
    """

    os.makedirs("models", exist_ok=True)
    os.makedirs("reports/figures", exist_ok=True)

    # Convert to numpy (TensorFlow prefers numpy arrays)
    X_train_np = np.array(X_train_scaled)
    X_val_np = np.array(X_val_scaled)
    X_test_np = np.array(X_test_scaled)

    y_train_np = np.array(y_train_cls)
    y_val_np = np.array(y_val_cls)
    y_test_np = np.array(y_test_cls)

    # Build model
    model = build_model(input_dim=X_train_np.shape[1])

    # Early stopping
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True
    )

    # Train
    history = model.fit(
        X_train_np,
        y_train_np,
        validation_data=(X_val_np, y_val_np),
        epochs=100,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1
    )

    # Final evaluation on test set (ONLY ONCE)
    test_loss, test_accuracy = model.evaluate(X_test_np, y_test_np, verbose=0)

    print("\nNeural Network Test Accuracy:", round(test_accuracy, 4))

    # Plot Training vs Validation Accuracy
    plt.figure()
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.title("Training vs Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(["Train", "Validation"])
    plt.tight_layout()
    plt.savefig("reports/figures/nn_accuracy.png")
    plt.close()

    # Plot Training vs Validation Loss
    plt.figure()
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(["Train", "Validation"])
    plt.tight_layout()
    plt.savefig("reports/figures/nn_loss.png")
    plt.close()

    # Save model
    model.save("models/neural_network_model.h5")

    print("Neural network model saved.")
