import os
import joblib
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def train_svm_model(
    X_train_scaled,
    X_val_scaled,
    X_test_scaled,
    y_train_cls,
    y_val_cls,
    y_test_cls
):
    """
    Trains SVM with Linear and RBF kernels.
    Selects best model based on validation accuracy.
    Evaluates on test set.
    Saves final model.
    """

    os.makedirs("models", exist_ok=True)

    best_accuracy = 0
    best_model = None
    best_config = None

    # Try Linear Kernel
    for C in [0.1, 1, 10]:

        model = SVC(kernel="linear", C=C)
        model.fit(X_train_scaled, y_train_cls)

        val_pred = model.predict(X_val_scaled)
        acc = accuracy_score(y_val_cls, val_pred)

        if acc > best_accuracy:
            best_accuracy = acc
            best_model = model
            best_config = f"Linear (C={C})"

    # Try RBF Kernel
    for C in [0.1, 1, 10]:

        model = SVC(kernel="rbf", C=C, gamma="scale")
        model.fit(X_train_scaled, y_train_cls)

        val_pred = model.predict(X_val_scaled)
        acc = accuracy_score(y_val_cls, val_pred)

        if acc > best_accuracy:
            best_accuracy = acc
            best_model = model
            best_config = f"RBF (C={C})"

    print(f"Best SVM Configuration (Validation): {best_config}")
    print(f"Validation Accuracy: {best_accuracy:.4f}")

    # Final Evaluation on Test Set
    test_pred = best_model.predict(X_test_scaled)

    acc_test = accuracy_score(y_test_cls, test_pred)
    precision = precision_score(y_test_cls, test_pred, average="weighted")
    recall = recall_score(y_test_cls, test_pred, average="weighted")
    f1 = f1_score(y_test_cls, test_pred, average="weighted")

    print("\nSVM Test Metrics:")
    print(f"Accuracy: {acc_test:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Save Model
    joblib.dump(
        {
            "model": best_model,
            "config": best_config
        },
        "models/svm_model.pkl"
    )

    print("SVM model saved.")
