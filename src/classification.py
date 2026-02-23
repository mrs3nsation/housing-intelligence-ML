import os
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)


def train_classification_models(
    X_train_scaled,
    X_val_scaled,
    X_test_scaled,
    y_train_cls,
    y_val_cls,
    y_test_cls
):
    """
    Trains Logistic Regression, Decision Tree, and Random Forest.
    Selects best model using validation accuracy.
    Evaluates final model on test set.
    Saves model and confusion matrix plot.
    """

    os.makedirs("models", exist_ok=True)
    os.makedirs("reports/figures", exist_ok=True)

    # -----------------------------------
    # 1. Logistic Regression
    # -----------------------------------
    log_model = LogisticRegression(max_iter=1000)
    log_model.fit(X_train_scaled, y_train_cls)

    val_pred_log = log_model.predict(X_val_scaled)
    acc_log = accuracy_score(y_val_cls, val_pred_log)

    # -----------------------------------
    # 2. Decision Tree
    # -----------------------------------
    dt_model = DecisionTreeClassifier(
        max_depth=10,
        random_state=42
    )
    dt_model.fit(X_train_scaled, y_train_cls)

    val_pred_dt = dt_model.predict(X_val_scaled)
    acc_dt = accuracy_score(y_val_cls, val_pred_dt)

    # -----------------------------------
    # 3. Random Forest
    # -----------------------------------
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train_scaled, y_train_cls)

    val_pred_rf = rf_model.predict(X_val_scaled)
    acc_rf = accuracy_score(y_val_cls, val_pred_rf)

    print(f"Validation Accuracy - Logistic: {acc_log:.4f}")
    print(f"Validation Accuracy - Decision Tree: {acc_dt:.4f}")
    print(f"Validation Accuracy - Random Forest: {acc_rf:.4f}")

    # -----------------------------------
    # Model Selection
    # -----------------------------------
    val_accuracies = {
        "logistic": acc_log,
        "decision_tree": acc_dt,
        "random_forest": acc_rf
    }

    best_model_name = max(val_accuracies, key=val_accuracies.get)

    if best_model_name == "logistic":
        best_model = log_model
    elif best_model_name == "decision_tree":
        best_model = dt_model
    else:
        best_model = rf_model

    print(f"Selected Model: {best_model_name}")

    # -----------------------------------
    # Final Test Evaluation
    # -----------------------------------
    test_pred = best_model.predict(X_test_scaled)

    acc_test = accuracy_score(y_test_cls, test_pred)
    precision = precision_score(y_test_cls, test_pred, average="weighted")
    recall = recall_score(y_test_cls, test_pred, average="weighted")
    f1 = f1_score(y_test_cls, test_pred, average="weighted")

    print("\nTest Metrics:")
    print(f"Accuracy: {acc_test:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # -----------------------------------
    # Confusion Matrix Plot
    # -----------------------------------
    cm = confusion_matrix(y_test_cls, test_pred)

    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix (Test Set)")
    plt.tight_layout()
    plt.savefig("reports/figures/classification_confusion_matrix.png")
    plt.close()

    # -----------------------------------
    # Save Model
    # -----------------------------------
    joblib.dump(
        {
            "model": best_model,
            "model_name": best_model_name
        },
        "models/classification_model.pkl"
    )

    print("Classification model saved.")
