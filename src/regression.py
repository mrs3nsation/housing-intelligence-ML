import os
import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


def select_best_single_feature(X_train, y_train):
    """
    Select feature with highest absolute correlation with target.
    """
    df = X_train.copy()
    df["target"] = y_train

    correlations = df.corr()["target"].drop("target")
    best_feature = correlations.abs().idxmax()

    return best_feature


def train_regression_models(
    X_train_scaled,
    X_val_scaled,
    X_test_scaled,
    y_train,
    y_val,
    y_test,
    X_train_original,
):
    """
    Trains simple and multiple linear regression.
    Selects best model based on validation R².
    Evaluates on test set.
    Saves best model.
    """

    os.makedirs("models", exist_ok=True)
    os.makedirs("reports/figures", exist_ok=True)

    best_feature = select_best_single_feature(X_train_original, y_train)

    simple_model = LinearRegression()
    simple_model.fit(
        X_train_scaled[[best_feature]],
        y_train
    )

    val_pred_simple = simple_model.predict(
        X_val_scaled[[best_feature]]
    )

    r2_simple = r2_score(y_val, val_pred_simple)

    multiple_model = LinearRegression()
    multiple_model.fit(X_train_scaled, y_train)

    val_pred_multiple = multiple_model.predict(X_val_scaled)

    r2_multiple = r2_score(y_val, val_pred_multiple)

    if r2_multiple >= r2_simple:
        best_model = multiple_model
        model_type = "multiple"
        print("Multiple Linear Regression selected.")
    else:
        best_model = simple_model
        model_type = "simple"
        print("Simple Linear Regression selected.")

    if model_type == "multiple":
        test_pred = best_model.predict(X_test_scaled)
    else:
        test_pred = best_model.predict(X_test_scaled[[best_feature]])

    mse_test = mean_squared_error(y_test, test_pred)
    r2_test = r2_score(y_test, test_pred)

    print(f"Test MSE: {mse_test:.4f}")
    print(f"Test R²: {r2_test:.4f}")

    joblib.dump(
        {
            "model": best_model,
            "type": model_type,
            "feature": best_feature if model_type == "simple" else None,
        },
        "models/regression_model.pkl",
    )

    plt.figure()
    plt.scatter(y_test, test_pred, alpha=0.5)
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Actual vs Predicted (Regression)")
    plt.tight_layout()
    plt.savefig("reports/figures/regression_actual_vs_predicted.png")
    plt.close()

    print("Regression model saved and plot generated.")
