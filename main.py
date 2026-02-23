# main.py

from src.data_loader import load_data
from src.preprocessing import split_data, scale_features, create_class_labels
from src.eda import run_eda
from src.regression import train_regression_models
from src.classification import train_classification_models
from src.svm import train_svm_model
from src.neural_network import train_neural_network
from src.clustering import train_clustering_model


def main():

    print("\n========== STARTING PROJECT PIPELINE ==========\n")

    # --------------------------------------------------
    # 1. Load Dataset
    # --------------------------------------------------
    df = load_data()

    if df is None or df.empty:
        raise ValueError("Dataset failed to load or is empty.")

    print(f"Dataset loaded successfully. Shape: {df.shape}\n")

    # --------------------------------------------------
    # 2. Exploratory Data Analysis (Full Dataset)
    # --------------------------------------------------
    print("Running EDA...")
    run_eda(df)
    print("EDA completed.\n")

    # --------------------------------------------------
    # 3. Train / Validation / Test Split (70/15/15)
    # --------------------------------------------------
    print("Splitting dataset...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    assert len(X_train) > 0, "Training set is empty."
    assert len(X_val) > 0, "Validation set is empty."
    assert len(X_test) > 0, "Test set is empty."

    print(f"Train size: {len(X_train)}")
    print(f"Validation size: {len(X_val)}")
    print(f"Test size: {len(X_test)}\n")

    # --------------------------------------------------
    # 4. Feature Scaling (Fit on Training Only)
    # --------------------------------------------------
    print("Scaling features...")
    X_train_scaled, X_val_scaled, X_test_scaled = scale_features(
        X_train, X_val, X_test
    )

    if X_train_scaled.isnull().values.any():
        raise ValueError("NaN detected after scaling.")

    print("Feature scaling completed.\n")

    # --------------------------------------------------
    # 5. Create Classification Labels (Using Training Quantiles Only)
    # --------------------------------------------------
    print("Creating classification labels...")
    y_train_cls, y_val_cls, y_test_cls, q1, q2 = create_class_labels(
        y_train, y_val, y_test
    )

    print("Classification thresholds:")
    print(f"Low <= {q1:.3f}")
    print(f"Medium <= {q2:.3f}")
    print(f"High > {q2:.3f}\n")

    # --------------------------------------------------
    # 6. Regression Phase
    # --------------------------------------------------
    print("Training Regression models...")
    train_regression_models(
        X_train_scaled,
        X_val_scaled,
        X_test_scaled,
        y_train,
        y_val,
        y_test,
        X_train
    )
    print("Regression phase completed.\n")

    # --------------------------------------------------
    # 7. Classification Phase
    # --------------------------------------------------
    print("Training Classification models...")
    train_classification_models(
        X_train_scaled,
        X_val_scaled,
        X_test_scaled,
        y_train_cls,
        y_val_cls,
        y_test_cls
    )
    print("Classification phase completed.\n")

    # --------------------------------------------------
    # 8. SVM Phase
    # --------------------------------------------------
    print("Training SVM model...")
    train_svm_model(
        X_train_scaled,
        X_val_scaled,
        X_test_scaled,
        y_train_cls,
        y_val_cls,
        y_test_cls
    )
    print("SVM phase completed.\n")

    # --------------------------------------------------
    # 9. Neural Network Phase
    # --------------------------------------------------
    print("Training Neural Network...")
    train_neural_network(
        X_train_scaled,
        X_val_scaled,
        X_test_scaled,
        y_train_cls,
        y_val_cls,
        y_test_cls
    )
    print("Neural Network phase completed.\n")

    # --------------------------------------------------
    # 10. Clustering Phase
    # --------------------------------------------------
    print("Training Clustering model...")
    train_clustering_model(
        X_train_scaled,
        X_val_scaled
    )
    print("Clustering phase completed.\n")

    print("========== ALL PHASES COMPLETED SUCCESSFULLY ==========\n")


if __name__ == "__main__":
    main()
    
    
#.\venv310\Scripts\Activate.ps1