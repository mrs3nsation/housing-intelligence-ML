# src/preprocessing.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os


def split_data(df, random_state=42):
    """
    Splits dataset into 70% train, 15% validation, 15% test.
    """

    X = df.drop(columns=["MedHouseVal"])
    y = df["MedHouseVal"]

    # First split: 70% train, 30% temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=random_state, shuffle=True
    )

    # Second split: 15% val, 15% test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=random_state, shuffle=True
    )

    return X_train, X_val, X_test, y_train, y_val, y_test

def scale_features(X_train, X_val, X_test, save_path="models/scaler.pkl"):
    """
    Applies StandardScaler using training statistics only.
    Saves scaler for later use in website.
    Returns scaled DataFrames (not numpy arrays).
    """

    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train_scaled = pd.DataFrame(
        scaler.transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )

    X_val_scaled = pd.DataFrame(
        scaler.transform(X_val),
        columns=X_val.columns,
        index=X_val.index
    )

    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )

    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, save_path)

    return X_train_scaled, X_val_scaled, X_test_scaled

def create_class_labels(y_train, y_val, y_test, save_path="models/class_thresholds.pkl"):
    """
    Converts continuous house values into Low / Medium / High categories.
    Uses training quantiles only.
    Saves thresholds for website usage.
    """

    q1 = y_train.quantile(0.33)
    q2 = y_train.quantile(0.66)

    def categorize(y):
        if y <= q1:
            return 0
        elif y <= q2:
            return 1
        else:
            return 2

    y_train_cls = y_train.apply(categorize)
    y_val_cls = y_val.apply(categorize)
    y_test_cls = y_test.apply(categorize)

    joblib.dump((q1, q2), save_path)

    return y_train_cls, y_val_cls, y_test_cls, q1, q2
