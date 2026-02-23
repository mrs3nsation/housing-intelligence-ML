from sklearn.datasets import fetch_california_housing
import pandas as pd


def load_data():
    """
    Loads California Housing dataset from sklearn.
    Returns:
        df (pd.DataFrame): full dataset including target column.
    """
    data = fetch_california_housing(as_frame=True)
    df = data.frame
    return df
