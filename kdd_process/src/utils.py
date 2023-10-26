import pandas as pd


def load_original_data(file_path="../artifacts/fetal_health.csv"):
    df = pd.read_csv(file_path)
    return df


def load_cleaned_data(file_path="../artifacts/fetal_health_cleaned.csv"):
    df = pd.read_csv(file_path)
    return df
