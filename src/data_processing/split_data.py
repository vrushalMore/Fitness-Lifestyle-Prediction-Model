import pandas as pd
from sklearn.model_selection import train_test_split
import os

def load_preprocessed_data(file_path):
    """Loads the preprocessed dataset from a CSV file."""
    return pd.read_csv(file_path)

def split_data(df, test_size=0.2, random_state=42):
    """Splits the dataset into training and testing sets."""
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    return train_df, test_df

def save_split_data(train_df, test_df, train_path, test_path):
    """Saves the split datasets to CSV files."""
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    print("Train and test datasets saved successfully.")

if __name__ == "__main__":
    preprocessed_data_path = "../../data/processed/preprocessed_data.csv"
    train_data_path = "../../data/processed/train.csv"
    test_data_path = "../../data/processed/test.csv"
    
    df = load_preprocessed_data(preprocessed_data_path)
    train_df, test_df = split_data(df)
    save_split_data(train_df, test_df, train_data_path, test_data_path)
