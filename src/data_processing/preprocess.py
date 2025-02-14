import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import os

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(df):
    df = df.drop_duplicates()
    
    imputer = SimpleImputer(strategy='most_frequent')
    df.iloc[:, :] = imputer.fit_transform(df)
    
    categorical_cols = ['diet_pref', 'act_level', 'career', 'gender']
    encoder = OneHotEncoder(drop='first', sparse_output=False)
    cat_encoded = encoder.fit_transform(df[categorical_cols])
    cat_encoded_df = pd.DataFrame(cat_encoded, columns=encoder.get_feature_names_out(categorical_cols))
    
    df = df.drop(columns=categorical_cols)
    df = pd.concat([df, cat_encoded_df], axis=1)
    
    numeric_cols = ['phy_fitness', 'sleep_hrs', 'mindfulness', 'daily_avg_steps', 'daily_avg_calories']
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    return df, scaler, encoder

def save_preprocessed_data(df, output_path):
    df.to_csv(output_path, index=False)
    
if __name__ == "__main__":
    raw_data_path = "../../data/raw/innovize_final_ml.csv"
    processed_data_path = "../../data/processed/preprocessed_data.csv"
    
    df = load_data(raw_data_path)
    processed_df, scaler, encoder = preprocess_data(df)
    save_preprocessed_data(processed_df, processed_data_path)