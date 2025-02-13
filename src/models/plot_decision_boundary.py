import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os

def load_data(file_path):
    """Loads the dataset from a CSV file."""
    return pd.read_csv(file_path)

def preprocess_data(df):
    """Performs preprocessing including missing value handling, encoding, and scaling."""
    
    # Drop duplicates if any
    df = df.drop_duplicates()
    
    # Handle missing values
    imputer = SimpleImputer(strategy='most_frequent')
    df.iloc[:, :] = imputer.fit_transform(df)
    
    # Encode categorical variables
    categorical_cols = ['diet_pref', 'act_level', 'career', 'gender']
    encoder = OneHotEncoder(drop='first', sparse_output=False)
    cat_encoded = encoder.fit_transform(df[categorical_cols])
    cat_encoded_df = pd.DataFrame(cat_encoded, columns=encoder.get_feature_names_out(categorical_cols))
    
    # Drop original categorical columns and concatenate encoded ones
    df = df.drop(columns=categorical_cols)
    df = pd.concat([df, cat_encoded_df], axis=1)
    
    # Normalize numerical features
    numeric_cols = ['phy_fitness', 'sleep_hrs', 'mindfulness', 'daily_avg_steps', 'daily_avg_calories']
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    return df, encoder, scaler

def train_and_plot_decision_boundary(df):
    """Trains a classifier and plots the decision boundary for phy_fitness vs mindfulness."""
    
    # Selecting features and target
    X = df[['phy_fitness', 'mindfulness']]
    y = df['is_healthy']  # Assuming 'is_healthy' is the target variable
    
    # Splitting data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    # Predict and evaluate
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.2f}")
    
    # Plot decision boundary
    x_min, x_max = X['phy_fitness'].min() - 1, X['phy_fitness'].max() + 1
    y_min, y_max = X['mindfulness'].min() - 1, X['mindfulness'].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3)
    sns.scatterplot(x=X['phy_fitness'], y=X['mindfulness'], hue=y, palette='coolwarm', edgecolor='k')
    plt.xlabel('Physical Fitness')
    plt.ylabel('Mindfulness')
    plt.title('Decision Boundary for Healthy vs Unhealthy')
    plt.show()
    
if __name__ == "__main__":
    raw_data_path = "../../data/raw/innovize_final_ml.csv"
    df = load_data(raw_data_path)
    processed_df, encoder, scaler = preprocess_data(df)
    train_and_plot_decision_boundary(processed_df)
    print("Preprocessing and model training complete.")
