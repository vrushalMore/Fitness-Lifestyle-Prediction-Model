import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score

def load_data(train_path):
    return pd.read_csv(train_path)

def train_models(X, y):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100),
        "Support Vector Machine": SVC(kernel='rbf', probability=True),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5)
    }
    
    best_model = None
    best_score = 0
    results = {}
    
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=5, scoring='f1')
        mean_score = np.mean(scores)
        results[name] = mean_score
        
        if mean_score > best_score:
            best_score = mean_score
            best_model = model
    
    print("Model Performance:")
    for model, score in results.items():
        print(f"{model}: F1 Score = {score:.4f}")
    
    return best_model

def save_model(model, output_path):
    joblib.dump(model, output_path)
    print(f"Best model saved: {output_path}")

if __name__ == "__main__":
    train_data_path = "../../data/processed/train.csv"
    model_output_path = "../../model_store/trained_model.pkl"
    
    df = load_data(train_data_path)
    
    X = df.drop(columns=['is_healthy'])
    y = df['is_healthy']
    
    best_model = train_models(X, y)
    best_model.fit(X, y)
    
    save_model(best_model, model_output_path)
    
    print("Model training complete.")