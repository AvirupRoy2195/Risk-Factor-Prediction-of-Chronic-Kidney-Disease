import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def evaluate_best_model():
    print("Loading engineered data...")
    X = pd.read_csv('X_engineered.csv')
    y = pd.read_csv('y_labels.csv').values.ravel()
    
    # Use the same split as in the training scripts (random_state=42, test_size=0.2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Loading the saved stacking ensemble model...")
    try:
        model = joblib.load('stacking_model.joblib')
    except FileNotFoundError:
        print("Error: stacking_model.joblib not found. Please run model_benchmark_ensemble.py first.")
        return

    # Check accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("\n--- MODEL ACCURACY REPORT ---")
    print(f"Final Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

if __name__ == "__main__":
    evaluate_best_model()
