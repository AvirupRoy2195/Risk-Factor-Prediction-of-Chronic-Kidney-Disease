"""
Evaluate the accuracy of the trained stacking ensemble model.
Provides detailed classification report and confusion matrix.
"""
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def evaluate_best_model():
    """Evaluate the stacking ensemble on held-out test data."""
    print("Loading engineered data...")
    X = pd.read_csv('X_engineered.csv')
    y = pd.read_csv('y_labels.csv').values.ravel()
    
    # Use the same split as in the training scripts (random_state=42, test_size=0.2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Loading the saved stacking ensemble model...")
    try:
        model = joblib.load('stacking_model.joblib')
    except FileNotFoundError:
        print("Error: stacking_model.joblib not found. Please run train.py first.")
        return
    
    # Check accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("\n" + "="*50)
    print("        MODEL ACCURACY REPORT")
    print("="*50)
    print(f"\n✅ Final Accuracy: {accuracy:.2%}")
    
    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['No CKD', 'CKD']))
    
    print("\n🔢 Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"              Predicted")
    print(f"              No CKD  CKD")
    print(f"Actual No CKD   {cm[0][0]:4d}  {cm[0][1]:4d}")
    print(f"       CKD      {cm[1][0]:4d}  {cm[1][1]:4d}")
    
    # Additional metrics
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    print(f"\n📈 Additional Metrics:")
    print(f"   Sensitivity (Recall): {sensitivity:.2%}")
    print(f"   Specificity: {specificity:.2%}")
    print("="*50)

if __name__ == "__main__":
    evaluate_best_model()
