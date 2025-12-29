import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

def train_model():
    # 1. Fetch data
    print("Fetching dataset...")
    # NOTE: In a real scenario, we should use the engineered data from feature_engineering.py
    # But for consistency with the current flow, we'll load the processed X_engineered.csv if available
    try:
        print("Loading engineered features...")
        X = pd.read_csv('X_engineered.csv')
        y = pd.read_csv('y_labels.csv').values.ravel()
    except FileNotFoundError:
        print("Engineered data not found. Please run feature_engineering.py first.")
        return

    print(f"Data shape: {X.shape}")

    # 3. Model Training (with SMOTE and Balancing)
    # Handle Imbalance
    try:
        from imblearn.over_sampling import SMOTE
        print("balancing classes with SMOTE...")
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
    except ImportError:
        print("⚠️ 'imblearn' not installed. Proceeding with standard split (using class_weight).")
        X_resampled, y_resampled = X, y

    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    # Define Base Learners (Optimized for probabilities)
    base_learners = [
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')),
        ('svm', SVC(probability=True, random_state=42, class_weight='balanced')),
        ('knn', KNeighborsClassifier(n_neighbors=5))
    ]

    # Stacking Classifier
    print("\nTraining Stacking Ensemble (RF + SVM + KNN -> LR)...")
    stacking_model = StackingClassifier(
        estimators=base_learners,
        final_estimator=LogisticRegression(class_weight='balanced'),
        cv=5
    )
    
    stacking_model.fit(X_train, y_train)
    y_pred = stacking_model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"Stacking Ensemble Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))

    # Save the robust model
    joblib.dump(stacking_model, 'stacking_model.joblib')
    print("✅ Model saved to 'stacking_model.joblib'")

    # Standalone RF for Feature Importance
    rf_feat = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_feat.fit(X_train, y_train)
    importances = rf_feat.feature_importances_
    feature_names = X.columns
    feat_df = pd.DataFrame({'feature': feature_names, 'importance': importances}).sort_values(by='importance', ascending=False)
    
    print("\nTop 15 Important Features:")
    print(feat_df.head(15))
    
    # Save results to a file
    with open('model_results.txt', 'w') as f:
        f.write("--- MODEL PERFORMANCE ---\n")
        f.write(f"Stacking Ensemble Accuracy: {acc}\n")
        f.write("\nClassification Report:\n")
        f.write(classification_report(y_test, y_pred))
        f.write("\n--- FEATURE IMPORTANCE ---\n")
        f.write(feat_df.head(20).to_string())

if __name__ == "__main__":
    train_model()
