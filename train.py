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
    # CRITICAL FIX: Split FIRST, then apply SMOTE only to training data
    # This prevents test set contamination and inflated metrics
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Original train size: {len(X_train)}, Test size: {len(X_test)}")
    
    # Handle Imbalance - ONLY on training data
    try:
        from imblearn.over_sampling import SMOTE
        print("Balancing training classes with SMOTE (test set unchanged)...")
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        print(f"After SMOTE train size: {len(X_train)} (test still: {len(X_test)})")
    except ImportError:
        print("⚠️ 'imblearn' not installed. Proceeding with class_weight balancing.")

    # Define Base Learners (Optimized for probabilities)
    # 1. XGBoost (Gradient Boosting - Fast execution O(n log n))
    # 2. HistGradientBoosting (LightGBM inspired - Fast implementation from sklearn)
    # 3. RandomForest (Parallelized with n_jobs=-1)
    
    from xgboost import XGBClassifier
    from sklearn.ensemble import HistGradientBoostingClassifier
    
    # ==========================================
    # FAST MODEL (XGBoost Only - <1s inference)
    # ==========================================
    print("\n--- Training FAST MODEL (XGBoost) ---")
    fast_model = XGBClassifier(
        n_estimators=150,  # Slightly fewer for speed
        learning_rate=0.1,  # Faster convergence
        max_depth=6,
        n_jobs=-1,
        eval_metric='logloss',
        random_state=42
    )
    fast_model.fit(X_train, y_train)
    fast_acc = accuracy_score(y_test, fast_model.predict(X_test))
    print(f"✅ Fast Model Accuracy: {fast_acc:.4f}")
    joblib.dump(fast_model, 'fast_model.joblib')
    print("✅ Fast Model saved to 'fast_model.joblib'")

    # ==========================================
    # DEEP THINK MODEL (Stacking Ensemble)
    # ==========================================
    print("\n--- Training DEEP THINK MODEL (Stacking Ensemble) ---")
    base_learners = [
        ('xgb', XGBClassifier(
            n_estimators=200, 
            learning_rate=0.05, 
            max_depth=5, 
            n_jobs=-1,
            eval_metric='logloss',
            random_state=42
        )),
        ('hgb', HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.05,
            random_state=42
        )),
        ('rf', RandomForestClassifier(
            n_estimators=100, 
            class_weight='balanced', 
            n_jobs=-1,
            random_state=42
        ))
    ]

    stacking_model = StackingClassifier(
        estimators=base_learners,
        final_estimator=LogisticRegression(class_weight='balanced', max_iter=1000),
        cv=3,
        n_jobs=-1
    )
    
    stacking_model.fit(X_train, y_train)
    y_pred = stacking_model.predict(X_test)
    
    deep_acc = accuracy_score(y_test, y_pred)
    print(f"✅ Stacking Ensemble Accuracy: {deep_acc:.4f}")
    print(classification_report(y_test, y_pred))

    # Save the robust model
    joblib.dump(stacking_model, 'stacking_model.joblib')
    print("✅ Stacking Model saved to 'stacking_model.joblib'")

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
        f.write(f"Fast Model Accuracy: {fast_acc}\n")
        f.write(f"Stacking Ensemble Accuracy: {deep_acc}\n")
        f.write("\nClassification Report:\n")
        f.write(classification_report(y_test, y_pred))
        f.write("\n--- FEATURE IMPORTANCE ---\n")
        f.write(feat_df.head(20).to_string())

if __name__ == "__main__":
    train_model()
