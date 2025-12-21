import pandas as pd
import numpy as np
import xgboost as xgb
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

def run_advanced_modeling():
    # 1. Load data
    print("Loading engineered data...")
    X = pd.read_csv('X_engineered.csv')
    y = pd.read_csv('y_labels.csv').values.ravel()
    
    # 2. Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Train XGBoost
    print("Training XGBoost...")
    model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"XGBoost Accuracy: {acc}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # 4. SHAP Analysis
    print("Performing SHAP analysis...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    # Global Interpretation
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.title("SHAP Summary Plot - Feature Impact")
    plt.savefig('shap_summary_plot.png', bbox_inches='tight')
    plt.close()
    
    # Local Interpretation (RCA for first sample in test set)
    plt.figure()
    shap.plots.force(explainer.expected_value, shap_values[0,:], X_test.iloc[0,:], matplotlib=True, show=False)
    plt.title("SHAP RCA for Sample 0")
    plt.savefig('shap_rca_sample_0.png', bbox_inches='tight')
    plt.close()

    # Save Results
    with open('advanced_model_results.txt', 'w') as f:
        f.write("--- ADVANCED MODEL RESULTS ---\n")
        f.write(f"XGBoost Accuracy: {acc}\n")
        f.write("\nClassification Report:\n")
        f.write(classification_report(y_test, y_pred))
        
    print("Advanced modeling and SHAP analysis complete results saved.")

if __name__ == "__main__":
    run_advanced_modeling()
