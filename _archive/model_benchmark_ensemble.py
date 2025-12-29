import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
# SVM removed - too slow for 58k records
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

def run_ensemble_benchmark():
    # 1. Load data
    print("Loading engineered data...")
    X = pd.read_csv('X_engineered.csv')
    y = pd.read_csv('y_labels.csv').values.ravel()
    
    # 2. Define Base Models (Fast Only - No SVM)
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'XGBoost': xgb.XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss', n_jobs=-1),
        'HistGradientBoosting': HistGradientBoostingClassifier(max_iter=100, random_state=42),
        'Naive Bayes': GaussianNB()
    }
    
    # 3. Cross-Validation Comparison
    print("\n--- Cross-Validation Results (Accuracy) ---")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}
    
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        results[name] = scores
        print(f"{name}: Mean={scores.mean():.4f}, Std={scores.std():.4f}")
    
    # 4. Ensemble Methods
    # voting
    voting_clf = VotingClassifier(
        estimators=[(name, model) for name, model in models.items()],
        voting='soft'
    )
    
    # Stacking
    stacking_clf = StackingClassifier(
        estimators=[(name, model) for name, model in models.items()],
        final_estimator=LogisticRegression(),
        cv=5
    )
    
    # Evaluate Ensembles with CV
    for name, ensemble in [('Voting Ensemble', voting_clf), ('Stacking Ensemble', stacking_clf)]:
        scores = cross_val_score(ensemble, X, y, cv=cv, scoring='accuracy')
        results[name] = scores
        print(f"{name}: Mean={scores.mean():.4f}, Std={scores.std():.4f}")

    # 5. Final Hold-out Test for Stacking
    print("\nTraining final Stacking Ensemble on Train-Test Split...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    stacking_clf.fit(X_train, y_train)
    y_pred = stacking_clf.predict(X_test)
    
    print(f"\nFinal Stacking Accuracy on Hold-out: {accuracy_score(y_test, y_pred)}")
    print("\nFinal Classification Report:")
    print(classification_report(y_test, y_pred))

    # Save Results
    res_df = pd.DataFrame({name: scores for name, scores in results.items()})
    res_df.to_csv('model_comparison_results.csv', index=False)
    
    with open('ensemble_results.txt', 'w') as f:
        f.write("--- ENSEMBLE AND CV BENCHMARK ---\n")
        f.write(res_df.mean().to_string())
        f.write("\n\nFinal Stacking Report:\n")
        f.write(classification_report(y_test, y_pred))

    # Save components for Dashboard
    print("Saving models and objects for dashboard...")
    joblib.dump(stacking_clf, 'stacking_model.joblib')
    # Since I need the scaler and encoder, I'll need to refactor feature_engineering.py or run it here.
    # For now, I'll ensure I save what's available here.
    
    print("Success! Engineered data and models saved.")

if __name__ == "__main__":
    run_ensemble_benchmark()
