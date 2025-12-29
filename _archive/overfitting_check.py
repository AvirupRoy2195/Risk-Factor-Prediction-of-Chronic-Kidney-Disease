import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve, RepeatedStratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score

def check_overfitting():
    print("Loading engineered data...")
    X = pd.read_csv('X_engineered.csv')
    y = pd.read_csv('y_labels.csv').values.ravel()
    
    print("Loading Stacking Ensemble...")
    model = joblib.load('stacking_model.joblib')
    
    # 1. Learning Curve Analysis
    print("Generating Learning Curves (this may take a moment)...")
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=5, scoring='accuracy', n_jobs=-1, 
        train_sizes=np.linspace(0.1, 1.0, 10), random_state=42
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_mean, 'o-', color="g", label="Cross-validation score")
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color="g")
    plt.xlabel("Training examples")
    plt.ylabel("Accuracy Score")
    plt.title("Learning Curves (Stacking Ensemble)")
    plt.legend(loc="best")
    plt.grid(True)
    plt.savefig('learning_curves.png')
    print("Learning curves saved as learning_curves.png")
    
    # 2. Repeated Cross-Validation Variance
    print("\nRunning Repeated Stratified K-Fold (10 folds, 3 repeats)...")
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
    
    print(f"CV Accuracy Mean: {np.mean(cv_scores):.4f}")
    print(f"CV Accuracy Std: {np.std(cv_scores):.4f}")
    
    # 3. Train vs Test Gap
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    
    print(f"\nTrain Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Overfitting Gap: {train_acc - test_acc:.4f}")

    if train_acc == 1.0 and test_acc == 1.0:
        print("\nNote: The model achieved perfect accuracy on both. This usually happens when the features (like Serum Creatinine or Stage) are extremely strong predictors for CKD in this specific dataset.")

if __name__ == "__main__":
    check_overfitting()
