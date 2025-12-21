import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def train_model():
    # 1. Fetch data
    print("Fetching dataset...")
    risk_factor_prediction_of_chronic_kidney_disease = fetch_ucirepo(id=857)
    X = risk_factor_prediction_of_chronic_kidney_disease.data.features
    y = risk_factor_prediction_of_chronic_kidney_disease.data.targets

    # Combine features and targets for preprocessing
    df = X.copy()
    df['target'] = y

    print(f"Initial shape: {df.shape}")

    # 2. Preprocessing
    # Handle missing values
    # For numerical, use mean; for categorical, use most frequent
    num_cols = df.select_dtypes(include=['number']).columns
    cat_cols = df.select_dtypes(exclude=['number']).columns

    num_imputer = SimpleImputer(strategy='mean')
    df[num_cols] = num_imputer.fit_transform(df[num_cols])

    if len(cat_cols) > 0:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

    # Encode categorical variables
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col])

    # 3. Model Training
    X_processed = df.drop('target', axis=1)
    y_processed = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X_processed, y_processed, test_size=0.2, random_state=42)

    # Scale numeric features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Logistic Regression
    print("\nTraining Logistic Regression...")
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    print("Logistic Regression Accuracy:", accuracy_score(y_test, lr_pred))
    print(classification_report(y_test, lr_pred))

    # Random Forest
    print("\nTraining Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    print("Random Forest Accuracy:", accuracy_score(y_test, rf_pred))
    print(classification_report(y_test, rf_pred))

    # 4. Feature Importance (Random Forest)
    importances = rf.feature_importances_
    feature_names = X_processed.columns
    feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances}).sort_values(by='importance', ascending=False)
    
    print("\nTop 10 Important Features:")
    print(feature_importance_df.head(10))

    # Save results to a file
    with open('model_results.txt', 'w') as f:
        f.write("--- MODEL PERFORMANCE ---\n")
        f.write(f"Logistic Regression Accuracy: {accuracy_score(y_test, lr_pred)}\n")
        f.write(f"Random Forest Accuracy: {accuracy_score(y_test, rf_pred)}\n")
        f.write("\nTop 10 Features:\n")
        f.write(feature_importance_df.head(10).to_string())

if __name__ == "__main__":
    train_model()
