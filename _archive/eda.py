import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

def perform_eda():
    # Fetch data
    print("Fetching dataset for EDA...")
    risk_factor_prediction = fetch_ucirepo(id=857)
    X = risk_factor_prediction.data.features
    y = risk_factor_prediction.data.targets
    
    df = X.copy()
    df['target'] = y

    # Impute for visualization
    num_cols = df.select_dtypes(include=['number']).columns
    imputer = SimpleImputer(strategy='mean')
    df[num_cols] = imputer.fit_transform(df[num_cols])
    
    # 1. Correlation Heatmap
    plt.figure(figsize=(15, 10))
    # Encode target for correlation
    temp_df = df.copy()
    le = LabelEncoder()
    temp_df['target'] = le.fit_transform(temp_df['target'])
    # Encode other categoricals
    cat_cols = temp_df.select_dtypes(exclude=['number']).columns
    for col in cat_cols:
        temp_df[col] = le.fit_transform(temp_df[col].astype(str))
        
    corr = temp_df.corr()
    sns.heatmap(corr, annot=False, cmap='coolwarm')
    plt.title("Correlation Heatmap")
    plt.savefig('correlation_heatmap.png')
    plt.close()

    # 2. Non-linearity check: Age vs Target with LOESS-like Trend
    plt.figure(figsize=(10, 6))
    sns.regplot(x='age', y='target', data=temp_df, logistic=True, y_jitter=.03, scatter_kws={'alpha':0.5})
    plt.title("Age vs Target (Logistic Regression Line)")
    plt.savefig('age_vs_target_non_linear.png')
    plt.close()

    # 3. Serum Creatinine vs Target
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='target', y='sc', data=temp_df)
    plt.title("Serum Creatinine (sc) Distribution by Target")
    plt.savefig('sc_vs_target.png')
    plt.close()

    print("EDA plots saved to files.")

if __name__ == "__main__":
    perform_eda()
