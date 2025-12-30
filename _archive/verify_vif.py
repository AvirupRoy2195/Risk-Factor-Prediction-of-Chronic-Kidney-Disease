import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor

print("Loading X_engineered.csv (Post-Feature Engineering)...")
try:
    df = pd.read_csv('X_engineered.csv')
    df_num = df.select_dtypes(include=[np.number])
    
    # Check if redundant columns exist
    if 'dm' in df.columns or 'htn' in df.columns:
        print("⚠️ WARNING: 'dm' or 'htn' columns still present! VIF fix failed?")
    else:
        print("✅ Redundant columns 'dm' and 'htn' successfully dropped.")

    # Drop infinite/NaN before VIF
    df_num = df_num.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how='any')
    
    print(f"Calculating VIF for {len(df_num.columns)} columns on {len(df_num)} rows...")
    vif_data = pd.DataFrame()
    vif_data['feature'] = df_num.columns
    vif_data['VIF'] = [variance_inflation_factor(df_num.values, i) for i in range(len(df_num.columns))]
    
    print("\nTop 10 VIF Scores:")
    print(vif_data.sort_values('VIF', ascending=False).head(10))
    
except Exception as e:
    print(f"Error: {e}")
