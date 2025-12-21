import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
import statsmodels.api as sm
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

def run_diagnostics():
    # 1. Fetch data
    print("Fetching dataset...")
    risk_factor_prediction = fetch_ucirepo(id=857)
    X = risk_factor_prediction.data.features
    y = risk_factor_prediction.data.targets

    # Preprocess for diagnostics (Statsmodels requires no NaN and numeric data)
    df = X.copy()
    num_cols = df.select_dtypes(include=['number']).columns
    cat_cols = df.select_dtypes(exclude=['number']).columns

    imputer = SimpleImputer(strategy='mean')
    df[num_cols] = imputer.fit_transform(df[num_cols])
    
    cat_imputer = SimpleImputer(strategy='most_frequent')
    df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col])

    # Add constant for statsmodels
    X_diag = sm.add_constant(df)

    # 1. Multicollinearity (VIF)
    print("\n--- Multicollinearity (VIF) ---")
    vif_data = pd.DataFrame()
    vif_data["feature"] = X_diag.columns
    vif_data["VIF"] = [variance_inflation_factor(X_diag.values, i) for i in range(len(X_diag.columns))]
    print(vif_data.sort_values(by="VIF", ascending=False).head(10))

    # 2. Autocorrelation (Durbin-Watson)
    # Target encoding for regression check
    y_encoded = le.fit_transform(y.values.ravel())
    model = sm.OLS(y_encoded, X_diag).fit()
    dw_stat = durbin_watson(model.resid)
    print(f"\n--- Autocorrelation (Durbin-Watson) ---\nDW Statistic: {dw_stat}")
    print("(Values near 2 indicate no autocorrelation; < 1.5 or > 2.5 may be problematic)")

    # 3. Heteroskedasticity (Breusch-Pagan)
    print("\n--- Heteroskedasticity (Breusch-Pagan) ---")
    bp_test = het_breuschpagan(model.resid, X_diag.values)
    labels = ['LM Statistic', 'LM-Test p-value', 'F-Statistic', 'F-Test p-value']
    print(dict(zip(labels, bp_test)))

    # Save diagnostics to file
    with open('diagnostics_results.txt', 'w') as f:
        f.write("--- DIAGNOSTICS ---\n")
        f.write(vif_data.to_string())
        f.write(f"\n\nDW Statistic: {dw_stat}\n")
        f.write(f"Breusch-Pagan p-value: {bp_test[1]}\n")

if __name__ == "__main__":
    run_diagnostics()
