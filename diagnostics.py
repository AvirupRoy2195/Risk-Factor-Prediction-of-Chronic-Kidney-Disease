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

def calculate_gfr(creatinine, age, sex_male=True, race_black=False):
    """
    Calculates eGFR using the CKD-EPI 2021 Creatinine Equation.
    
    GFR = 142 * min(Scr/kappa, 1)**alpha * max(Scr/kappa, 1)**-1.200 * 0.9938**Age * 1.012 [if female]
    
    kappa = 0.7 (female), 0.9 (male)
    alpha = -0.241 (female), -0.302 (male)
    """
    try:
        scr = float(creatinine)
        age = float(age)
    except:
        return None
        
    if scr <= 0: return None

    if sex_male:
        kappa = 0.9
        alpha = -0.302
        factor_sex = 1.0
    else:
        kappa = 0.7
        alpha = -0.241
        factor_sex = 1.012
        
    # 2021 Equation does not use race factor, removing for general applicability.
    # Older 2009 equation used 1.159 for Black. 
    # We will stick to the 2021 race-free standard for broad application.
    
    gfr = 142 * (min(scr / kappa, 1) ** alpha) * \
          (max(scr / kappa, 1) ** -1.200) * \
          (0.9938 ** age) * factor_sex
          
    return round(gfr, 1)

def create_diagnostic_report(prediction, confidence, patient_data=None):
    """Generates a text summary of the diagnosis/prediction."""
    msg = f"The model predicts: **{prediction}** with **{confidence:.1%} confidence**."
    
    if patient_data:
        gfr = calculate_gfr(
            patient_data.get('sc', 1.0), 
            patient_data.get('age', 40),
            sex_male=True # Default to male if unknown, usually not critical for MVP app
        )
        if gfr:
            msg += f"\n\n**Calculated eGFR**: {gfr} mL/min/1.73m²"
            if gfr < 60:
                msg += " (⚠️ Indicators of Kidney Disease)"
            else:
                msg += " (Normal Range)"
    
    return msg

if __name__ == "__main__":
    run_diagnostics()
    print(f"\nTest GFR (Male, 50, Scr 1.2): {calculate_gfr(1.2, 50, True)}")
