import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

def engineer_features():
    # Fetch data
    print("Fetching dataset for feature engineering...")
    risk_factor_prediction = fetch_ucirepo(id=857)
    X = risk_factor_prediction.data.features
    y = risk_factor_prediction.data.targets
    
    df = X.copy()
    
    # 1. Identification and Coercion
    print("Coerced types...")
    # These features are expected to be numeric based on UCI info
    expected_num = ['bp (Diastolic)', 'sg', 'al', 'su', 'bgr', 'bu', 'sod', 'sc', 'pot', 'hemo', 'pcv', 'rbcc', 'wbcc', 'grf', 'age']
    # These are categorical and should be treated as such
    expected_cat = ['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane', 'affected', 'stage']
    
    for col in expected_num:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    for col in expected_cat:
        if col in df.columns:
            df[col] = df[col].astype(str).replace('nan', np.nan) # Ensure they are strings and handle NaNs

    num_cols = df.select_dtypes(include=['number']).columns
    cat_cols = df.select_dtypes(exclude=['number']).columns
    
    print(f"Num cols: {len(num_cols)}, Cat cols: {len(cat_cols)}")
    
    # 2. Imputation
    num_imputer = SimpleImputer(strategy='mean', keep_empty_features=True)
    if not num_cols.empty:
        imputed_num = num_imputer.fit_transform(df[num_cols])
        print(f"Imputed num shape: {imputed_num.shape}")
        df[num_cols] = imputed_num
    
    cat_imputer = SimpleImputer(strategy='most_frequent', keep_empty_features=True)
    if not cat_cols.empty:
        imputed_cat = cat_imputer.fit_transform(df[cat_cols])
        print(f"Imputed cat shape: {imputed_cat.shape}")
        df[cat_cols] = imputed_cat
    
    # 3. Advanced Feature Creation
    print("Creating new features...")
    
    # Safely create features
    df['sod_pot_ratio'] = df['sod'] / (df['pot'] + 1e-6)
    df['anemia_index'] = df['hemo'] * df['rbcc']
    df['creatinine_gfr_interaction'] = df['sc'] * df['grf']
    
    # Metabolic Risk: Interaction between Hypertension and Diabetes
    le = LabelEncoder()
    df['htn_num'] = le.fit_transform(df['htn'].astype(str))
    df['dm_num'] = le.fit_transform(df['dm'].astype(str))
    df['metabolic_risk'] = df['htn_num'] * df['dm_num']
    
    # BP Marker
    df['bp_limit_num'] = le.fit_transform(df['bp limit'].astype(str))
    df['bp_combined'] = df['bp (Diastolic)'] * df['bp_limit_num']

    # 4. Binning
    print("Binning features...")
    # Age binning: 0-100+
    df['age_bin'] = pd.cut(df['age'], bins=[-1, 18, 40, 60, 150], labels=False)
    # Blood Pressure binning: 0-200+
    df['bp_bin'] = pd.cut(df['bp (Diastolic)'], bins=[-1, 60, 80, 90, 120, 300], labels=False)

    # 5. Final Encoding and Scaling
    current_cat = df.select_dtypes(exclude=['number']).columns
    encoders = {}
    for col in current_cat:
        le_col = LabelEncoder()
        df[col] = le_col.fit_transform(df[col].astype(str))
        encoders[col] = le_col
        
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    
    y_le = LabelEncoder()
    y_encoded = y_le.fit_transform(y.values.ravel())
    y_final = pd.DataFrame(y_encoded, columns=['target'])
    
    X_scaled.to_csv('X_engineered.csv', index=False)
    y_final.to_csv('y_labels.csv', index=False)
    
    # Save objects for dashboard
    joblib.dump(scaler, 'scaler.joblib')
    joblib.dump(encoders, 'encoders.joblib')
    joblib.dump(y_le, 'target_encoder.joblib')
    joblib.dump(num_imputer, 'num_imputer.joblib')
    joblib.dump(cat_imputer, 'cat_imputer.joblib')
    
    print("Success! Engineered data and objects saved.")

if __name__ == "__main__":
    engineer_features()
