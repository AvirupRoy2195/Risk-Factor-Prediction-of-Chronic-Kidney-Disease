import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os

def engineer_features():
    # --- 1. LOAD DATASETS ---
    dfs = []
    
    # D1: mansoordaku/ckdisease (Primary, Rich Features)
    if os.path.exists('kaggle_ckd.csv'):
        print("✅ Found D1: 'kaggle_ckd.csv'")
        df1 = pd.read_csv('kaggle_ckd.csv')
        # D1 Cleaning
        if 'id' in df1.columns: df1 = df1.drop(columns=['id'])
        df1 = df1.rename(columns={
            'wc': 'wbcc', 'rc': 'rbcc', 'classification': 'class', 'bp': 'bp (Diastolic)'
        })
        df1.columns = [c.split('(')[0].strip().lower() for c in df1.columns]
        dfs.append(df1)

    # D2: miadul/kidney-function-health-dataset
    if os.path.exists('kaggle_health.csv'):
        print("✅ Found D2: 'kaggle_health.csv'")
        df2 = pd.read_csv('kaggle_health.csv')
        df2_mapped = pd.DataFrame()
        df2_mapped['age'] = df2['Age']
        df2_mapped['sc'] = df2['Creatinine']
        df2_mapped['bu'] = df2['BUN']
        df2_mapped['al'] = df2['Protein_in_Urine'].round().astype(int)
        df2_mapped['dm'] = df2['Diabetes'].map({1: 'yes', 0: 'no'})
        df2_mapped['htn'] = df2['Hypertension'].map({1: 'yes', 0: 'no'})
        df2_mapped['class'] = df2['CKD_Status'].map({1: 'ckd', 0: 'notckd'})
        dfs.append(df2_mapped)

    # D4: s3programmerlead/end-stage-renal-disease-esrd-dataset (Rich Feature Set)
    if os.path.exists('kaggle_esrd.csv'):
        print("✅ Found D4: 'kaggle_esrd.csv'")
        df4 = pd.read_csv('kaggle_esrd.csv')
        df4_mapped = pd.DataFrame()
        # Map Columns (Verified Names)
        df4_mapped['age'] = df4['Age']
        # Using Mean Creatinine as it's more representative
        df4_mapped['sc'] = df4['Mean Serum Creatinine (mg/dL)']
        # No BUN in this dataset list, but we have Uric Acid? Skip BU.
        df4_mapped['hemo'] = df4['Hemoglobin (g/dL)']
        
        # LANCET INSIGHT: Serum Albumin is a critical predictor (nutritional marker)
        # We map it to a NEW feature 'serum_al' to distinguish from 'al' (urine)
        df4_mapped['serum_al'] = df4['Albumin (g/dL)']
        
        df4_mapped['bgr'] = df4['Glucose (mg/dL)']
        
        # --- NEW RICH FEATURES ---
        # Lipids
        df4_mapped['chol'] = df4['Cholesterol (mg/dL)']
        df4_mapped['tg'] = df4['Triglyceride (mg/dL)']
        df4_mapped['ldl'] = df4['LDL-C (mg/dL)']
        df4_mapped['hdl'] = df4['HDL-C (mg/dL)']
        
        # Minerals
        df4_mapped['ua'] = df4['Uric Acid (mg/dL)']
        df4_mapped['ca'] = df4['Calcium (mg/dL)']
        df4_mapped['phos'] = df4['Phosphate (mg/dL)']
        
        # Long-term Sugar
        df4_mapped['hba1c'] = df4['HbA1c (%)']
        
        # Medications (Binary 0/1)
        # Assuming dataset uses "Yes"/"No" or 1/0
        def map_bin(x):
            if isinstance(x, str):
                return 1 if x.lower() in ['yes', 'true', '1'] else 0
            return 1 if x == 1 else 0

        df4_mapped['statin'] = df4['Statin'].apply(map_bin)
        df4_mapped['metformin'] = df4['Metformin'].apply(map_bin)
        df4_mapped['insulin'] = df4['Insulin'].apply(map_bin)
        df4_mapped['nsaid'] = df4['NSAID'].apply(map_bin)
        
        # -------------------------
        
        df4_mapped['htn'] = df4['Hypertension'].replace({1: 'yes', 0: 'no', 'Yes': 'yes', 'No': 'no'})
        df4_mapped['cad'] = df4['Coronary Artery Disease'].replace({1: 'yes', 0: 'no'})
        
        df4_mapped['class'] = df4['ESRD Risk'].map({'Yes': 'ckd', 'No': 'notckd'})
        
        # Add to list
        dfs.append(df4_mapped)

    if not dfs:
        print("❌ No datasets found!")
        return

    # Concatenate (Outer Join automatically fills missing cols like 'hemo' with NaN for D2)
    print(f"Merging {len(dfs)} datasets...")
    df = pd.concat(dfs, ignore_index=True)
    print(f"Combined Data Shape: {df.shape}")

    # --- 2. CLEANING & STANDARDIZATION ---
    
    # Target Cleaning
    if 'class' in df.columns:
        # Fix messy labels in Kaggle dataset
        df['class'] = df['class'].replace(to_replace={
            'ckd\t': 'ckd', 
            'ckd': 'ckd', 
            'notckd': 'notckd'
        })
        # Encode
        df['target'] = df['class'].map({'ckd': 1, 'notckd': 0})
        df = df.drop(columns=['class'])
    
    # Clean Categorical columns (remove '\t')
    obj_cols = df.select_dtypes(include=['object']).columns
    for col in obj_cols:
        df[col] = df[col].astype(str).str.replace('\t', '').str.strip()
        df[col] = df[col].replace({'nan': np.nan, 'None': np.nan, 'nan': np.nan})

    # Columns where we expect numbers
    expected_num = ['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wbcc', 'rbcc', 'chol', 'tg', 'ldl', 'hdl', 'ua', 'ca', 'phos', 'hba1c', 'serum_al']
    for col in expected_num:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Identify columns again after merge
    num_cols = df.select_dtypes(include=['number']).columns
    if 'target' in num_cols: num_cols = num_cols.drop('target')
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
    
    # Initialize encoders dict for persistence
    encoders = {}
    
    # Safely create features
    if 'sod' in df.columns and 'pot' in df.columns:
        df['sod_pot_ratio'] = df['sod'] / (df['pot'] + 1e-6)
    
    if 'hemo' in df.columns and 'rbcc' in df.columns:
        df['anemia_index'] = df['hemo'] * df['rbcc']
    
    # Metabolic Risk: Interaction between Hypertension and Diabetes
    # CRITICAL FIX: Use separate encoders to prevent overwriting!
    if 'htn' in df.columns and 'dm' in df.columns:
        le_htn = LabelEncoder()
        le_dm = LabelEncoder()
        df['htn_num'] = le_htn.fit_transform(df['htn'].astype(str))
        df['dm_num'] = le_dm.fit_transform(df['dm'].astype(str))
        df['metabolic_risk'] = df['htn_num'] * df['dm_num']
        # Store separately in encoders dict for persistence
        encoders['htn_num'] = le_htn
        encoders['dm_num'] = le_dm
    
    # 4. Binning
    print("Binning features...")
    # Age binning
    if 'age' in df.columns:
        df['age_bin'] = pd.cut(df['age'], bins=[-1, 18, 40, 60, 150], labels=False)
    # Blood Pressure binning
    if 'bp' in df.columns:
        df['bp_bin'] = pd.cut(df['bp'], bins=[-1, 60, 80, 90, 120, 300], labels=False)

    # 5. Final Encoding and Scaling
    current_cat = df.select_dtypes(exclude=['number']).columns
    # Add to existing encoders dict (don't reinitialize!)
    for col in current_cat:
        le_col = LabelEncoder()
        df[col] = le_col.fit_transform(df[col].astype(str))
        encoders[col] = le_col
        
    # Target Handling
    if 'target' in df.columns:
        y = df['target']
        X_df = df.drop(columns=['target'])
    else:
        y = y
        X_df = df
        
    y_le = LabelEncoder()
    y_encoded = y_le.fit_transform(y.values.ravel())
    y_final = pd.DataFrame(y_encoded, columns=['target'])
    
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_df), columns=X_df.columns)
    
    X_scaled.to_csv('X_engineered.csv', index=False)
    y_final.to_csv('y_labels.csv', index=False)
    
    # Save objects for dashboard
    joblib.dump(scaler, 'scaler.joblib')
    joblib.dump(encoders, 'encoders.joblib')
    joblib.dump(y_le, 'target_encoder.joblib')
    joblib.dump(num_imputer, 'num_imputer.joblib')
    joblib.dump(cat_imputer, 'cat_imputer.joblib')
    
    # CRITICAL: Save feature column order for consistent inference
    joblib.dump(list(X_df.columns), 'feature_columns.joblib')
    print(f\"✅ Saved {len(X_df.columns)} feature columns to 'feature_columns.joblib'\")
    
    print(\"Success! Engineered data and objects saved.\")

if __name__ == "__main__":
    engineer_features()
