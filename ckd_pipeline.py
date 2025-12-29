"""
CKD Pipeline - Unified Feature Engineering for Training and Inference

This module ensures consistent feature transformation between training 
and inference, preventing feature mismatch bugs.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import joblib
from typing import Dict, List, Optional


class CKDPipeline:
    """
    Unified pipeline for CKD prediction that ensures consistent 
    feature engineering between training and inference.
    
    Usage:
        # Training
        pipeline = CKDPipeline()
        X_processed = pipeline.fit_transform(df_train)
        pipeline.save('ckd_pipeline.joblib')
        
        # Inference
        pipeline = CKDPipeline.load('ckd_pipeline.joblib')
        X_processed = pipeline.transform(df_new)
    """
    
    def __init__(self):
        self.encoders: Dict[str, LabelEncoder] = {}
        self.scaler: Optional[StandardScaler] = None
        self.num_imputer: Optional[SimpleImputer] = None
        self.cat_imputer: Optional[SimpleImputer] = None
        self.feature_columns: Optional[List[str]] = None
        self.is_fitted = False
    
    def _create_domain_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create domain-specific medical features."""
        df = df.copy()
        
        # Electrolyte balance
        if 'sod' in df.columns and 'pot' in df.columns:
            df['sod_pot_ratio'] = df['sod'] / (df['pot'] + 1e-6)
        
        # Anemia indicator
        if 'hemo' in df.columns and 'rbcc' in df.columns:
            df['anemia_index'] = df['hemo'] * df['rbcc']
        
        # Creatinine-based risk (avoiding GFR leakage)
        if 'sc' in df.columns and 'age' in df.columns:
            # Use creatinine and age for risk estimation (not GFR directly)
            df['creatinine_risk'] = df['sc'] / (df['age'].clip(lower=1) / 10)
        
        # Metabolic risk (HTN + DM interaction)
        if 'htn' in df.columns and 'dm' in df.columns:
            # Map to numeric - using explicit mapping
            htn_map = {'yes': 1, 'no': 0, 'Yes': 1, 'No': 0, '1': 1, '0': 0, 1: 1, 0: 0}
            dm_map = {'yes': 1, 'no': 0, 'Yes': 1, 'No': 0, '1': 1, '0': 0, 1: 1, 0: 0}
            
            df['htn_num'] = df['htn'].map(htn_map).fillna(0).astype(int)
            df['dm_num'] = df['dm'].map(dm_map).fillna(0).astype(int)
            df['metabolic_risk'] = df['htn_num'] * df['dm_num']
        
        # Age binning
        if 'age' in df.columns:
            df['age_bin'] = pd.cut(df['age'], bins=[-1, 18, 40, 60, 150], labels=False)
        
        # Blood pressure binning
        bp_col = 'bp' if 'bp' in df.columns else 'bp (Diastolic)' if 'bp (Diastolic)' in df.columns else None
        if bp_col:
            df['bp_bin'] = pd.cut(df[bp_col], bins=[-1, 60, 80, 90, 120, 300], labels=False)
        
        return df
    
    def fit_transform(self, df: pd.DataFrame, target_col: str = 'target') -> pd.DataFrame:
        """
        Fit the pipeline on training data and transform it.
        
        Args:
            df: Training DataFrame
            target_col: Name of target column to exclude
            
        Returns:
            Transformed feature DataFrame
        """
        df = df.copy()
        
        # Extract target if present
        y = None
        if target_col in df.columns:
            y = df[target_col]
            df = df.drop(columns=[target_col])
        
        # Create domain features
        df = self._create_domain_features(df)
        
        # Identify column types
        num_cols = df.select_dtypes(include=['number']).columns.tolist()
        cat_cols = df.select_dtypes(exclude=['number']).columns.tolist()
        
        # Numeric imputation
        if num_cols:
            self.num_imputer = SimpleImputer(strategy='mean')
            df[num_cols] = self.num_imputer.fit_transform(df[num_cols])
        
        # Categorical imputation and encoding
        if cat_cols:
            self.cat_imputer = SimpleImputer(strategy='most_frequent')
            df[cat_cols] = self.cat_imputer.fit_transform(df[cat_cols])
            
            # Label encode each categorical column
            for col in cat_cols:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.encoders[col] = le
        
        # Store feature columns order
        self.feature_columns = df.columns.tolist()
        
        # Scale
        self.scaler = StandardScaler()
        df_scaled = pd.DataFrame(
            self.scaler.fit_transform(df),
            columns=df.columns
        )
        
        self.is_fitted = True
        return df_scaled
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using fitted pipeline.
        
        Args:
            df: New DataFrame to transform
            
        Returns:
            Transformed feature DataFrame with same columns as training
        """
        if not self.is_fitted:
            raise ValueError("Pipeline not fitted. Call fit_transform first or load a saved pipeline.")
        
        df = df.copy()
        
        # Remove target if present
        if 'target' in df.columns:
            df = df.drop(columns=['target'])
        
        # Create domain features
        df = self._create_domain_features(df)
        
        # Ensure all expected columns exist (fill missing with 0)
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0
        
        # Identify column types
        num_cols = [c for c in df.columns if df[c].dtype in ['int64', 'float64', 'int32', 'float32']]
        cat_cols = [c for c in df.columns if c in self.encoders]
        
        # Numeric imputation
        if self.num_imputer and num_cols:
            # Only impute columns that imputer knows
            known_num = [c for c in num_cols if c in self.num_imputer.feature_names_in_]
            if known_num:
                df[known_num] = self.num_imputer.transform(df[known_num])
        
        # Categorical imputation and encoding
        for col in cat_cols:
            if col in self.encoders:
                le = self.encoders[col]
                # Handle unseen labels
                val = str(df[col].iloc[0]) if len(df) > 0 else '0'
                if val in le.classes_:
                    df[col] = le.transform(df[col].astype(str))
                else:
                    df[col] = 0  # Default for unknown
        
        # Ensure column order matches training
        df = df[self.feature_columns]
        
        # Scale
        df_scaled = pd.DataFrame(
            self.scaler.transform(df),
            columns=df.columns
        )
        
        return df_scaled
    
    def save(self, path: str):
        """Save pipeline to file."""
        joblib.dump(self, path)
        print(f"✅ Pipeline saved to '{path}'")
    
    @staticmethod
    def load(path: str) -> 'CKDPipeline':
        """Load pipeline from file."""
        pipeline = joblib.load(path)
        print(f"✅ Pipeline loaded from '{path}'")
        return pipeline


# Convenience function for backward compatibility
def get_pipeline(path: str = 'ckd_pipeline.joblib') -> CKDPipeline:
    """Load or create a new CKD pipeline."""
    try:
        return CKDPipeline.load(path)
    except FileNotFoundError:
        print("⚠️ No saved pipeline found. Creating new one.")
        return CKDPipeline()


if __name__ == "__main__":
    # Test the pipeline
    print("Testing CKD Pipeline...")
    
    # Create sample data
    test_data = pd.DataFrame({
        'age': [65, 45],
        'sc': [2.4, 1.1],
        'sod': [140, 138],
        'pot': [4.5, 4.2],
        'hemo': [10.5, 14.0],
        'rbcc': [4.0, 5.2],
        'htn': ['yes', 'no'],
        'dm': ['yes', 'no'],
        'bp': [80, 70]
    })
    
    pipeline = CKDPipeline()
    X = pipeline.fit_transform(test_data)
    
    print(f"Transformed shape: {X.shape}")
    print(f"Features: {list(X.columns)}")
    
    # Test single row transform
    single = pd.DataFrame([test_data.iloc[0]])
    X_single = pipeline.transform(single)
    print(f"Single row transform: {X_single.shape}")
    
    # Save and reload
    pipeline.save('test_pipeline.joblib')
    loaded = CKDPipeline.load('test_pipeline.joblib')
    X_loaded = loaded.transform(single)
    print(f"Loaded pipeline works: {X_loaded.shape}")
    
    import os
    os.remove('test_pipeline.joblib')
    print("✅ All tests passed!")
