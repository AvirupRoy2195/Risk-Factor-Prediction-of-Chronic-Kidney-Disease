import pytest
import pandas as pd
import numpy as np
import os
import joblib
from feature_engineering import engineer_features

def test_feature_engineering_output_exists():
    # Run the engineer_features function (it saves files)
    # To keep it isolated, we might want to mock the fetch_ucirepo, 
    # but for a production check, we'll verify the saved outputs.
    if os.path.exists('X_engineered.csv'):
        os.remove('X_engineered.csv')
    
    # We'll just run it as it is since it uses the real data
    # In a more complex CI, we'd use a small mock dataset.
    from feature_engineering import engineer_features
    engineer_features()
    
    assert os.path.exists('X_engineered.csv')
    assert os.path.exists('y_labels.csv')
    assert os.path.exists('scaler.joblib')
    assert os.path.exists('encoders.joblib')

def test_data_integrity():
    X = pd.read_csv('X_engineered.csv')
    y = pd.read_csv('y_labels.csv')
    
    # Check for No NaNs
    assert X.isnull().sum().sum() == 0
    
    # Check for expected engineered columns
    expected_new = ['sod_pot_ratio', 'anemia_index', 'creatinine_gfr_interaction', 'metabolic_risk', 'age_bin', 'bp_bin']
    for col in expected_new:
        assert col in X.columns
        
    # Check scaling (StandardScaler should result in approx 0 mean)
    assert np.allclose(X.mean(), 0, atol=1e-1) # loose check for mean

def test_encoder_persistence():
    encoders = joblib.load('encoders.joblib')
    # Check if key categorical columns are present in encoders
    assert 'rbc' in encoders
    assert 'htn' in encoders
    assert 'dm' in encoders

if __name__ == "__main__":
    pytest.main([__file__])
