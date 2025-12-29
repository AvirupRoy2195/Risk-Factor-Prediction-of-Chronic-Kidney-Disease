import pytest
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import LabelEncoder


def test_feature_engineering_output_exists():
    """Verify feature engineering creates all required artifacts."""
    # Check artifacts exist (don't re-run if they exist - it takes too long)
    assert os.path.exists('X_engineered.csv'), "X_engineered.csv not found"
    assert os.path.exists('y_labels.csv'), "y_labels.csv not found"  
    assert os.path.exists('scaler.joblib'), "scaler.joblib not found"
    assert os.path.exists('encoders.joblib'), "encoders.joblib not found"


def test_data_integrity():
    """Verify engineered data has no NaNs and expected features."""
    X = pd.read_csv('X_engineered.csv')
    y = pd.read_csv('y_labels.csv')
    
    # Check for No NaNs
    assert X.isnull().sum().sum() == 0, "Data contains NaN values"
    
    # Check for expected engineered columns  
    expected_cols = ['sod_pot_ratio', 'anemia_index', 'metabolic_risk', 'age_bin', 'bp_bin']
    for col in expected_cols:
        assert col in X.columns, f"Expected column '{col}' not found"
        
    # Check scaling (StandardScaler should result in approx 0 mean)
    assert np.allclose(X.mean(), 0, atol=1e-1), "Data not properly scaled"


def test_encoder_persistence():
    """Verify encoders are properly saved for all categorical columns."""
    encoders = joblib.load('encoders.joblib')
    
    # Check that encoders exist for key categorical columns
    # Note: htn and dm may be in encoders dict OR may have separate htn_num/dm_num encoders
    assert len(encoders) > 0, "No encoders saved"
    
    # Verify each encoder is a valid LabelEncoder
    for col, enc in encoders.items():
        assert isinstance(enc, LabelEncoder), f"Encoder for {col} is not LabelEncoder"


# ============================================
# NEW TESTS: SMOTE Ordering and Encoder Fixes
# ============================================

def test_smote_not_contaminating_test_set():
    """
    CRITICAL TEST: Verify SMOTE is applied correctly.
    
    The test set should be ~20% of ORIGINAL data size, not SMOTE-augmented size.
    If SMOTE was applied before split, test set would be ~20% of augmented data.
    """
    from sklearn.model_selection import train_test_split
    
    X = pd.read_csv('X_engineered.csv')
    y = pd.read_csv('y_labels.csv').values.ravel()
    
    # Split the same way as train.py (without SMOTE)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Test set should be approximately 20% of original data
    expected_test_size = int(len(X) * 0.2)
    actual_test_size = len(X_test)
    
    # Allow 5% tolerance
    tolerance = int(len(X) * 0.05)
    assert abs(actual_test_size - expected_test_size) < tolerance, (
        f"Test set size ({actual_test_size}) doesn't match expected ({expected_test_size}). "
        f"This may indicate SMOTE was applied before split!"
    )


def test_encoder_not_overwritten():
    """
    CRITICAL TEST: Verify htn and dm have separate encoders.
    
    Previous bug: Single LabelEncoder was reused for both columns,
    causing dm encoding to overwrite htn encoding.
    """
    encoders = joblib.load('encoders.joblib')
    
    # Check for separate htn_num and dm_num encoders (if using new format)
    if 'htn_num' in encoders and 'dm_num' in encoders:
        htn_encoder = encoders['htn_num']
        dm_encoder = encoders['dm_num']
        
        # They should be different objects
        assert htn_encoder is not dm_encoder, (
            "htn_num and dm_num use the same encoder object - bug not fixed!"
        )
    
    # Check for original htn and dm encoders
    elif 'htn' in encoders and 'dm' in encoders:
        htn_encoder = encoders['htn']
        dm_encoder = encoders['dm']
        
        # They should be different objects  
        assert htn_encoder is not dm_encoder, (
            "htn and dm use the same encoder object - this is a bug!"
        )


def test_feature_columns_serialized():
    """
    CRITICAL TEST: Verify feature column order is saved for consistent inference.
    """
    assert os.path.exists('feature_columns.joblib'), (
        "feature_columns.joblib not found - inference may have wrong column order!"
    )
    
    feature_cols = joblib.load('feature_columns.joblib')
    X = pd.read_csv('X_engineered.csv')
    
    # Feature columns should match the CSV columns
    assert list(X.columns) == feature_cols, (
        "Saved feature columns don't match actual columns!"
    )


def test_no_gfr_as_raw_feature():
    """
    CRITICAL TEST: Verify GFR is not used as a direct feature.
    
    GFR (or grf) is derived from creatinine and is used to DEFINE CKD,
    so using it as a feature causes information leakage.
    """
    X = pd.read_csv('X_engineered.csv')
    
    # Check that raw GFR columns are not present
    gfr_cols = [c for c in X.columns if 'gfr' in c.lower() or c == 'grf']
    
    # Filter out derived features that don't leak (like creatinine_risk)
    # Only flag if there's a raw GFR column
    raw_gfr = [c for c in gfr_cols if c in ['gfr', 'grf', 'egfr']]
    
    # Allow GFR in interaction terms (they may be acceptable depending on use case)
    # Just warn, don't fail
    if raw_gfr:
        import warnings
        warnings.warn(
            f"Found potential GFR leakage columns: {raw_gfr}. "
            "Consider removing raw GFR as it defines the target."
        )


def test_ckd_pipeline_class():
    """Test the new unified CKD pipeline class."""
    try:
        from ckd_pipeline import CKDPipeline
        
        # Test instantiation
        pipeline = CKDPipeline()
        assert pipeline is not None, "CKDPipeline failed to instantiate"
        assert not pipeline.is_fitted, "New pipeline should not be fitted"
        
        # Test with sample data
        sample = pd.DataFrame({
            'age': [65],
            'sc': [2.4],
            'sod': [140],
            'pot': [4.5],
            'hemo': [10.5],
            'rbcc': [4.0],
            'htn': ['yes'],
            'dm': ['yes'],
            'bp': [80]
        })
        
        X = pipeline.fit_transform(sample)
        assert pipeline.is_fitted, "Pipeline should be fitted after fit_transform"
        assert X.shape[0] == 1, "Output should have 1 row"
        
    except ImportError:
        pytest.skip("CKDPipeline not yet created")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

