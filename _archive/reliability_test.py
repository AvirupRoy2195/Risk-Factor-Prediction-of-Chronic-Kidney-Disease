import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

def test_sensitivity():
    print("Loading model and objects...")
    model = joblib.load('stacking_model.joblib')
    scaler = joblib.load('scaler.joblib')
    encoders = joblib.load('encoders.joblib')
    
    # Load engineered data to get a base case
    X = pd.read_csv('X_engineered.csv')
    base_patient = X.iloc[0:1].copy()
    
    # We want to perturb the "Serum Creatinine" (sc) which is a key feature
    # Note: In the engineered CSV, features are already scaled.
    # To do a real sensitivity test, we should use the raw feature and transform it.
    # However, for a quick robustness check, we can perturb the scaled space or use the app's logic.
    
    # Let's find 'sc' index or name
    sc_col = 'sc'
    original_val = base_patient[sc_col].values[0]
    
    perturbations = np.linspace(-2, 2, 50) # In scaled units (approx -2 to +2 std)
    probabilities = []
    
    print(f"Testing sensitivity for feature: {sc_col}...")
    for p in perturbations:
        temp_patient = base_patient.copy()
        temp_patient[sc_col] = original_val + p
        prob = model.predict_proba(temp_patient)[0][1]
        probabilities.append(prob)
        
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(perturbations, probabilities, marker='o', linestyle='-', color='blue')
    plt.title(f'Model Sensitivity Analysis: {sc_col}')
    plt.xlabel(f'Perturbation (Scaled Units of {sc_col})')
    plt.ylabel('CKD Probability')
    plt.grid(True)
    plt.savefig('sensitivity_analysis_sc.png')
    print("Sensitivity plot saved as sensitivity_analysis_sc.png")

if __name__ == "__main__":
    test_sensitivity()
