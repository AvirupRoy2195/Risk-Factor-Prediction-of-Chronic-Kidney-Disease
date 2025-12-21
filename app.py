import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
from streamlit_shap import st_shap

# Set page config
st.set_page_config(page_title="KidneyPred AI", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for Premium Look
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #ff4b4b;
        color: white;
    }
    .stMetric {
        background-color: #1e2130;
        padding: 15px;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_assets():
    try:
        model = joblib.load('stacking_model.joblib')
        scaler = joblib.load('scaler.joblib')
        encoders = joblib.load('encoders.joblib')
        num_imputer = joblib.load('num_imputer.joblib')
        cat_imputer = joblib.load('cat_imputer.joblib')
        return model, scaler, encoders, num_imputer, cat_imputer
    except FileNotFoundError:
        st.error("🚨 Model assets not found! Please run `python feature_engineering.py` and `python model_benchmark_ensemble.py` first.")
        st.stop()

def engineer_features_app(df, le):
    # Apply the same logic as feature_engineering.py
    df_new = df.copy()
    
    # 1. Domain Features
    df_new['sod_pot_ratio'] = df_new['sod'] / (df_new['pot'] + 1e-6)
    df_new['anemia_index'] = df_new['hemo'] * df_new['rbcc']
    df_new['creatinine_gfr_interaction'] = df_new['sc'] * df_new['grf']
    
    # Metabolic & BP (using mock/mapped values since it's a single prediction)
    # Categorical htn/dm should already be numeric or string
    # We use the label encoder mapping if needed, but here we'll assume the input form handles types
    df_new['htn_num'] = 1 if df_new['htn'].iloc[0] == 'yes' else 0
    df_new['dm_num'] = 1 if df_new['dm'].iloc[0] == 'yes' else 0
    df_new['metabolic_risk'] = df_new['htn_num'] * df_new['dm_num']
    
    # BP combined logic
    bp_limit_map = {'normal': 0, 'stage 1': 1, 'stage 2': 2} # simplified example
    df_new['bp_limit_num'] = bp_limit_map.get(df_new['bp limit'].iloc[0].lower(), 0)
    df_new['bp_combined'] = df_new['bp (Diastolic)'] * df_new['bp_limit_num']

    # 2. Binning
    df_new['age_bin'] = pd.cut(df_new['age'], bins=[-1, 18, 40, 60, 150], labels=False)
    df_new['bp_bin'] = pd.cut(df_new['bp (Diastolic)'], bins=[-1, 60, 80, 90, 120, 300], labels=False)
    
    return df_new

def main():
    st.title("🩺 KidneyPred AI")
    st.subheader("Advanced Multi-Ensemble Diagnostic System")
    
    model, scaler, encoders, num_imputer, cat_imputer = load_assets()
    
    # Sidebar Inputs
    st.sidebar.header("Patient Biometrics")
    
    with st.sidebar:
        age = st.slider("Age", 0, 100, 45)
        bp = st.number_input("Blood Pressure (Diastolic)", 0, 200, 80)
        bp_limit = st.selectbox("BP Limit", ['normal', 'stage 1', 'stage 2'])
        sg = st.number_input("Specific Gravity (sg)", 1.0, 1.05, 1.02, step=0.005)
        al = st.number_input("Albumin (al)", 0, 5, 0)
        su = st.number_input("Sugar (su)", 0, 5, 0)
        rbc = st.selectbox("Red Blood Cells", ['normal', 'abnormal'])
        pc = st.selectbox("Pus Cell", ['normal', 'abnormal'])
        pcc = st.selectbox("Pus Cell Clumps", ['present', 'absent'])
        ba = st.selectbox("Bacteria", ['present', 'absent'])
        bgr = st.number_input("Blood Glucose Random", 0, 500, 120)
        bu = st.number_input("Blood Urea", 0, 300, 40)
        sc = st.number_input("Serum Creatinine", 0.0, 15.0, 1.2)
        sod = st.number_input("Sodium", 0, 200, 135)
        pot = st.number_input("Potassium", 0.0, 50.0, 4.5)
        hemo = st.number_input("Hemoglobin", 0.0, 20.0, 15.0)
        pcv = st.number_input("Packed Cell Volume", 0, 60, 44)
        wbcc = st.number_input("White Blood Cell Count", 0, 20000, 7800)
        rbcc = st.number_input("Red Blood Cell Count", 0.0, 10.0, 5.2)
        htn = st.selectbox("Hypertension", ['yes', 'no'])
        dm = st.selectbox("Diabetes Mellitus", ['yes', 'no'])
        cad = st.selectbox("Coronary Artery Disease", ['yes', 'no'])
        appet = st.selectbox("Appetite", ['good', 'poor'])
        pe = st.selectbox("Pedal Edema", ['yes', 'no'])
        ane = st.selectbox("Anemia", ['yes', 'no'])
        grf = st.number_input("GFR (Glomerular Filtration Rate)", 0, 200, 90)
        stage = st.number_input("CKD Stage", 1, 5, 1)
        affected = st.selectbox("Is Affected?", ['yes', 'no'])

    # Prepare input dataframe
    input_data = {
        'age': [age], 'bp (Diastolic)': [bp], 'bp limit': [bp_limit], 'sg': [sg],
        'al': [al], 'su': [su], 'rbc': [rbc], 'pc': [pc], 'pcc': [pcc], 'ba': [ba],
        'bgr': [bgr], 'bu': [bu], 'sod': [sod], 'sc': [sc], 'pot': [pot],
        'hemo': [hemo], 'pcv': [pcv], 'wbcc': [wbcc], 'rbcc': [rbcc],
        'htn': [htn], 'dm': [dm], 'cad': [cad], 'appet': [appet], 'pe': [pe],
        'ane': [ane], 'grf': [grf], 'stage': [stage], 'affected': [affected]
    }
    
    input_df = pd.DataFrame(input_data)
    
    if st.button("Generate Diagnostic Report"):
        # 1. Feature Engineering
        # We pass None or dummy for le if no longer used globally
        processed_df = engineer_features_app(input_df, None)
        
        # 2. Encoding and Scaling
        for col in processed_df.columns:
            if col in encoders:
                le_col = encoders[col]
                # Handle unknown labels if necessary, here we assume standard inputs
                val = str(processed_df[col].iloc[0])
                try:
                    processed_df[col] = le_col.transform([val])[0]
                except:
                    # Fallback for app inputs
                    processed_df[col] = 0 
            elif processed_df[col].dtype == 'object':
                # Emergency fallback for unrecorded cats
                val = str(processed_df[col].iloc[0]).lower()
                processed_df[col] = 1 if val in ['yes', 'normal', 'present', 'good'] else 0
        
        # Ensure column order matches training columns
        train_cols = pd.read_csv('X_engineered.csv').columns
        processed_df = processed_df[train_cols] 
        
        scaled_data = scaler.transform(processed_df)
        
        # 3. Prediction
        prob = model.predict_proba(scaled_data)[0][1]
        prediction = "Positive (CKD Detected)" if prob > 0.5 else "Negative (No CKD)"
        
        # 4. Results Section
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Diagnostic Result", prediction, delta=f"{prob:.2%}", delta_color="inverse")
        with col2:
            st.write("---")
            st.write(f"Confidence Level: {max(prob, 1-prob):.2%}")
            
        # 5. RCA Section (Multi-XAI)
        st.write("---")
        st.write("### 🧬 Advanced Root Cause Analysis (Multi-XAI)")
        
        tab1, tab2, tab3 = st.tabs(["🎯 SHAP Explanation", "🧪 LIME Analysis", "📊 Global Importance"])
        
        X_train_bg = pd.read_csv('X_engineered.csv')
        feature_names = X_train_bg.columns.tolist()
        
        with tab1:
            st.write("#### SHAP (Game Theory Based)")
            try:
                # Use a small background for speed
                bg_subset = X_train_bg.sample(50, random_state=42)
                explainer = shap.KernelExplainer(model.predict_proba, bg_subset)
                shap_vals = explainer.shap_values(scaled_data)
                st_shap(shap.force_plot(explainer.expected_value[1], shap_vals[0][:,1], scaled_data, feature_names=feature_names))
            except Exception as e:
                st.error(f"SHAP Error: {e}")

        with tab2:
            st.write("#### LIME (Local Surrogate Based)")
            try:
                # LIME Tabular Explainer
                lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                    training_data=X_train_bg.values,
                    feature_names=feature_names,
                    class_names=['No CKD', 'CKD'],
                    mode='classification'
                )
                
                # Explain the instance
                exp = lime_explainer.explain_instance(
                    data_row=scaled_data[0],
                    predict_fn=model.predict_proba,
                    num_features=10
                )
                
                # Display LIME result
                st.write("Local feature contributions to this prediction:")
                fig = exp.as_pyplot_figure()
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
            except Exception as e:
                st.error(f"LIME Error: {e}")

        with tab3:
            st.write("#### Global Feature Importance")
            # We can use the Random Forest or XGBoost component's importance for speed
            # Or show the saved model results
            try:
                # If we have the XGBoost component within the stack, we can show it
                # For simplicity, we'll display the Top 10 from our previous training
                top_features = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': np.random.rand(len(feature_names)) # Mock if not readily available
                }).sort_values('Importance', ascending=False).head(10)
                
                # But actually, let's use the model's feature importance if it exists
                # Since it's a stack, we'll use a representative baseline
                st.info("Showing global feature impact across all patients.")
                st.image('shap_summary_plot.png')
            except:
                st.warning("Global summary plot not found.")
            
        st.success("Multi-XAI Diagnostic Analysis Complete.")

if __name__ == "__main__":
    main()
