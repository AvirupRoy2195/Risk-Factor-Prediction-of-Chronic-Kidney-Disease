"""
Minimal KidneyPred App - Fast Loading Version
Skips heavy XAI libraries (SHAP, LIME) for quick startup.
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="KidneyPred AI (Lite)", layout="wide")

st.title("🩺 KidneyPred AI - Lite Version")
st.caption("Fast-loading version without SHAP/LIME visualization")

# Check for model files
@st.cache_resource
def load_models():
    try:
        fast_model = joblib.load('fast_model.joblib')
        scaler = joblib.load('scaler.joblib')
        encoders = joblib.load('encoders.joblib')
        return fast_model, scaler, encoders
    except FileNotFoundError as e:
        st.error(f"Model not found: {e}")
        st.stop()

fast_model, scaler, encoders = load_models()
st.success("✅ Models loaded successfully!")

# Simple prediction interface
st.subheader("Quick Patient Assessment")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", 1, 120, 50)
    sc = st.number_input("Serum Creatinine", 0.1, 20.0, 1.2)
    
with col2:
    hemo = st.number_input("Hemoglobin", 3.0, 20.0, 12.0)
    bp = st.number_input("Blood Pressure (Diastolic)", 40, 200, 80)
    
with col3:
    htn = st.selectbox("Hypertension", ["no", "yes"])
    dm = st.selectbox("Diabetes", ["no", "yes"])

if st.button("🔍 Quick Diagnosis"):
    st.info("This is a minimal demo. For full predictions, use the main app.")
    
    # Simple risk assessment
    risk = 0
    if sc > 1.5:
        risk += 30
    if hemo < 11:
        risk += 20
    if age > 60:
        risk += 15
    if htn == "yes":
        risk += 20
    if dm == "yes":
        risk += 15
    
    if risk < 30:
        st.success(f"✅ Low CKD Risk Score: {risk}%")
    elif risk < 60:
        st.warning(f"⚠️ Moderate CKD Risk Score: {risk}%")
    else:
        st.error(f"🔴 High CKD Risk Score: {risk}%")
    
    st.caption("Note: This is a simplified risk calculator. Full ML predictions require the main app.")
