"""
Fast Startup Wrapper for KidneyPred AI
Shows progress while loading heavy libraries.
"""
import sys
import os

# Force unbuffered output
os.environ['PYTHONUNBUFFERED'] = '1'

def show_progress(msg):
    print(f"⏳ {msg}...", flush=True)

print("="*50, flush=True)
print("🚀 KIDNEYPRED AI - STARTUP", flush=True)
print("="*50, flush=True)
print(flush=True)

# Step 1: Core imports
show_progress("Loading core libraries")
import pandas as pd
import numpy as np
print("   ✅ pandas, numpy loaded", flush=True)

# Step 2: Streamlit
show_progress("Loading Streamlit")
import streamlit.web.bootstrap as bootstrap
print("   ✅ streamlit loaded", flush=True)

# Step 3: Check model files exist
show_progress("Checking model files")
required_files = [
    'fast_model.joblib',
    'stacking_model.joblib', 
    'scaler.joblib',
    'encoders.joblib'
]
missing = [f for f in required_files if not os.path.exists(f)]
if missing:
    print(f"   ❌ MISSING: {missing}", flush=True)
    print("   Run: python train.py", flush=True)
    sys.exit(1)
else:
    print("   ✅ All model files present", flush=True)

# Step 4: Start Streamlit
print(flush=True)
print("="*50, flush=True)
print("🎯 Starting Streamlit server...", flush=True)
print("   Open: http://localhost:8501", flush=True)
print("="*50, flush=True)
print(flush=True)

# Run streamlit
if __name__ == "__main__":
    sys.argv = ["streamlit", "run", "app.py", "--server.headless", "true"]
    bootstrap.run("app.py", "", [], {})
