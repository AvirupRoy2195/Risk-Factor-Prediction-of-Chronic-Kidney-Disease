"""
Quick diagnostic script to test app startup issues.
Runs each import step-by-step with timing.
"""
import time
import sys

def timed_import(name, module_name):
    start = time.time()
    try:
        exec(f"import {module_name}")
        elapsed = time.time() - start
        print(f"✅ {name}: {elapsed:.2f}s")
        return True
    except Exception as e:
        elapsed = time.time() - start
        print(f"❌ {name}: {e} ({elapsed:.2f}s)")
        return False

print("="*50)
print("APP STARTUP DIAGNOSTIC")
print("="*50)
print()

# Core imports
print("1. Core Python imports...")
timed_import("pandas", "pandas")
timed_import("numpy", "numpy")

print("\n2. Heavy ML imports...")
timed_import("sklearn", "sklearn")
timed_import("xgboost", "xgboost")

print("\n3. Streamlit...")
timed_import("streamlit", "streamlit")

print("\n4. XAI libraries...")
timed_import("shap", "shap")
timed_import("lime", "lime")

print("\n5. Loading model files...")
import joblib
import os

models = [
    ('scaler.joblib', 'Scaler'),
    ('encoders.joblib', 'Encoders'),
    ('num_imputer.joblib', 'Num Imputer'),
    ('cat_imputer.joblib', 'Cat Imputer'),
    ('fast_model.joblib', 'Fast Model'),
    ('stacking_model.joblib', 'Stacking Model (LARGE)'),
]

for fname, desc in models:
    if os.path.exists(fname):
        start = time.time()
        try:
            obj = joblib.load(fname)
            elapsed = time.time() - start
            size_mb = os.path.getsize(fname) / (1024*1024)
            print(f"✅ {desc}: {elapsed:.2f}s ({size_mb:.1f} MB)")
        except Exception as e:
            print(f"❌ {desc}: {e}")
    else:
        print(f"⚠️ {desc}: FILE NOT FOUND")

print("\n6. Custom modules...")
custom_modules = [
    ('pdf_analyzer', 'PDFAnalyzer'),
    ('rag_engine', 'RAGEngine'),
    ('pipeline', 'Pipeline'),
]

for mod, desc in custom_modules:
    start = time.time()
    try:
        exec(f"from {mod} import *")
        elapsed = time.time() - start
        print(f"✅ {desc}: {elapsed:.2f}s")
    except Exception as e:
        elapsed = time.time() - start
        print(f"❌ {desc}: {e} ({elapsed:.2f}s)")

print("\n" + "="*50)
print("DIAGNOSTIC COMPLETE")
print("="*50)
