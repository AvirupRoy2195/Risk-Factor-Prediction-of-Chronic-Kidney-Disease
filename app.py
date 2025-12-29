import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import re
from streamlit_shap import st_shap
from pdf_analyzer import PDFAnalyzer
from orchestrator import OrchestratorAgent
from rag_engine import RAGEngine
import os
from dotenv import load_dotenv

def clean_llm_output(text: str) -> str:
    """Clean HTML artifacts from LLM output."""
    text = text.replace('<br>', '\n')
    text = text.replace('<br/>', '\n')
    text = text.replace('<br />', '\n')
    text = text.replace('&nbsp;', ' ')
    text = text.replace('&amp;', '&')
    text = text.replace('&lt;', '<')
    text = text.replace('&gt;', '>')
    text = re.sub(r'<[^>]+>', '', text)
    return text

load_dotenv()

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
        fast_model = joblib.load('fast_model.joblib')
        stacking_model = joblib.load('stacking_model.joblib')
        scaler = joblib.load('scaler.joblib')
        encoders = joblib.load('encoders.joblib')
        num_imputer = joblib.load('num_imputer.joblib')
        cat_imputer = joblib.load('cat_imputer.joblib')
        return fast_model, stacking_model, scaler, encoders, num_imputer, cat_imputer
    except FileNotFoundError as e:
        st.error(f"🚨 Model assets not found ({e})! Please run `python train.py` first.")
        st.stop()

def engineer_features_app(df, le):
    # Apply the same logic as feature_engineering.py
    df_new = df.copy()
    
    # 1. Domain Features
    df_new['sod_pot_ratio'] = df_new['sod'] / (df_new['pot'] + 1e-6)
    df_new['anemia_index'] = df_new['hemo'] * df_new['rbcc']
    df_new['creatinine_gfr_interaction'] = df_new['sc'] * df_new['grf']
    
    # Metabolic & BP (using mock/mapped values since it's a single prediction)
    df_new['htn_num'] = 1 if df_new['htn'].iloc[0] == 'yes' else 0
    df_new['dm_num'] = 1 if df_new['dm'].iloc[0] == 'yes' else 0
    df_new['metabolic_risk'] = df_new['htn_num'] * df_new['dm_num']
    
    # BP combined logic
    bp_limit_map = {'normal': 0, 'stage 1': 1, 'stage 2': 2} 
    df_new['bp_limit_num'] = bp_limit_map.get(df_new['bp limit'].iloc[0].lower(), 0)
    df_new['bp_combined'] = df_new['bp (Diastolic)'] * df_new['bp_limit_num']

    # 2. Binning
    df_new['age_bin'] = pd.cut(df_new['age'], bins=[-1, 18, 40, 60, 150], labels=False)
    df_new['bp_bin'] = pd.cut(df_new['bp (Diastolic)'], bins=[-1, 60, 80, 90, 120, 300], labels=False)
    
    return df_new

def main():
    
    # --- INITIALIZATION (Restoring Critical State) ---
    if 'patient_data' not in st.session_state:
        # Defaults: 0 means UNKNOWN/not provided by user
        st.session_state.patient_data = {
            'age': 0, 'bp': 0, 'bp_limit': 'normal', 'sg': 1.020,
            'al': 0, 'su': 0, 'rbc': 'normal', 'pc': 'normal', 
            'pcc': 'notpresent', 'ba': 'notpresent', 'bgr': 0, 'bu': 0,
            'sc': 0, 'sod': 0, 'pot': 0, 'hemo': 0, 'pcv': 0,
            'wbcc': 0, 'rbcc': 0, 'htn': 'no', 'dm': 'no', 
            'cad': 'no', 'appet': 'good', 'pe': 'no', 'ane': 'no',
            'grf': 0, 'stage': 0, 'chol': 0, 'tg': 0, 'ldl': 0, 
            'hdl': 0, 'ua': 0, 'ca': 0, 'phos': 0, 'hba1c': 0,
            'statin': 0, 'metformin': 0, 'insulin': 0, 'nsaid': 0
        }
    
    if 'report_index' not in st.session_state:
        st.session_state.report_index = None
        st.session_state.report_generated = False
        st.session_state.prediction = None
        st.session_state.prob = None
        st.session_state.scaled_data = None
    
    # Initialize Pipeline Coordinator (singleton for conversation memory)
    if 'pipeline' not in st.session_state:
        from pipeline import get_coordinator
        st.session_state.pipeline = get_coordinator()
    
    # === LAZY AGENT SINGLETONS (avoid memory exhaustion) ===
    # Only initialize when actually used, not at startup
    if 'rag_engine' not in st.session_state:
        st.session_state.rag_engine = None  # Created on first use
    
    if 'sql_agent' not in st.session_state:
        try:
            from sql_agent import SQLAgent
            st.session_state.sql_agent = SQLAgent()
        except Exception as e:
            st.session_state.sql_agent = None
    
    if 'council' not in st.session_state:
        try:
            from council import MedicalCouncil
            st.session_state.council = MedicalCouncil()
        except Exception as e:
            st.session_state.council = None
    
    if 'query_planner' not in st.session_state:
        from query_planner import QueryPlanner
        st.session_state.query_planner = QueryPlanner()

    # Load Assets ONCE
    fast_model, stacking_model, scaler, encoders, num_imputer, cat_imputer = load_assets()

    # --- CATCHY HEADER ---
    st.markdown("# 🩺 **NephroAI** | *Your Renal IQ*")
    st.caption("58k cases. 3 AI doctors. 1 diagnosis.")
    
    # --- SIDEBAR: Chat Export ---
    with st.sidebar:
        st.markdown("### 📥 Export Chat")
        if st.button("📄 Export to PDF", use_container_width=True):
            if "messages" in st.session_state and st.session_state.messages:
                from datetime import datetime
                from report_generator import create_chat_log_pdf
                
                export_date = datetime.now().strftime("%Y-%m-%d")
                pat_name = st.session_state.patient_data.get('name', 'Unknown')
                
                # Generate PDF
                try:
                    pdf_buffer = create_chat_log_pdf(pat_name, st.session_state.messages)
                    
                    st.download_button(
                        label="⬇️ Download PDF Transcript",
                        data=pdf_buffer,
                        file_name=f"nephroai_chat_{pat_name}_{export_date}.pdf",
                        mime="application/pdf"
                    )
                    st.success("PDF Ready for Download!")
                except Exception as e:
                    st.error(f"Export failed: {e}")
            else:
                st.warning("No chat history to export.")
        
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    
    st.divider()
    
    # Deep Think toggle in sidebar (cleaner UI)
    with st.sidebar:
        st.markdown("### ⚙️ Analysis Mode")
        deep_research = st.toggle("🧠 Deep Think", value=False, 
            help="Enable LLM Council + ML synthesis for comprehensive diagnosis")
        if deep_research:
            st.success("🔬 Deep analysis enabled")
        else:
            st.caption("Quick RAG-based responses")
            
        if 'pdf_text' in st.session_state and st.session_state.pdf_text:
             st.success(f"📄 PDF Active ({len(st.session_state.pdf_text)} chars)")
    
    # Hidden analyze button flag (auto-triggered by chat)
    analyze_btn = False  # Will be triggered by chat when patient data exists

    # File Upload Area (Prominent)
    with st.expander("📂 Upload Medical Reports (PDF/Image)", expanded=True):
        uploaded_file = st.file_uploader("Drop your lab report here to auto-fill data", type=['pdf', 'png', 'jpg', 'jpeg'])
        # Force re-process if we have file but no text (handling legacy state or clears)
        should_process = uploaded_file and (
            st.session_state.report_index is None or 
            ('pdf' in uploaded_file.type and 'pdf_text' not in st.session_state)
        )
        
        if should_process:
            file_bytes = uploaded_file.read()
            file_type = uploaded_file.type
            st.session_state.uploaded_image_bytes = file_bytes
            
            # ===============================
            # PATH 1: PDF PROCESSING (Text Extraction)
            # ===============================
            if 'pdf' in file_type:
                with st.spinner("📄 Extracting text from PDF (using LLM-enhanced parser)..."):
                    try:
                        from pdf_analyzer import PDFAnalyzer
                        parser = PDFAnalyzer()
                        
                        # Extract text
                        all_text = parser.extract_text_from_bytes(file_bytes)
                        st.session_state.pdf_text = all_text
                        
                        # Extract medical entities & metadata
                        entities = parser.extract_clinical_entities(all_text)
                        
                        # Update patient data
                        if entities:
                            entity_map = {
                                'creatinine': 'sc', 'gfr': 'grf', 'hemoglobin': 'hemo',
                                'albumin': 'al', 'potassium': 'pot', 'sodium': 'sod', 'bp': 'bp',
                                'name': 'name', 'gender': 'gender', 'report_date': 'report_date',
                                'report_id': 'report_id', 'sample_type': 'sample_type',
                                'age': 'age'
                            }
                            
                            updated_count = 0
                            for key, val in entities.items():
                                if key in entity_map and val is not None:
                                    # Handle numeric conversions if needed, though LLM usually returns correct types
                                    st.session_state.patient_data[entity_map[key]] = val
                                    updated_count += 1
                                    
                            st.success(f"📊 Extracted {updated_count} fields (Metadata & Vitals)")
                            with st.expander("Parsed Data", expanded=False):
                                st.json(entities)
                        
                        # Create Semantic Chunks for RAG
                        chunk_texts = parser.get_semantic_chunks(all_text)
                        st.session_state.document_chunks = chunk_texts
                        st.info(f"✅ Created {len(chunk_texts)} semantic chunks")
                        
                        # Debug Indicator
                        if 'pdf_text' in st.session_state:
                            st.sidebar.success(f"📄 PDF Loaded ({len(st.session_state.pdf_text)} chars)")
                        
                        # Show preview
                        with st.expander("📋 PDF Text Preview", expanded=False):
                            st.text(all_text[:2000] + ("..." if len(all_text) > 2000 else ""))
                        
                        st.session_state.report_index = "Loaded"
                        
                    except Exception as e:
                        st.error(f"PDF parsing error: {str(e)}")
                        import traceback
                        st.error(traceback.format_exc())
                        st.session_state.report_index = "Loaded"
            
            # ===============================
            # PATH 2: IMAGE PROCESSING (Vision LLM)
            # ===============================
            else:
                with st.spinner("🔬 Analyzing image with AI Vision..."):
                    try:
                        from vision_llm import VisionLLM
                        import base64
                        
                        vision = VisionLLM()
                        image_b64 = base64.b64encode(file_bytes).decode('utf-8')
                        
                        st.write("🔍 Sending to Vision AI...")
                        
                        interpretation = vision.analyze_medical_image(
                            image_b64,
                            "Analyze this medical image. If it's a lab report, extract kidney-related values (creatinine, GFR, hemoglobin, albumin). If it's an imaging study, describe findings related to kidney health.",
                            str(st.session_state.patient_data)
                        )
                        
                        st.success("✅ Image analyzed!")
                        with st.expander("🔬 AI Vision Interpretation", expanded=True):
                            st.markdown(interpretation)
                        
                        st.session_state.image_interpretation = interpretation
                        st.session_state.report_index = "Loaded"
                        
                    except Exception as e:
                        st.error(f"Vision analysis error: {e}")
                        st.session_state.report_index = "Loaded"

    # ===============================
    # PATIENT VITALS & VERIFICATION
    # ===============================
    with st.expander("📝 Patient Vitals", expanded=(st.session_state.report_index is None)):
        st.info("👇 Enter or verify patient data below to generate a prediction.")
        
        pat = st.session_state.patient_data
        
        # Patient Identification
        pat['name'] = st.text_input("Patient Name", value=pat.get('name', ''), placeholder="Enter patient name for records...")
        
        # Tabs for organized data entry
        t1, t2, t3, t4, t5 = st.tabs(["🫀 Vitals", "🩸 Blood", "🧪 Urine", "📜 History", "💊 Meds"])
        
        with t1:
            c1, c2, c3 = st.columns(3)
            # Using slider for Age as requested
            pat['age'] = c1.slider("Age (Years)", 0, 100, int(pat.get('age', 50)))
            
            # [Medical Logic]: Diastolic BP < 40 implies shock/crisis, but 0 = 'Missing/Unknown'.
            # We allow 0 so the Imputation Engine can supply the population mean (76 mmHg)
            # derived from the dataset, ensuring robust predictions even with partial data.
            pat['bp'] = c2.number_input("BP Diastolic (mmHg)", 0, 180, int(pat.get('bp', 80)))
            pat['bgr'] = c3.number_input("Blood Glucose Random (mg/dL)", 0, 600, int(pat.get('bgr', 120)))
            
            c4, c5, c6 = st.columns(3)
            pat['bp_limit'] = c4.selectbox("BP Status", ["normal", "stage 1", "stage 2"], index=["normal", "stage 1", "stage 2"].index(pat.get('bp_limit', 'normal')))
            pat['appet'] = c5.selectbox("Appetite", ["good", "poor"], index=["good", "poor"].index(pat.get('appet', 'good')))
            pat['pe'] = c6.selectbox("Pedal Edema", ["no", "yes"], index=["no", "yes"].index(pat.get('pe', 'no')))
            
            pat['ane'] = st.selectbox("Anemia", ["no", "yes"], index=["no", "yes"].index(pat.get('ane', 'no')))

        with t2:
            st.caption("Common Renal Markers")
            b1, b2, b3, b4 = st.columns(4)
            pat['bu'] = b1.number_input("Blood Urea (mg/dL)", 0.0, 300.0, float(pat.get('bu', 0.0)))
            pat['sc'] = b2.number_input("Serum Creatinine (mg/dL)", 0.0, 30.0, float(pat.get('sc', 0.0)))
            pat['sod'] = b3.number_input("Sodium (mEq/L)", 0.0, 200.0, float(pat.get('sod', 0.0)))
            pat['pot'] = b4.number_input("Potassium (mEq/L)", 0.0, 10.0, float(pat.get('pot', 0.0)))
            
            st.caption("Complete Blood Count (CBC)")
            b5, b6, b7, b8 = st.columns(4)
            pat['hemo'] = b5.number_input("Hemoglobin (g/dL)", 0.0, 25.0, float(pat.get('hemo', 0.0)))
            pat['pcv'] = b6.number_input("Packed Cell Volume (%)", 0.0, 100.0, float(pat.get('pcv', 0.0)))
            pat['wbcc'] = b7.number_input("WBC Count (cells/cumm)", 0, 50000, int(pat.get('wbcc', 0)))
            pat['rbcc'] = b8.number_input("RBC Count (millions/cmm)", 0.0, 15.0, float(pat.get('rbcc', 0.0)))

            st.caption("Extended Chemistry Panel")
            b9, b10, b11, b12 = st.columns(4)
            pat['al_serum'] = b9.number_input("Serum Albumin (g/dL)", 0.0, 10.0, float(pat.get('al_serum', 0.0)))
            pat['chol'] = b10.number_input("Cholesterol (mg/dL)", 0.0, 600.0, float(pat.get('chol', 0.0)))
            pat['ua'] = b11.number_input("Uric Acid (mg/dL)", 0.0, 50.0, float(pat.get('ua', 0.0)))
            pat['hba1c'] = b12.number_input("HbA1c (%)", 0.0, 20.0, float(pat.get('hba1c', 0.0)))
            
            b13, b14, b15, b16 = st.columns(4)
            pat['tg'] = b13.number_input("Triglycerides (mg/dL)", 0.0, 1000.0, float(pat.get('tg', 0.0)))
            pat['ldl'] = b14.number_input("LDL (mg/dL)", 0.0, 400.0, float(pat.get('ldl', 0.0)))
            pat['hdl'] = b15.number_input("HDL (mg/dL)", 0.0, 150.0, float(pat.get('hdl', 0.0)))
            pat['ca'] = b16.number_input("Calcium (mg/dL)", 0.0, 20.0, float(pat.get('ca', 0.0)))
            pat['phos'] = st.number_input("Phosphate (mg/dL)", 0.0, 15.0, float(pat.get('phos', 0.0)))

        with t3:
            u1, u2, u3 = st.columns(3)
            # Specific Gravity: Allow precise input (1.000 - 1.050)
            pat['sg'] = u1.number_input("Specific Gravity", 1.000, 1.050, float(pat.get('sg', 1.020)), step=0.005, format="%.3f")
            
            # Albumin & Sugar: 0-5 scale
            pat['al'] = u2.number_input("Albumin (Urine)", 0.0, 5.0, float(pat.get('al', 0.0)), step=1.0)
            pat['su'] = u3.number_input("Sugar (Urine)", 0.0, 5.0, float(pat.get('su', 0.0)), step=1.0)
            
            u4, u5 = st.columns(2)
            pat['rbc'] = u4.selectbox("Red Blood Cells", ["normal", "abnormal"], index=["normal", "abnormal"].index(pat.get('rbc', 'normal')))
            pat['pc'] = u5.selectbox("Pus Cells", ["normal", "abnormal"], index=["normal", "abnormal"].index(pat.get('pc', 'normal')))
            
            u6, u7 = st.columns(2)
            pat['pcc'] = u6.selectbox("Pus Cell Clumps", ["notpresent", "present"], index=["notpresent", "present"].index(pat.get('pcc', 'notpresent')))
            pat['ba'] = u7.selectbox("Bacteria", ["notpresent", "present"], index=["notpresent", "present"].index(pat.get('ba', 'notpresent')))

        with t4:
            h1, h2, h3 = st.columns(3)
            pat['htn'] = h1.selectbox("Hypertension", ["no", "yes"], index=["no", "yes"].index(pat.get('htn', 'no')))
            pat['dm'] = h2.selectbox("Diabetes", ["no", "yes"], index=["no", "yes"].index(pat.get('dm', 'no')))
            pat['cad'] = h3.selectbox("Coronary Artery", ["no", "yes"], index=["no", "yes"].index(pat.get('cad', 'no')))

        with t5:
            m1, m2, m3, m4 = st.columns(4)
            pat['statin'] = 1 if m1.checkbox("Statin", value=bool(pat.get('statin', 0))) else 0
            pat['metformin'] = 1 if m2.checkbox("Metformin", value=bool(pat.get('metformin', 0))) else 0
            pat['insulin'] = 1 if m3.checkbox("Insulin", value=bool(pat.get('insulin', 0))) else 0
            pat['nsaid'] = 1 if m4.checkbox("NSAID Use", value=bool(pat.get('nsaid', 0))) else 0

        # Sync back to session state
        st.session_state.patient_data = pat
        
        if st.button("🚀 Analyze Patient (Tier 1 & 2)", type="primary", use_container_width=True):
            analyze_btn = True

    # =====================
    # TIERED WORKFLOW LOGIC (ROBUST & MULTI-MODAL)
    # =====================
    if analyze_btn:
        st.toast("🔍 Starting Robust Analysis...")
        
        # Determine Data Availability
        pd_data = st.session_state.patient_data
        # Check if non-zero vitals entered
        has_vitals = any(float(pd_data.get(k, 0)) != 0 for k in ['bp', 'sc', 'hemo', 'bgr'])
        has_pdf = 'pdf_text' in st.session_state and st.session_state.pdf_text
        has_vision = 'vision_analysis' in st.session_state and st.session_state.vision_analysis
        
        consensus = "Clinical Analysis (No Vitals)"
        confidence = 0.0
        prob_fast = 0.0
        prob_stack = 0.0
        
        # --- TIER 1: NUMERICAL PREDICTION (Only if Vitals Present) ---
        if has_vitals:
            with st.status("**Tier 1: Robust Model Inference**", expanded=True) as status:
                try:
                    # Imputation defaults (Derived from training data)
                    imps = {
                        'age': 65, 'bp': 76, 'bgr': 151, 'bu': 33.0, 'sc': 1.1, 
                        'sod': 138.0, 'pot': 4.6, 'hemo': 12.6, 'pcv': 39.0, 
                        'wbcc': 8406, 'rbcc': 4.7, 'al_serum': 3.9,
                        'chol': 175.0, 'tg': 136.0, 'ldl': 105.0, 'hdl': 45.0,
                        'ua': 6.0, 'ca': 9.2, 'phos': 3.3, 'hba1c': 5.7 
                    }
                    
                    def get_v(k, allow_zero=False):
                        v = pd_data.get(k, 0)
                        if not allow_zero and (v is None or v == 0):
                            repl = imps.get(k, 0)
                            # Log imputation (skip basic zeros)
                            if k not in ['statin', 'metformin', 'insulin', 'nsaid']:
                                imputed_log.append(f"{k}: {repl}")
                            return repl
                        return v

                    imputed_log = []
                    input_data = {
                        'age': [get_v('age')], 'bp (Diastolic)': [get_v('bp')], 
                        'bp limit': [pd_data.get('bp_limit', 'normal')], 'sg': [pd_data.get('sg', 1.020)],
                        'al': [pd_data.get('al', 0)], 'su': [pd_data.get('su', 0)], 
                        'rbc': [pd_data.get('rbc', 'normal')], 'pc': [pd_data.get('pc', 'normal')], 
                        'pcc': [pd_data.get('pcc', 'notpresent')], 'ba': [pd_data.get('ba', 'notpresent')],
                        'bgr': [get_v('bgr')], 'bu': [get_v('bu')], 
                        'sod': [get_v('sod')], 'sc': [get_v('sc')], 
                        'pot': [get_v('pot')], 'hemo': [get_v('hemo')], 
                        'pcv': [get_v('pcv')], 'wbcc': [get_v('wbcc')], 
                        'rbcc': [get_v('rbcc')], 'htn': [pd_data.get('htn', 'no')], 
                        'dm': [pd_data.get('dm', 'no')], 'cad': [pd_data.get('cad', 'no')], 
                        'appet': [pd_data.get('appet', 'good')], 'pe': [pd_data.get('pe', 'no')],
                        'ane': [pd_data.get('ane', 'no')], 'grf': [pd_data.get('grf', 0)],
                        'chol': [get_v('chol')], 'tg': [get_v('tg')],
                        'ldl': [get_v('ldl')], 'hdl': [get_v('hdl')],
                        'ua': [get_v('ua')], 'ca': [get_v('ca')],
                        'phos': [get_v('phos')], 'hba1c': [get_v('hba1c')],
                        'statin': [pd_data.get('statin', 0)], 'metformin': [pd_data.get('metformin', 0)],
                        'insulin': [pd_data.get('insulin', 0)], 'nsaid': [pd_data.get('nsaid', 0)],
                        'serum_al': [get_v('al_serum')]
                    }
                    
                    if imputed_log:
                        st.info(f"🤖 **Auto-Imputed Missing Data**: {', '.join(imputed_log[:5])}" + (", ..." if len(imputed_log) > 5 else ""))
                    
                    input_df = pd.DataFrame(input_data)
                    processed_df = engineer_features_app(input_df, None)
                    for col in processed_df.columns:
                        if col in encoders:
                            le = encoders[col]
                            if str(processed_df[col].iloc[0]) in le.classes_:
                                processed_df[col] = le.transform([str(processed_df[col].iloc[0])])[0]
                            else:
                                processed_df[col] = 0
                        elif processed_df[col].dtype == 'object':
                            processed_df[col] = 0
                            
                    train_cols = pd.read_csv('X_engineered.csv').columns
                    for c in train_cols:
                        if c not in processed_df.columns: processed_df[c] = 0
                    processed_df = processed_df[train_cols]
                    
                    scaled = scaler.transform(processed_df)
                    st.write("⚡ Step 1/2: XGBoost Inference...")
                    prob_fast = fast_model.predict_proba(scaled)[0][1]
                    pred_fast = "CKD" if prob_fast > 0.5 else "Healthy"
                    
                    st.write("🧠 Step 2/2: Stacking Ensemble Validation...")
                    prob_stack = stacking_model.predict_proba(scaled)[0][1]
                    pred_stack = "CKD" if prob_stack > 0.5 else "Healthy"
                    
                    if pred_fast == pred_stack:
                        consensus = pred_fast
                        confidence = max(prob_fast, prob_stack)
                        agreement = "✅ Models Agree"
                    else:
                        consensus = pred_stack
                        confidence = prob_stack
                        agreement = "⚠️ Conflicting - Using Ensemble"
                        
                    status.update(label="**Tier 1 Complete**", state="complete", expanded=False)
                    
                    # Display Result
                    color = "red" if consensus == "CKD" else "green"
                    st.markdown(f"""
                    <div style="background: linear-gradient(145deg, rgba(20,30,40,0.95), rgba(30,50,60,0.95)); padding: 25px; border-radius: 15px; border-left: 5px solid {color}; margin: 20px 0;">
                        <h2 style='color: {color}; margin:0;'>🔍 {consensus.upper()} ({confidence:.1%})</h2>
                        <p style='margin:5px 0; color: #aaa;'>{agreement}</p>
                    </div>
                    """, unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Tier 1 Prediction Error: {e}")
                    consensus = "Error"
        else:
            st.info("ℹ️ No Vitals provided. Skipping ML Models. Proceeding to Clinical Analysis.")

        # --- TIER 2: INTELLIGENT COUNCIL ---
        # Run if user pressed analyze OR deep research is active
        # Actually logic says 'if deep_research'. But if User provided ONLY PDF, they expect analysis.
        # So we force Tier 2 if PDF/Vision present even if deep_research defaults to True?
        # Assuming deep_research is usually True or checked. 
        should_run_tier2 = deep_research and (has_vitals or has_pdf or has_vision) and consensus != "Error"
        
        if should_run_tier2:
             with st.status("**Tier 2: Cortex Coordinator (The Brain)**", expanded=True) as status2:
                try:
                    from cortex_coordinator import CortexCoordinator
                    coordinator = CortexCoordinator()
                    
                    # Construct Query
                    pat_name = pd_data.get('name', 'Unknown Patient')
                    analysis_query = f"Comprehensive clinical assessment for patient {pat_name}."
                    if has_vitals:
                        analysis_query += " Evaluate renal function and associated risks based on vitals."
                    if has_pdf:
                        analysis_query += " Incorporate findings from the attached medical report."
                    
                    # Run Diagnosis
                    st.write("🧠 Coordinating Agents (Council, Safety, Documents)...")
                    result = coordinator.diagnose(
                        query=analysis_query,
                        patient_data=pd_data if has_vitals else {},
                        ml_prediction=consensus if has_vitals else None,
                        ml_probability=confidence if has_vitals else None,
                        pdf_context=st.session_state.pdf_text if has_pdf else ""
                    )
                    
                    # --- DISPLAY RESULTS ---
                    status2.update(label="**Tier 2 Complete: Clinical Judgment Ready**", state="complete", expanded=False)
                    
                    st.markdown("---")
                    
                    # 1. Main Judgment
                    st.subheader("🧠 Integrated Clinical Judgment")
                    st.markdown(f"""
                    <div style="background-color: rgba(40, 50, 70, 0.5); padding: 20px; border-radius: 10px; border-left: 4px solid #4CAF50;">
                        {result.judgment}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # 2. Evidence Breakdown
                    with st.expander("📂 Evidence & Agent Deliberations", expanded=False):
                        # Council Opinions
                        if 'council' in result.evidence:
                            st.markdown("### 🏛️ Medical Council")
                            council_ev = result.evidence['council']
                            st.info(council_ev.finding)
                            st.caption(f"Confidence: {council_ev.confidence}")
                        
                        # Document Findings
                        if 'document' in result.evidence and has_pdf:
                            st.markdown("### 📄 Document Analysis")
                            doc_ev = result.evidence['document']
                            st.markdown(doc_ev.finding)
                        
                        # ML Findings
                        if 'ml_model' in result.evidence:
                            st.markdown("### 🤖 Predictive Models")
                            ml_ev = result.evidence['ml_model']
                            st.success(f"Prediction: {ml_ev.finding}")

                        # Safety Checks
                        if 'safety' in result.evidence:
                            st.markdown("### 🛡️ Safety & Contraindications")
                            safe_ev = result.evidence['safety']
                            st.warning(safe_ev.finding)

                    # 3. Quality Assurance (Judge)
                    if result.quality_score:
                        with st.expander("📊 Quality Assurance (Self-Reflection)", expanded=False):
                             c1, c2 = st.columns([1, 4])
                             c1.metric("Confidence Score", f"{result.quality_score}/10")
                             c2.caption("The system self-evaluated this diagnosis for consistency and safety.")

                except Exception as e:
                    st.error(f"Tier 2 Analysis Error: {e}")
                    # fallback to basic error display
                    st.exception(e)

                    # SQL AGENT (Only if Vitals present or specific queries needed)
                    # If Has Vitals, query stats.
                    if has_vitals:
                        with st.expander("🔍 Data Cross-Reference (SQL Agent)", expanded=False):
                            try:
                                from sql_agent import SQLAgent
                                if 'sql_agent' not in st.session_state: st.session_state.sql_agent = SQLAgent()
                                sa = st.session_state.sql_agent
                                
                                if pd_data.get('sc', 0) > 1.2:
                                    q = f"What is the prevalence of CKD in patients with Serum Creatinine > {pd_data['sc']}?"
                                elif pd_data.get('htn', 'no') == 'yes':
                                    q = "What percentage of CKD patients have Hypertension?"
                                else:
                                    q = "What is the average Hemoglobin for patients with CKD vs Healthy?"
                                
                                st.write(f"**Query:** *{q}*")
                                sql, ans, _ = sa.query(q)
                                st.success(ans)
                            except Exception as e:
                                st.warning(f"SQL unavailable: {e}")

                    # === EXPORT REPORT (PDF) ===
                    import datetime
                    with st.spinner("Generating PDF Report..."):
                        try:
                            from report_generator import create_pdf_report
                            
                            # Prepare data for PDF
                            pred_data = {
                                "consensus": consensus,
                                "confidence": confidence,
                                "score": evaluation.overall if 'evaluation' in locals() else "N/A"
                            }
                            
                            # Vitals dictionary for report (filter zeroes/empties for clean table)
                            vitals_report = {k: v for k, v in pd_data.items() if v and str(v) != '0' and str(v) != '0.0'}
                            
                            pdf_buffer = create_pdf_report(pat_name, pred_data, final_prescription, vitals_report)
                            
                            st.download_button(
                                label="📄 Download PDF Outcome Report",
                                data=pdf_buffer,
                                file_name=f"KidneyPred_Outcome_{pat_name.replace(' ', '_')}_{datetime.date.today()}.pdf",
                                mime="application/pdf"
                            )
                        except Exception as e_pdf:
                            st.error(f"PDF Generation failed: {e_pdf}")
                            # Fallback to Text Export
                            st.warning("Falling back to text report...")
                            now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
                            report_content = f"### KIDNEYPRED AI REPORT\nDate: {now_str}\nPatient: {pat_name}\n\nStatus: {consensus}\n\n{final_prescription}"
                            st.download_button("📄 Download Text Report", data=report_content, file_name="report_fallback.txt")

                except Exception as e:
                    st.error(f"Tier 2 Error: {e}")

    
    if "messages" not in st.session_state:
        st.session_state.messages = [{
            "role": "assistant", 
            "content": "Ready when you are. 🩺"
        }]

    # Display Chat History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat Input
    if prompt := st.chat_input("Ask me anything about kidney health..."):
        # 1. User Message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # === INTELLIGENT MEMORY (Hippocampus) ===
        from conversation_memory import get_memory
        memory = get_memory(st.session_state)
        memory.add_message("user", prompt)

        # 2. Agent Orchestration with Query Planner
        with st.chat_message("assistant"):
            
            # === TRANSLATION LAYER (any language → English) ===
            from translator import translate_to_english
            english_prompt, detected_lang, was_translated = translate_to_english(prompt)
            
            if was_translated:
                st.caption(f"🌍 Detected: **{detected_lang}** → Translated to English")
                prompt = english_prompt  # Use translated version
            
            # === ML PREDICTION CONTEXT ===
            ml_context = ""
            if st.session_state.patient_data.get('sc', 0) > 0:
                try:
                    # Run ML prediction if we have patient data
                    patient_data = st.session_state.patient_data
                    
                    # Create prediction context
                    if st.session_state.prediction is not None:
                        pred_label = "CKD Positive" if st.session_state.prediction == 1 else "CKD Negative"
                        prob = st.session_state.prob if st.session_state.prob else 0.5
                        ml_context = f"\n\n**[ML Model Prediction: {pred_label} ({prob:.1%} confidence)]**"
                except:
                    pass
            
            with st.spinner("🧠 Planning query execution..."):
                
                # === QUERY UNDERSTANDING AGENT (First Layer) ===
                from query_understanding import QueryUnderstandingAgent
                query_agent = QueryUnderstandingAgent()
                
                # Build chat history for context
                chat_history_str = ""
                for msg in st.session_state.messages[-6:]:
                    role = "Patient" if msg["role"] == "user" else "Doctor"
                    chat_history_str += f"{role}: {msg['content'][:100]}\n"
                
                understanding = query_agent.understand(
                    prompt,
                    patient_context=str(st.session_state.patient_data),
                    conversation_history=chat_history_str
                )
                
                # Show intent understanding
                intent = understanding.get("intent", "general_question")
                refined_query = understanding.get("refined_query", prompt)
                search_keywords = understanding.get("search_keywords", prompt)
                
                st.caption(f"🎯 Intent: **{intent.upper()}** | Query refined for better search")
                
                # Use refined query for downstream processing
                original_prompt = prompt
                optimized_prompt = refined_query
                
                # === INTENT-BASED ROUTING (Brain's Basal Ganglia) ===
                # Override query planner based on understood intent
                
                if intent == "general":
                    # SIMPLE PATH: Skip heavy processing
                    response = f"""Hello! I'm NephroAI, your kidney health assistant. 🩺

I can help you with:
- **CKD Risk Assessment** - Enter your lab values and I'll predict your risk
- **Diet Recommendations** - Personalized meal plans for kidney health
- **Medication Safety** - Check for nephrotoxic drugs
- **Prognosis Questions** - Understanding CKD progression

What would you like to know about your kidney health?"""
                    st.markdown(response)
                    st.session_state.messages.append({{"role": "assistant", "content": response}})
                    st.stop()
                
                elif intent == "prediction":
                    # PREDICTION PATH: Check for required parameters
                    required_params = ['sc', 'age', 'htn', 'dm']
                    missing_params = []
                    for p in required_params:
                        if st.session_state.patient_data.get(p, 0) == 0 or st.session_state.patient_data.get(p) == 'no_data':
                            missing_params.append(p)
                    
                    if missing_params:
                        param_names = {"sc": "Serum Creatinine", "age": "Age", "htn": "Hypertension (yes/no)", "dm": "Diabetes (yes/no)"}
                        missing_list = [param_names.get(p, p) for p in missing_params]
                        response = f"""To predict your CKD risk, I need the following information:

**Missing Parameters:**
{chr(10).join([f"- {p}" for p in missing_list])}

Please provide these values or upload your lab report (PDF/image) to auto-extract them."""
                        st.markdown(response)
                        st.session_state.messages.append({{"role": "assistant", "content": response}})
                        st.stop()
                
                # === QUERY PLANNER (use singleton) ===
                planner = st.session_state.query_planner
                plan = planner.get_execution_plan(prompt, st.session_state.patient_data)
                
                st.caption(f"📍 Route: **{plan['tool'].upper()}** | {plan['reasoning']}")
                
                # === SQL ROUTE: Data Queries ===
                if plan['tool'] == 'sql':
                    try:
                        sql_agent = st.session_state.sql_agent
                        if sql_agent is None:
                            from sql_agent import SQLAgent
                            sql_agent = SQLAgent()
                        
                        st.write("📊 Querying patient database...")
                        sql_query, answer, df = sql_agent.query(prompt)
                        
                        if sql_query:
                            with st.expander("🔍 SQL Query", expanded=False):
                                st.code(sql_query, language="sql")
                        
                        st.markdown(clean_llm_output(answer))
                        
                        if df is not None and not df.empty:
                            with st.expander(f"📋 Results ({len(df)} rows)", expanded=False):
                                st.dataframe(df.head(20))
                        
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                        st.stop()
                        
                    except Exception as e:
                        st.warning(f"SQL query failed: {e}. Falling back to RAG...")
                
                # === HYBRID ROUTE: SQL + RAG ===
                elif plan['tool'] == 'hybrid':
                    st.write("🔀 Hybrid Mode: Combining database analysis with medical knowledge...")
                    hybrid_response = ""
                    
                    try:
                        # 1. Get SQL data
                        from sql_agent import SQLAgent
                        sql_agent = SQLAgent()
                        sql_query, sql_answer, df = sql_agent.query(prompt)
                        
                        data_context = f"**Database Findings:**\n{sql_answer}"
                        if df is not None and not df.empty:
                            data_context += f"\n\nData: {df.head(5).to_markdown()}"
                        
                        # 2. Get RAG knowledge interpretation (lazy init)
                        if st.session_state.rag_engine is None:
                            st.session_state.rag_engine = RAGEngine()
                        rag = st.session_state.rag_engine
                        knowledge_query = f"Based on the following data findings, provide medical interpretation and recommendations:\n\n{data_context}\n\nOriginal question: {prompt}"
                        rag_response = rag.chat_reasoning(knowledge_query, [], None)
                        
                        # 3. Merge responses
                        hybrid_response = f"""### 📊 Data Analysis
{sql_answer}

---

### 🧠 Medical Interpretation
{clean_llm_output(rag_response)}"""
                        
                        if sql_query:
                            with st.expander("🔍 SQL Query Used", expanded=False):
                                st.code(sql_query, language="sql")
                        
                        st.markdown(hybrid_response)
                        st.session_state.messages.append({"role": "assistant", "content": hybrid_response})
                        st.stop()
                        
                    except Exception as e:
                        st.warning(f"Hybrid query failed: {e}. Falling back to RAG...")
                
                # === SIMPLE ROUTE: Greetings ===
                elif plan['tool'] == 'simple':
                    simple_response = "Hello! I'm NephroAI, your kidney health assistant. 🩺\n\nAsk me anything about kidney health, lab values, or patient data. I'll route your question to the right tool automatically."
                    st.markdown(simple_response)
                    st.session_state.messages.append({"role": "assistant", "content": simple_response})
                    st.stop()
                
                # Continue with RAG/Council for other routes
                engine = RAGEngine()
                
                # A. SMART ENTITY EXTRACTION
                import re
                extracted_updates = {}
                
                # Age patterns: "age 69", "69 years old", "I'm 69", "patient is 69"
                age_match = re.search(r'(?:age\s*[:=]?\s*|(?:i\'m|im|patient\s+is|he\s+is|she\s+is)\s*)?(\d{1,3})\s*(?:years?\s*old|yrs?|yo)?', prompt.lower())
                if age_match:
                    extracted_age = int(age_match.group(1))
                    if 1 <= extracted_age <= 120:  # Sanity check
                        st.session_state.patient_data['age'] = extracted_age
                        extracted_updates['age'] = extracted_age

                # Creatinine patterns: "creatinine 1.4", "sc 2.3", "creat: 3.1"
                creat_match = re.search(r'(?:creatinine|creat|sc)\s*[:=]?\s*(\d+\.?\d*)', prompt.lower())
                if creat_match:
                    extracted_sc = float(creat_match.group(1))
                    if 0.1 <= extracted_sc <= 30:  # Sanity check
                        st.session_state.patient_data['sc'] = extracted_sc
                        extracted_updates['creatinine'] = extracted_sc

                # GFR patterns: "gfr 45", "egfr 30", "gfr: 15"
                gfr_match = re.search(r'(?:e?gfr)\s*[:=]?\s*(\d+\.?\d*)', prompt.lower())
                if gfr_match:
                    extracted_gfr = float(gfr_match.group(1))
                    if 0 <= extracted_gfr <= 150:
                        st.session_state.patient_data['grf'] = extracted_gfr
                        extracted_updates['GFR'] = extracted_gfr

                # Hemoglobin: "hb 10.5", "hemoglobin 9.2"
                hb_match = re.search(r'(?:hemo(?:globin)?|hb)\s*[:=]?\s*(\d+\.?\d*)', prompt.lower())
                if hb_match:
                    extracted_hb = float(hb_match.group(1))
                    if 3 <= extracted_hb <= 20:
                        st.session_state.patient_data['hemo'] = extracted_hb
                        extracted_updates['hemoglobin'] = extracted_hb

                # Blood Pressure: "bp 140/90", "blood pressure 160"
                bp_match = re.search(r'(?:bp|blood\s*pressure)\s*[:=]?\s*\d*/?\s*(\d+)', prompt.lower())
                if bp_match:
                    extracted_bp = int(bp_match.group(1))
                    if 40 <= extracted_bp <= 200:
                        st.session_state.patient_data['bp'] = extracted_bp
                        extracted_updates['BP'] = extracted_bp

                # Notify if we extracted something
                if extracted_updates:
                    st.toast(f"📊 Updated patient data: {extracted_updates}")
                    st.info(f"**Extracted & Updated**: {', '.join([f'{k}={v}' for k,v in extracted_updates.items()])}")

                # B. DUCKDUCKGO WEB SEARCH (for Deep Think - uses optimized keywords)
                web_context = ""
                if deep_research and (intent in ["prognosis", "treatment", "diagnosis", "medication"] or len(prompt) > 50):
                    try:
                        from duckduckgo_search import DDGS
                        st.write("🔍 Searching medical literature...")
                        with DDGS() as ddg:
                            # Use optimized search keywords from Query Understanding Agent
                            search_query = f"kidney CKD {search_keywords}"
                            results = list(ddg.text(search_query, max_results=3))
                            if results:
                                web_context = "\n\n**Web Research:**\n"
                                
                                # Translate foreign language citations to English
                                from translator import translate_to_english
                                for r in results:
                                    title = r['title']
                                    body = r['body'][:150]
                                    
                                    # Translate if not English
                                    title_en, title_lang, title_was_translated = translate_to_english(title)
                                    body_en, body_lang, body_was_translated = translate_to_english(body)
                                    
                                    if title_was_translated or body_was_translated:
                                        lang_tag = f" [{title_lang}→EN]"
                                    else:
                                        lang_tag = ""
                                    
                                    web_context += f"- [{title_en}]({r['href']}){lang_tag}: {body_en}...\n"
                                
                                st.markdown(web_context)
                    except ImportError:
                        st.caption("💡 Tip: Install `duckduckgo-search` for web research.")
                    except Exception as e:
                        st.caption(f"Web search unavailable: {e}")

                # === GLOBAL CONTEXT BUILDING (Applies to both Deep & Fast modes) ===
                # 1. Build Conversation History
                chat_history = ""
                recent_msgs = st.session_state.messages[-10:]  # Last 5 turns
                if len(recent_msgs) > 2:
                    chat_history = "\n--- Recent Conversation ---\n"
                    # Exclude current message (which is just 'prompt')
                    for msg in recent_msgs: 
                        if msg['content'] == prompt: continue
                        role = "Patient" if msg["role"] == "user" else "Doctor"
                        chat_history += f"{role}: {msg['content'][:200]}\n"
                    chat_history += "---\n\n"
                
                # 2. Convert session data to string
                patient_str = str(st.session_state.patient_data)
                
                # 3. Construct Context-Aware Prompt
                prompt_with_context = f"{chat_history}Current Question: {prompt}\n\nPatient Data: {patient_str}"
                
                # 4. Inject Uploaded Evidence (PDF/Vision)
                if 'pdf_text' in st.session_state and st.session_state.pdf_text:
                    prompt_with_context += f"\n\n[ATTACHED PDF REPORT CONTENT]:\n{st.session_state.pdf_text[:4000]}..." # Truncate to avoid context overflow
                
                if 'vision_analysis' in st.session_state and st.session_state.vision_analysis:
                    prompt_with_context += f"\n\n[ATTACHED IMAGE ANALYSIS]:\n{st.session_state.vision_analysis}"

                # C. Reasoning / Deep Research
                if deep_research:
                     # Council Deliberation (The "Subconscious Mind" of the Doctor)
                    try:
                        from council import MedicalCouncil
                        council = MedicalCouncil()
                        
                        # 1. Deliberation Phase
                        with st.status("🧠 Council of LLMs Deliberating...", expanded=True) as status:
                            st.write("Dr. Nemotron (Nephrology) is analyzing...")
                            st.write("Dr. Mistral (Diagnosis) is reviewing alternatives...")
                            st.write("Dr. GLM (Pharmacology) is checking safety...")
                            
                            opinions = council.consult(prompt_with_context, patient_str)
                            
                            # Show internal thoughts - use TABS to avoid overlap
                            with st.expander("📋 View Council Deliberations", expanded=False):
                                tab1, tab2, tab3 = st.tabs(["🩺 Dr. Nephro", "🔬 Dr. Diag", "💊 Dr. Pharma"])
                                
                                with tab1:
                                    st.markdown("### Nephrology Assessment")
                                    st.markdown(opinions['nephrologist'][:1500] if len(opinions['nephrologist']) > 1500 else opinions['nephrologist'])
                                
                                with tab2:
                                    st.markdown("### Differential Diagnosis")
                                    st.markdown(opinions['diagnostician'][:1500] if len(opinions['diagnostician']) > 1500 else opinions['diagnostician'])
                                
                                with tab3:
                                    st.markdown("### Medication Safety")
                                    st.markdown(opinions['pharmacologist'][:1500] if len(opinions['pharmacologist']) > 1500 else opinions['pharmacologist'])
                            
                            status.update(label="✅ Consensus Reached", state="complete", expanded=False)

                        # 2. CORTEX COORDINATOR - Unified Doctor-like Judgment
                        # Synthesizes ML + Council + Safety into coherent assessment
                        with st.status("🧠 Synthesizing Clinical Judgment...", expanded=True) as synth_status:
                            st.write("Integrating ML pattern recognition...")
                            st.write("Combining council deliberations...")
                            st.write("Checking safety constraints...")
                            st.write("Generating unified diagnosis...")
                            
                            from cortex_coordinator import CortexCoordinator
                            cortex = CortexCoordinator()
                            
                            # Get ML prediction if available
                            ml_pred = st.session_state.prediction
                            ml_prob = st.session_state.prob
                            
                            # Full cognitive synthesis (use contextualized prompt)
                            diagnosis_result = cortex.diagnose(
                                prompt_with_context,  # Include conversation history
                                st.session_state.patient_data,
                                ml_prediction=ml_pred,
                                ml_probability=ml_prob
                            )
                            # Extract the judgment string from the DiagnosisResult object
                            response = diagnosis_result.judgment
                            
                            synth_status.update(label="✅ Clinical Judgment Ready", state="complete", expanded=False)
                        st.markdown(clean_llm_output(response))
                        
                        # === ML MODEL PREDICTION DISPLAY ===
                        if ml_context and st.session_state.prediction is not None:
                            pred_label = "🔴 CKD Positive" if st.session_state.prediction == 1 else "🟢 CKD Negative"
                            prob = st.session_state.prob if st.session_state.prob else 0.5
                            
                            st.divider()
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.markdown(f"### 🤖 ML Model Prediction: {pred_label}")
                                st.progress(prob, text=f"Confidence: {prob:.1%}")
                            with col2:
                                st.metric("CKD Risk", f"{prob:.0%}")
                        
                        # 3. Agent-as-a-Judge Evaluation
                        try:
                            from judge_agent import JudgeAgent
                            judge = JudgeAgent()
                            
                            with st.expander("📊 Quality Evaluation (Agent-as-a-Judge)", expanded=False):
                                with st.spinner("Evaluating response quality..."):
                                    score = judge.score_response(
                                        response, 
                                        prompt, 
                                        str(st.session_state.patient_data)
                                    )
                                
                                # Display scores as metrics
                                m1, m2, m3, m4, m5 = st.columns(5)
                                m1.metric("Overall", f"{score.overall}/10")
                                m2.metric("Accuracy", f"{score.accuracy}/10")
                                m3.metric("Safety", f"{score.safety}/10")
                                m4.metric("Helpfulness", f"{score.helpfulness}/10")
                                m5.metric("Evidence", f"{score.evidence}/10")
                                
                                # Quality bar
                                quality_color = "🟢" if score.overall >= 8 else "🟡" if score.overall >= 6 else "🔴"
                                st.progress(score.overall / 10, text=f"{quality_color} Quality: {score.overall}/10")
                                
                                # Reasoning
                                st.caption(f"**Judge's Reasoning**: {score.reasoning}")
                                
                                # Suggestions for improvement
                                if score.suggestions:
                                    st.markdown("**Suggestions for improvement:**")
                                    for s in score.suggestions[:3]:
                                        st.markdown(f"- {s}")
                        except Exception as je:
                            st.caption(f"Judge evaluation skipped: {je}")

                    except Exception as e:
                        st.error(f"Council Error: {e}")
                        # Fallback to simple RAG
                        engine = RAGEngine()
                        response = engine.chat_reasoning(prompt_with_context, [], None)
                        st.markdown(clean_llm_output(response))
                else:
                    # FAST / SIMPLE MODE (Single LLM or purely heuristic)
                    # NOW AWARE OF CONTEXT (History, PDF, Data) because we pass prompt_with_context!
                    engine = RAGEngine()
                    response = engine.chat_reasoning(prompt_with_context, [], None)
                    st.markdown(clean_llm_output(response))
                
                # D. Proactive Inference Check
                # If we have enough data, suggest running the model
                if st.session_state.patient_data['sc'] > 0 and st.session_state.patient_data['age'] > 0:
                     if st.button("Run Diagnostic Inference ⚡", key='chat_predict'):
                          # Call shared logic
                          # We need to trigger the generation logic which is in Tab 1
                          # For now, just a toast, user can go to Tab 1 or we auto-run if refactored
                          st.toast("Please switch to 'Prediction & XAI' tab to view the visual report.")

        st.session_state.messages.append({"role": "assistant", "content": response})

        # 3. Session Rating
        with st.popover("Rate this consultation"):
             st.feedback("stars")

if __name__ == "__main__":
    main()
