import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
from streamlit_shap import st_shap
from pdf_analyzer import PDFAnalyzer
from orchestrator import OrchestratorAgent
from rag_engine import RAGEngine
import os
from dotenv import load_dotenv

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
        model = joblib.load('stacking_model.joblib')
        scaler = joblib.load('scaler.joblib')
        encoders = joblib.load('encoders.joblib')
        num_imputer = joblib.load('num_imputer.joblib')
        cat_imputer = joblib.load('cat_imputer.joblib')
        return model, scaler, encoders, num_imputer, cat_imputer
    except FileNotFoundError:
        st.error("🚨 Model assets not found! Please run `python feature_engineering.py` and `python train.py` first.")
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
    st.title("🩺 KidneyPred AI")
    st.subheader("Advanced Multi-Ensemble Diagnostic System")
    
    model, scaler, encoders, num_imputer, cat_imputer = load_assets()
    
    # Initialize Session State
    if 'patient_data' not in st.session_state:
        # Defaults based on healthy/mild values
        st.session_state.patient_data = {
            'age': 45, 'bp': 80, 'bp_limit': 'normal', 'sg': 1.020,
            'al': 0, 'su': 0, 'rbc': 'normal', 'pc': 'normal', 
            'pcc': 'notpresent', 'ba': 'notpresent', 'bgr': 120, 'bu': 40,
            'sc': 1.2, 'sod': 135, 'pot': 4.5, 'hemo': 14.0, 'pcv': 44,
            'wbcc': 7800, 'rbcc': 5.2, 'htn': 'no', 'dm': 'no', 
            'cad': 'no', 'appet': 'good', 'pe': 'no', 'ane': 'no',
            'grf': 90, 'stage': 1 # Ignored by classifier but used for logic
        }
    
    if 'report_generated' not in st.session_state:
        st.session_state.report_generated = False
        st.session_state.prediction = None
        st.session_state.prob = None
        st.session_state.scaled_data = None
        st.session_state.report_index = None

    # Sidebar: PDF Upload & Key Inputs
    st.sidebar.header("📋 Patient Profile")
    
    # 1. PDF Upload Logic
    uploaded_file = st.sidebar.file_uploader("Upload Medical Report (PDF)", type="pdf")
    if uploaded_file and not st.session_state.report_index:
        with st.spinner("Processing PDF: Extracting & Indexing..."):
            with open("temp_report.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            analyzer = PDFAnalyzer()
            text = analyzer.extract_text_from_pdf("temp_report.pdf")
            
            # Extraction
            extracted_data = analyzer.extract_clinical_entities(text)
            
            # Update Session State with extracted values (careful casting)
            for key, val in extracted_data.items():
                if key in st.session_state.patient_data and val is not None:
                     # Simple heuristics for casting could be added here
                     # For now assuming LLM returns correct types or strings for selects
                     st.session_state.patient_data[key] = val
            
            # Indexing for RAG
            st.session_state.report_index = analyzer.create_report_index(text)
            
            # Count updated fields
            updated_count = sum(1 for k, v in extracted_data.items() if v is not None)
            st.sidebar.success(f"✅ PDF Processed! Auto-filled {updated_count} parameters.")
            st.rerun() # Force rerun to update widgets immediately

    # 2. Key Clinical Indicators (Top Features)
    st.sidebar.subheader("Key Indicators")
    
    def update_sidebar(key):
        st.session_state.patient_data[key] = st.session_state[f"sb_{key}"]
    
    # UI Elements synced with session state
    st.sidebar.selectbox("Hypertension", ['yes', 'no'], key='sb_htn', 
                        index=['yes', 'no'].index(st.session_state.patient_data['htn']), on_change=update_sidebar, args=('htn',))
    
    st.sidebar.selectbox("Diabetes Mellitus", ['yes', 'no'], key='sb_dm',
                        index=['yes', 'no'].index(st.session_state.patient_data['dm']), on_change=update_sidebar, args=('dm',))
    
    st.sidebar.selectbox("Pedal Edema", ['yes', 'no'], key='sb_pe',
                        index=['yes', 'no'].index(st.session_state.patient_data['pe']), on_change=update_sidebar, args=('pe',))
    
    st.sidebar.selectbox("Appetite", ['good', 'poor'], key='sb_appet',
                        index=['good', 'poor'].index(st.session_state.patient_data['appet']), on_change=update_sidebar, args=('appet',))

    # Sex Input for GFR (Crucial for CKD-EPI)
    is_male = st.sidebar.toggle("Sex: Male?", value=(st.session_state.patient_data.get('sex', 'Male') == 'Male'))
    st.session_state.patient_data['sex'] = 'Male' if is_male else 'Female'

    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.sidebar.number_input("BP (Diastolic)", value=st.session_state.patient_data['bp'], key='sb_bp', on_change=update_sidebar, args=('bp',))
    with col2:
        bp_opts = ['normal', 'stage 1', 'stage 2']
        st.sidebar.selectbox("BP Status", bp_opts, 
                            index=bp_opts.index(st.session_state.patient_data['bp_limit']) if st.session_state.patient_data['bp_limit'] in bp_opts else 0,
                            key='sb_bp_limit', on_change=update_sidebar, args=('bp_limit',))

    with st.sidebar.expander("Advanced Lab Values (Full List)", expanded=False):
        # Strict casting and bounds to prevent StreamlitJSNumberBoundsError
        st.number_input("Age", min_value=1, max_value=120, value=int(st.session_state.patient_data['age']), step=1, key='sb_age', on_change=update_sidebar, args=('age',))
        st.number_input("Albumin", min_value=0, max_value=5, value=int(st.session_state.patient_data['al']), step=1, key='sb_al', on_change=update_sidebar, args=('al',))
        st.number_input("Sugar", min_value=0, max_value=5, value=int(st.session_state.patient_data['su']), step=1, key='sb_su', on_change=update_sidebar, args=('su',))
        st.number_input("Serum Creatinine", min_value=0.0, max_value=20.0, value=float(st.session_state.patient_data['sc']), step=0.1, key='sb_sc', on_change=update_sidebar, args=('sc',))
        st.number_input("Hemoglobin", min_value=0.0, max_value=25.0, value=float(st.session_state.patient_data['hemo']), step=0.1, key='sb_hemo', on_change=update_sidebar, args=('hemo',))
        
        st.selectbox("Anemia", ['yes', 'no'], key='sb_ane', index=['yes', 'no'].index(st.session_state.patient_data['ane']), on_change=update_sidebar, args=('ane',))
        st.selectbox("Coronary Artery Disease", ['yes', 'no'], key='sb_cad', index=['yes', 'no'].index(st.session_state.patient_data['cad']), on_change=update_sidebar, args=('cad',))
        st.selectbox("Red Blood Cells", ['normal', 'abnormal'], key='sb_rbc', index=['normal', 'abnormal'].index(st.session_state.patient_data['rbc']), on_change=update_sidebar, args=('rbc',))
        st.selectbox("Pus Cell", ['normal', 'abnormal'], key='sb_pc', index=['normal', 'abnormal'].index(st.session_state.patient_data['pc']), on_change=update_sidebar, args=('pc',))
        
        # Less important / Defaults
        st.number_input("Sodium", min_value=0.0, max_value=200.0, value=float(st.session_state.patient_data['sod']), step=1.0, key='sb_sod', on_change=update_sidebar, args=('sod',))
        st.number_input("Potassium", min_value=0.0, max_value=10.0, value=float(st.session_state.patient_data['pot']), step=0.1, key='sb_pot', on_change=update_sidebar, args=('pot',))
        st.number_input("Blood Urea", min_value=0.0, max_value=300.0, value=float(st.session_state.patient_data['bu']), step=1.0, key='sb_bu', on_change=update_sidebar, args=('bu',))

    st.sidebar.divider()
    
    # --- MLOps: System Health ---
    with st.sidebar.expander("🛠️ System Health (MLOps)", expanded=False):
        try:
            from monitoring import ModelMonitor
            monitor = ModelMonitor()
            if st.button("Check Data Drift"):
                report = monitor.check_drift()
                if report['drift_detected']:
                    st.error("⚠️ Data Drift Detected!")
                    st.json(report['details'])
                else:
                    st.success("✅ System Healthy")
                    st.caption("No significant deviation from training data.")
        except Exception as e:
            st.error(f"Monitoring Unavailable: {e}")
    # ----------------------------

    st.sidebar.divider()
    
    # TABS
    tab1, tab2, tab3 = st.tabs(["🚀 Prediction & XAI", "🤖 Agentic Case Manager", "💬 Dr. Chat"])

    with tab1:
        # Prepare Dataframe
        # Note: We excluded 'affected' and 'stage' from training to avoid leaks
        pd_data = st.session_state.patient_data
        
        input_data = {
            'age': [pd_data['age']], 'bp (Diastolic)': [pd_data['bp']], 
            'bp limit': [pd_data['bp_limit']], 'sg': [pd_data['sg']],
            'al': [pd_data['al']], 'su': [pd_data['su']], 
            'rbc': [pd_data['rbc']], 'pc': [pd_data['pc']], 
            'pcc': [pd_data['pcc']], 'ba': [pd_data['ba']],
            'bgr': [pd_data['bgr']], 'bu': [pd_data['bu']], 
            'sod': [pd_data['sod']], 'sc': [pd_data['sc']], 
            'pot': [pd_data['pot']], 'hemo': [pd_data['hemo']], 
            'pcv': [pd_data['pcv']], 'wbcc': [pd_data['wbcc']], 
            'rbcc': [pd_data['rbcc']], 'htn': [pd_data['htn']], 
            'dm': [pd_data['dm']], 'cad': [pd_data['cad']], 
            'appet': [pd_data['appet']], 'pe': [pd_data['pe']],
            'ane': [pd_data['ane']], 'grf': [pd_data['grf']]
            # 'stage' and 'affected' removed
        }
    
        input_df = pd.DataFrame(input_data)
        
        if st.button("Generate Diagnostic Report", key='predict_btn'):
            with st.spinner("Running Advanced Diagnostics..."):
                # 1. Feature Engineering
                processed_df = engineer_features_app(input_df, None)
                
                # 2. Encoding
                for col in processed_df.columns:
                    if col in encoders:
                        le_col = encoders[col]
                        val = str(processed_df[col].iloc[0])
                        # Robust Encoding Logic
                        if val in le_col.classes_:
                             processed_df[col] = le_col.transform([val])[0]
                        else:
                             # Fallback
                             processed_df[col] = 0 # Default to 0 class on mismatch
                    
                    # CATCH-ALL: If still object (string) after encoder check, force encode
                    if processed_df[col].dtype == 'object':
                        try:
                            val = str(processed_df[col].iloc[0]).lower()
                            # Common medical binaries
                            if val in ['yes', 'present', 'normal', 'good', 'abnormal', 'poor', 'notpresent', 'no']:
                                # Contextual mapping (naive but safe for scaler)
                                if val in ['yes', 'present', 'abnormal', 'poor']:
                                     processed_df[col] = 1
                                else:
                                     processed_df[col] = 0
                            else:
                                processed_df[col] = 0
                        except:
                            processed_df[col] = 0
                
                # Align Columns & Ensure Float
                train_cols = pd.read_csv('X_engineered.csv').columns
                processed_df = processed_df[train_cols] 
                
                # Force numeric type to prevent scaler strings
                processed_df = processed_df.apply(pd.to_numeric, errors='coerce').fillna(0) 
                
                scaled_data = scaler.transform(processed_df)
                
                # Predict
                prob = model.predict_proba(scaled_data)[0][1]
                prediction = "Positive (CKD Detected)" if prob > 0.5 else "Negative (No CKD)"
                
                # --- MLOps: Monitoring Log ---
                try:
                    from monitoring import ModelMonitor
                    monitor = ModelMonitor()
                    # Log the *processed* numeric features that went into the model
                    # Convert single row df to dict
                    log_data = processed_df.iloc[0].to_dict()
                    monitor.log_prediction(log_data, prediction, float(prob))
                    print("✅ Prediction logged to inference_logs.csv")
                except Exception as e:
                    print(f"⚠️ Monitoring Error: {e}")
                # -----------------------------

                # Save
                st.session_state.report_generated = True
                st.session_state.prediction = prediction
                st.session_state.prob = prob
                st.session_state.scaled_data = scaled_data
                
        if st.session_state.report_generated:
            # Display Results
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Diagnostic Result", st.session_state.prediction, delta=f"{st.session_state.prob:.2%}", delta_color="inverse")
            with col2:
                st.info(f"Model Confidence: {max(st.session_state.prob, 1-st.session_state.prob):.2%}")
            
            # --- CLINICAL SAFETY NET (Exact GFR) ---
            try:
                from diagnostics import calculate_gfr
                gfr = calculate_gfr(
                    st.session_state.patient_data['sc'], 
                    st.session_state.patient_data['age'], 
                    sex_male=(st.session_state.patient_data.get('sex', 'Male') == 'Male')
                )
                
                if gfr:
                    col_gfr1, col_gfr2 = st.columns([1, 2])
                    with col_gfr1:
                        st.metric(
                            label="Estimated GFR (CKD-EPI)", 
                            value=f"{gfr}", 
                            delta="-Low Kidney Func" if gfr < 60 else "Normal Function",
                            delta_color="inverse"
                        )
                    with col_gfr2:
                        # Logic Check: Warning if Model says "Negative" but GFR is dangerously low
                        if gfr < 60 and "Negative" in st.session_state.prediction:
                            st.error(f"⚠️ **Safety Warning**: eGFR is {gfr} (<60). Even though the AI model predicts 'Negative' (likely due to other healthy markers), this GFR indicates **Stage 3+ CKD**. Please prioritize GFR.")
                        elif gfr < 15:
                            st.error("🚨 **CRITICAL**: Kidneys are failing (Stage 5). Immediate dialysis/transplant consult required.")
            except Exception as e:
                st.warning(f"Could not calc GFR: {e}")
            # ---------------------------------------
                
            st.divider()

            # --- DEBUGGER ---
            with st.expander("🐞 Debug: Inspect Model Input Data", expanded=False):
                st.write("These are the values the model actually sees:")
                st.markdown("### 1. Pre-processing (Raw -> Features)")
                st.dataframe(processed_df)
                
                st.markdown("### 2. Scaled Data (Normalized)")
                st.dataframe(pd.DataFrame(scaled_data, columns=processed_df.columns))
                
                st.write(f"Raw Probability: {st.session_state.prob:.4f}")
            # ----------------
            
            # XAI Section
            st.subheader("🧬 Advanced Root Cause Analysis")
            try:
                xai_tabs = st.tabs(["Global Importance", "Local Explanation (LIME)"])
                with xai_tabs[0]:
                    st.image('shap_summary_plot.png') if os.path.exists('shap_summary_plot.png') else st.warning("Global plot not available.")
                with xai_tabs[1]:
                    st.write("Generating Local Explanation...")
                    # Re-load bg data for LIME
                    X_bg = pd.read_csv('X_engineered.csv')
                    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                            training_data=X_bg.values,
                            feature_names=X_bg.columns.tolist(),
                            class_names=['No CKD', 'CKD'],
                            mode='classification'
                    )
                    exp = lime_explainer.explain_instance(
                            data_row=st.session_state.scaled_data[0],
                            predict_fn=model.predict_proba,
                            num_features=10
                    )
                    st.pyplot(exp.as_pyplot_figure())
            except Exception as e:
                st.error(f"XAI Visualization Error: {e}")

    with tab2:
        st.header("🤖 Agentic Case Manager (Planner + RAG + Feedback)")
        st.markdown("This autonomous agent utilizes the **patient's data**, **PDF report** (if uploaded), and **clinical guidelines** to generate personalised advice.")
        
        if st.session_state.report_generated:
             if st.button("Start Agentic Workflow"):
                orchestrator = OrchestratorAgent()
                
                with st.status("🤖 Agentic Workflow Running...", expanded=True) as status:
                    def update_status(msg):
                        status.write(msg)
                        
                    report = orchestrator.execute_workflow(
                        st.session_state.patient_data, 
                        st.session_state.prediction,
                        report_vector_store=st.session_state.report_index, # Pass the bespoke index
                        status_callback=update_status
                    )
                    status.update(label="✅ Workflow Complete!", state="complete", expanded=False)
                
                st.divider()
                st.subheader(f"📋 Clinical Plan for {st.session_state.patient_data['age']}y Male/Female")
                
                for section in report:
                    with st.expander(f"📌 {section['task']}", expanded=True):
                        st.markdown(section['final_content'])
                        
                        # Critique in a collapsed section for cleaner UI
                        with st.expander("🔍 View Clinical Reasoning & Safety Checks", expanded=False):
                            st.info(section['critique'])
        else:
            st.warning("⚠️ Please generate a prediction in the previous tab first.")

    with tab3:
        st.header("💬 Dr. AI Chat Assistant")
        st.caption("Ask questions about your report, kidney health, or the prediction.")
        
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat Input
        if prompt := st.chat_input("Ask a follow-up question (e.g., 'What should I eat?')"):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    engine = RAGEngine()
                    # Convert session chat history to LangChain format if needed, or pass list
                    # For simplicity, passing recent history pairs
                    history_pairs = []
                    for i in range(0, len(st.session_state.messages)-1, 2):
                        if i+1 < len(st.session_state.messages):
                            history_pairs.append((st.session_state.messages[i]["content"], st.session_state.messages[i+1]["content"]))
                    
                    response = engine.chat_reasoning(
                        prompt, 
                        chat_history=history_pairs,
                        report_vector_store=st.session_state.report_index
                    )
                    st.markdown(response)
            
            # Add assistant message
            st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
