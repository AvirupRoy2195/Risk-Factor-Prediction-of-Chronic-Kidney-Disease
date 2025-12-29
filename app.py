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
        if st.button("📄 Export to Markdown", use_container_width=True):
            if "messages" in st.session_state and st.session_state.messages:
                from datetime import datetime
                export_date = datetime.now().strftime("%Y-%m-%d %H:%M")
                
                # Build markdown export
                md_content = f"# NephroAI Consultation Report\n\n"
                md_content += f"**Date**: {export_date}\n\n"
                md_content += f"**Patient Data**: {st.session_state.patient_data}\n\n"
                md_content += "---\n\n## Conversation\n\n"
                
                for msg in st.session_state.messages:
                    role = "🧑 Patient" if msg["role"] == "user" else "🤖 NephroAI"
                    md_content += f"### {role}\n{msg['content']}\n\n"
                
                md_content += "---\n\n*Generated by NephroAI - AI-Assisted CKD Risk Assessment*\n"
                
                st.download_button(
                    label="⬇️ Download Report",
                    data=md_content,
                    file_name=f"nephroai_report_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                    mime="text/markdown"
                )
                st.success("Report ready!")
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
    
    # Hidden analyze button flag (auto-triggered by chat)
    analyze_btn = False  # Will be triggered by chat when patient data exists

    # File Upload Area (Prominent)
    with st.expander("📂 Upload Medical Reports (PDF/Image)", expanded=True):
        uploaded_file = st.file_uploader("Drop your lab report here to auto-fill data", type=['pdf', 'png', 'jpg', 'jpeg'])
        if uploaded_file and st.session_state.report_index is None:
            file_bytes = uploaded_file.read()
            file_type = uploaded_file.type
            st.session_state.uploaded_image_bytes = file_bytes
            
            # ===============================
            # PATH 1: PDF PROCESSING (Text Extraction)
            # ===============================
            if 'pdf' in file_type:
                with st.spinner("📄 Extracting text from PDF..."):
                    try:
                        from document_parser import SemanticDocumentParser
                        parser = SemanticDocumentParser(max_chunk_tokens=512)
                        
                        # Extract text and chunk semantically
                        chunks = parser.parse_pdf(file_bytes)
                        all_text = " ".join([c.content for c in chunks])
                        
                        # Extract medical entities
                        entities = parser.extract_medical_entities(all_text)
                        
                        # Update patient data
                        if entities:
                            entity_map = {
                                'creatinine': 'sc', 'gfr': 'grf', 'hemoglobin': 'hemo',
                                'albumin': 'al', 'potassium': 'pot', 'sodium': 'sod', 'bp_diastolic': 'bp'
                            }
                            for key, val in entities.items():
                                if key in entity_map:
                                    st.session_state.patient_data[entity_map[key]] = val
                            st.success(f"📊 **Extracted from PDF:** {entities}")
                        
                        # Store for RAG
                        st.session_state.document_chunks = chunks
                        st.info(f"✅ Created {len(chunks)} semantic chunks")
                        
                        # Show preview
                        with st.expander("📋 PDF Text Preview", expanded=False):
                            st.text(all_text[:2000] + ("..." if len(all_text) > 2000 else ""))
                        
                        st.session_state.report_index = "Loaded"
                        
                    except Exception as e:
                        st.error(f"PDF parsing error: {e}")
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

    # =====================
    # TIERED WORKFLOW LOGIC
    # =====================
    if analyze_btn:
        st.toast("🔍 Starting Tiered Analysis...")
        
        # --- TIER 1: Robust Model-Only (XGBoost → Stacking Validation) ---
        with st.status("**Tier 1: Robust Model Inference**", expanded=True) as status:
            try:
                # Prepare Input
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
                    'ane': [pd_data['ane']], 'grf': [pd_data['grf']],
                    'chol': [pd_data.get('chol', 200)], 'tg': [pd_data.get('tg', 150)],
                    'ldl': [pd_data.get('ldl', 100)], 'hdl': [pd_data.get('hdl', 50)],
                    'ua': [pd_data.get('ua', 5.5)], 'ca': [pd_data.get('ca', 9.0)],
                    'phos': [pd_data.get('phos', 3.5)], 'hba1c': [pd_data.get('hba1c', 5.5)],
                    'statin': [pd_data.get('statin', 0)], 'metformin': [pd_data.get('metformin', 0)],
                    'insulin': [pd_data.get('insulin', 0)], 'nsaid': [pd_data.get('nsaid', 0)],
                    'serum_al': [pd_data.get('al_serum', 4.0)]
                }
                input_df = pd.DataFrame(input_data)
                processed_df = engineer_features_app(input_df, None)
                for col in processed_df.columns:
                    if col in encoders:
                        le_col = encoders[col]
                        val = str(processed_df[col].iloc[0])
                        if val in le_col.classes_:
                             processed_df[col] = le_col.transform([val])[0]
                        else:
                             processed_df[col] = 0
                    if processed_df[col].dtype == 'object':
                        processed_df[col] = 0
                train_cols = pd.read_csv('X_engineered.csv').columns
                for c in train_cols:
                    if c not in processed_df.columns:
                        processed_df[c] = 0
                processed_df = processed_df[train_cols]
                scaled_data = scaler.transform(processed_df)
                
                # Step 1a: Fast XGBoost
                st.write("⚡ Step 1/2: Running XGBoost (Speed Model)...")
                prob_fast = fast_model.predict_proba(scaled_data)[0][1]
                pred_fast = "CKD" if prob_fast > 0.5 else "Healthy"
                
                # Step 1b: Stacking Ensemble Validation
                st.write("🧠 Step 2/2: Running Stacking Ensemble (Validation)...")
                prob_stack = stacking_model.predict_proba(scaled_data)[0][1]
                pred_stack = "CKD" if prob_stack > 0.5 else "Healthy"
                
                # Consensus Logic
                if pred_fast == pred_stack:
                    consensus = pred_fast
                    confidence = max(prob_fast, prob_stack)
                    agreement = "✅ Models Agree"
                else:
                    consensus = pred_stack  # Trust stacking more
                    confidence = prob_stack
                    agreement = "⚠️ Conflicting - Using Ensemble"
                
                status.update(label="**Tier 1 Complete**", state="complete", expanded=False)
                
            except Exception as e:
                st.error(f"Tier 1 Error: {e}")
                consensus = "Error"
                confidence = 0
                agreement = ""

        # Display Tier 1 Result
        color = "red" if consensus == "CKD" else "green"
        st.markdown(f"""
        <div style="background: linear-gradient(145deg, rgba(20,30,40,0.95), rgba(30,50,60,0.95)); padding: 25px; border-radius: 15px; border-left: 5px solid {color}; margin: 20px 0;">
            <h2 style='color: {color}; margin:0;'>🔍 {consensus.upper()} ({confidence:.1%})</h2>
            <p style='margin:5px 0; color: #aaa;'>{agreement}</p>
            <small>XGBoost: {pred_fast} ({prob_fast:.1%}) | Stacking: {pred_stack} ({prob_stack:.1%})</small>
        </div>
        """, unsafe_allow_html=True)
        
        # Store for Tier 2
        st.session_state.tier1_result = {"consensus": consensus, "confidence": confidence, "fast": prob_fast, "stack": prob_stack}
        
        # --- TIER 2: Intelligent (LLM Council) ---
        if deep_research and consensus != "Error":
            with st.status("**Tier 2: LLM Council Deliberation**", expanded=True) as status2:
                try:
                    from council import MedicalCouncil
                    council = MedicalCouncil()
                    
                    patient_str = str(st.session_state.patient_data)
                    case_context = f"Model Prediction: {consensus} (Confidence: {confidence:.1%}). Fast: {prob_fast:.1%}, Stacking: {prob_stack:.1%}."
                    
                    st.write("🩺 Dr. Nemotron (Nephrology)...")
                    st.write("🔍 Dr. Mistral (Differential Dx)...")
                    st.write("💊 Dr. GLM (Pharmacology)...")
                    
                    opinions = council.consult(case_context, patient_str)
                    
                    # Use TABS to avoid overlapping columns
                    with st.expander("📋 View Council Deliberations", expanded=False):
                        tab1, tab2, tab3 = st.tabs(["🩺 Dr. Nephro", "🔬 Dr. Diag", "💊 Dr. Pharma"])
                        with tab1:
                            st.markdown("### Nephrology")
                            st.markdown(opinions['nephrologist'][:1000])
                        with tab2:
                            st.markdown("### Differential Dx")
                            st.markdown(opinions['diagnostician'][:1000])
                        with tab3:
                            st.markdown("### Pharmacology")
                            st.markdown(opinions['pharmacologist'][:1000])
                    
                    # Synthesize
                    final_prescription = council.synthesize(case_context, opinions)
                    status2.update(label="**Tier 2 Complete: Prescription Ready**", state="complete", expanded=False)
                    
                    st.markdown("---")
                    st.subheader("📋 Clinical Prescription")
                    st.markdown(final_prescription)
                    
                    # === AGENT-AS-A-JUDGE EVALUATION ===
                    with st.expander("📊 Quality Evaluation (Agent-as-a-Judge)", expanded=False):
                        try:
                            from judge_agent import JudgeAgent
                            judge = JudgeAgent()
                            
                            with st.spinner("Evaluating response quality..."):
                                evaluation = judge.score(
                                    query=case_context,
                                    response=final_prescription,
                                    context=str(st.session_state.patient_data)
                                )
                            
                            # Display scores
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Overall", f"{evaluation.overall_score}/10")
                            with col2:
                                st.metric("Safety", f"{evaluation.safety_score}/10")
                            with col3:
                                st.metric("Accuracy", f"{evaluation.accuracy_score}/10")
                            
                            if evaluation.suggestions:
                                st.caption(f"💡 Suggestions: {evaluation.suggestions[:200]}")
                        except Exception as e:
                            st.caption(f"Judge unavailable: {e}")
                    
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

                # C. Reasoning / Deep Research
                if deep_research:
                     # Council Deliberation (The "Subconscious Mind" of the Doctor)
                    try:
                        from council import MedicalCouncil
                        council = MedicalCouncil()
                        
                        # === BUILD CONVERSATION HISTORY CONTEXT ===
                        # Get last 5 messages for context
                        chat_history = ""
                        recent_msgs = st.session_state.messages[-10:]  # Last 5 turns
                        if len(recent_msgs) > 2:
                            chat_history = "\n--- Recent Conversation ---\n"
                            for msg in recent_msgs[:-1]:  # Exclude current message
                                role = "Patient" if msg["role"] == "user" else "Doctor"
                                chat_history += f"{role}: {msg['content'][:200]}\n"
                            chat_history += "---\n\n"
                        
                        # Convert session data to string for the agents
                        patient_str = str(st.session_state.patient_data)
                        
                        # Include history in prompt
                        prompt_with_context = f"{chat_history}Current Question: {prompt}"
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
                            response = cortex.diagnose(
                                prompt_with_context,  # Include conversation history
                                st.session_state.patient_data,
                                ml_prediction=ml_pred,
                                ml_probability=ml_prob
                            )
                            
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
                        response = engine.chat_reasoning(prompt, [], None)
                        st.markdown(clean_llm_output(response))
                else:
                    # FAST / SIMPLE MODE (Single LLM or purely heuristic)
                    engine = RAGEngine()
                    response = engine.chat_reasoning(prompt, [], None)
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
