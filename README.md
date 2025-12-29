# KidneyPred AI: Advanced Diagnostic System 🩺

KidneyPred AI is a state-of-the-art machine learning solution designed to predict Chronic Kidney Disease (CKD) using clinical biometric data. This project delivers a production-grade diagnostic pipeline, combining high-accuracy ensemble modeling with multi-perspective Explainable AI (XAI) to ensure clinical transparency and trust.

![KidneyPred AI Dashboard](dashboard_ui_v2.png)

## 📚 Documentation
- **[System Architecture](ARCHITECTURE.md)**: Comprehensive technical documentation with architecture diagrams, process flows, and component details
- **[Visual Walkthrough](WALKTHROUGH.md)**: Interactive dashboard guide and XAI features

## 🌟 Project Overview
The system leverages a massive dataset of **58,000+ patients** (D4 ESRD + UCI), incorporating **40+ clinical features**. It is designed to assist medical researchers and practitioners in identifying CKD risk factors and understanding the underlying drivers of specific predictions.

### Key Highlights:
- **~87% Accuracy**: Achieved through robust Stacking Ensemble with proper train/test validation on 58k records.
- **Deep Intelligence**: Now integrates **Advanced Metabolic Markers** (Lipids, Minerals, HbA1c) and **Medication History**.
- **Gemini 2.0 Parsing**: Automated extraction of patient metadata and lab values from PDF reports using Google's SOTA multimodal model.
- **Explainable AI (XAI)**: Integrated **SHAP** and **LIME** for Root Cause Analysis (RCA).
- **Medical Feature Engineering**: Engineered synergistic markers like `Electrolyte Imbalance` and `Anemia Risk Index`.
- **Production Ready**: Fully Dockerized and verified with unit tests and sensitivity analysis.

## 🧠 Model Architecture
The core engine is a **Stacking Classifier** that intelligently combines the strengths of multiple diverse algorithms:
1.  **XGBoost**: Captures complex non-linear interactions.
2.  **Random Forest**: Provides robust tree-based classification.
3.  **SVM (Linear)**: Ensures strong margin-based separation.
4.  **Logistic Regression**: Serves as a reliable probabilistic baseline and the Meta-Learner.
5.  **Gaussian Naive Bayes**: Provides a Bayesian baseline for probabilistic reasoning.

## 🏗️ System Architecture
The system uses a **brain-inspired multi-agent architecture** that mimics human clinical reasoning:

### Core Components:
- **🧠 Cortex Coordinator**: Brain-inspired orchestrator with 4 cognitive layers (Reflexive, Analytical, Collaborative, Conscious)
- **🤖 Medical Council**: 3 specialist AI doctors (Nephrologist, Diagnostician, Pharmacologist) deliberating in parallel
- **🔍 RAG Engine**: Medical knowledge retrieval from clinical reasoning datasets
- **💾 SQL Agent**: Natural language to SQL for patient data analytics
- **🎯 Query Planner**: Intelligent routing (simple/sql/rag/council/hybrid)
- **📄 Document Analyzer**: Powered by **Gemini Flash 2.0**, extracts structured data (Name, Vitals, Metadata) from raw PDF reports with high precision.

**→ See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed diagrams and component documentation**

## 🧪 Advanced Feature Engineering
We created several domain-specific features to capture physiological interactions:
- **sod_pot_ratio**: Identifying electrolyte imbalances.
- **anemia_index**: Combining Hemoglobin and RBC count for a more accurate anemia marker.
- **creatinine_gfr_interaction**: Capturing the efficiency of kidney filtration.
- **metabolic_risk**: Clustering hypertension and diabetes as primary risk factors.

## 🧬 Explainable AI (XAI) Lab
The dashboard provides two primary lenses for interpretability:
- **🎯 SHAP (Game Theory Based)**: Fairly distributes "credit" among features to show the exact magnitude and direction of their influence.
- **🧪 LIME (Local Surrogate Based)**: Approximates the model locally for a specific patient to provide an intuitive, human-understandable explanation.

![Sensitivity Analysis](sensitivity_analysis_sc.png)
*Above: Sensitivity analysis showing the model's stable response to perturbations in Serum Creatinine.*

## 🛠️ Installation & Usage

### Local Setup
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the dashboard:
   ```bash
   streamlit run app.py
   ```

### Docker Deployment
```bash
docker build -t kidneypred-ai .
docker run -p 8501:8501 kidneypred-ai
```

## 🧪 Testing & Validation
Run unit tests to verify data pipelines:
```bash
pytest test_pipeline.py
```
Execute reliability and overfitting tests:
```bash
python reliability_test.py
python overfitting_check.py
```

### 🛡️ Overfitting & Generalization Report
We formally verified the model's robustness using proper validation methodology:
- **Learning Curves**: Training and Validation scores converge, indicating the model generalizes well without overfitting.
- **Proper Validation**: SMOTE applied only to training data to prevent test set contamination.
- **Honest Metrics**: ~87% accuracy on held-out test data (not affected by data augmentation).

> ⚠️ **Note**: Earlier versions reported higher accuracy (~95-100%) due to SMOTE being applied before train/test split. Current metrics reflect true generalization performance.

![Learning Curves](learning_curves.png)

## 📜 Disclaimer
This tool is for research purposes only. It is not intended for clinical use and should not substitute for professional medical judgment.
