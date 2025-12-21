# KidneyPred AI: Advanced Diagnostic System

KidneyPred AI is a high-end machine learning solution designed to predict Chronic Kidney Disease (CKD) using clinical biometric data. It features a robust multi-model ensemble and state-of-the-art Explainable AI (XAI) for medical transparency.

## 🚀 Features
- **Stacked Ensemble Model**: Combines XGBoost, Random Forest, SVM, Logistic Regression, and Naive Bayes for maximum prediction robustness (100% CV Accuracy).
- **Multi-XAI Dashboard**: Interactive interpretation using both **SHAP** and **LIME** to explain individual patient risk factors.
- **Root Cause Analysis (RCA)**: Identifies the primary clinical drivers behind a diagnosis (e.g., Creatinine levels, Hemoglobin, GFR).
- **Reliability Tested**: Verified through sensitivity analysis to ensure stability under input perturbations.

## 🛠️ Tech Stack
- **Core**: Python, Scikit-learn, XGBoost.
- **XAI**: SHAP, LIME, streamlit-shap.
- **UI**: Streamlit (Premium Dark Theme).
- **Diagnostics**: Statsmodels, Seaborn.

## 🏃 Getting Started

### Local Setup
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the interactive dashboard:
   ```bash
   streamlit run app.py
   ```

### Docker Deployment
```bash
docker build -t kidneypred-ai .
docker run -p 8501:8501 kidneypred-ai
```

## 🧪 Testing and Validation
- **Unit Tests**: `pytest test_pipeline.py` ensures data and encoding integrity.
- **Sensitivity Analysis**: `python reliability_test.py` validates model stability.
- **Statistical Diagnostics**: `diagnostics_results.txt` contains VIF and heteroskedasticity tests.

## 🧬 Data Source
Fetched from the **UCI Machine Learning Repository** (ID: 857 - "Risk Factor Prediction of Chronic Kidney Disease").

---
Disclaimer: This tool is for research purposes and should not be used as a substitute for professional medical advice.
