# KidneyPred AI: Visual Walkthrough & System Guide 🧪🩺

This document provides a detailed walkthrough of the KidneyPred AI diagnostic system, its interactive features, and its underlying predictive engine.

## 🖥️ The Interactive Dashboard
The primary interface for KidneyPred AI is a Streamlit-powered dashboard that allows for real-time patient diagnosis and root cause analysis.

![KidneyPred AI Dashboard](dashboard_preview.png)

### Key Frontend Components:
- **Patient Biometrics Sidebar**: Input parameters for 29 clinical features, including age, blood pressure, and specific gravity.
- **Diagnostic Engine**: A button to trigger the Stacking Ensemble and generate a full report.
- **Confidence Metrics**: Displays the model's certainty level for each diagnosis.

---

## 🧬 Advanced Root Cause Analysis (XAI)
We provide two distinct "lenses" into the model's decision-making process to ensure clinical trust.

### 1. SHAP Explanation (Game Theory)
Uses Shapley values to show how each biometric factor pushed the prediction towards or away from a CKD diagnosis.

### 2. LIME Analysis (Local Surrogates)
Provides a local, linear approximation of the model's behavior for a specific patient, cross-referencing SHAP results for higher confidence.

---

## 🛡️ Reliability & Generalization
The system is built to be robust against noise and input variations.

### Sensitivity Analysis
We systematically perturb key features like `Serum Creatinine` to ensure the model responds logically and predictably.
![Sensitivity Analysis](sensitivity_analysis_sc.png)

### Learning Curves
Our model shows high generalization capability, converging to 100% accuracy with minimal variance across various training set sizes.
![Learning Curves](learning_curves.png)

---

## 🛠️ Developer Checklist
- **Unit Tests**: Pass with 100% via `pytest test_pipeline.py`.
- **Packaging**: Fully containerized using the provided [Dockerfile](Dockerfile).
- **Automation**: `evaluate_accuracy.py` and `overfitting_check.py` are provided for rapid re-validation.

---
*Created by Avirup Roy - Powered by Advanced Agentic Coding.*
