# Cortex Coordinator - Brain-Inspired Cognitive Architecture
# Makes agents work together like an experienced doctor's mind

import os
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

@dataclass
class ClinicalEvidence:
    """Evidence from different cognitive sources."""
    source: str  # ml_model, council, data, safety
    finding: str
    confidence: float
    reasoning: str

@dataclass
class ClinicalJudgment:
    """Final unified clinical judgment."""
    diagnosis: str
    ckd_stage: str
    risk_level: str  # low, moderate, high, critical
    ml_probability: float
    council_agrees: bool
    prescription: List[str]
    lifestyle: List[str]
    follow_up: str
    warnings: List[str]
    evidence_summary: str


class CortexCoordinator:
    """
    Brain-inspired coordinator that makes agents work like an experienced doctor.
    
    Cognitive Layers:
    - REFLEXIVE: Fast pattern matching (ML model, safety flags)
    - SUBCONSCIOUS: Deep reasoning (LLM Council deliberation)  
    - CONSCIOUS: Integration and final judgment (this class)
    
    Like an experienced doctor who:
    1. Quickly recognizes patterns (ML)
    2. Deliberates on complex cases (Council)
    3. Retrieves relevant knowledge (RAG)
    4. Checks for dangers (Safety)
    5. Synthesizes everything into a unified diagnosis
    """
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model="nvidia/nemotron-3-nano-30b-a3b:free",
            temperature=0.2,
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY")
        )
        
        self.synthesis_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a senior nephrologist synthesizing findings from multiple sources.

Act like an experienced doctor who:
- Has seen 58,000+ kidney disease cases
- Weighs pattern recognition (ML) with clinical reasoning (Council)
- Never ignores safety warnings
- Provides clear, actionable guidance

You must synthesize all evidence into a UNIFIED clinical assessment.

FORMAT YOUR RESPONSE EXACTLY AS:
## 🩺 Clinical Assessment

**Diagnosis**: [Stage X CKD / No CKD / At Risk]
**Risk Level**: [Low/Moderate/High/Critical]
**Confidence**: [X%]

### 📊 Evidence Synthesis
[2-3 sentences synthesizing ML + Council + Data findings]

### 💊 Prescription
1. [Medication 1 with dosage]
2. [Medication 2 with dosage]

### 🥗 Lifestyle Modifications
1. [Specific dietary change]
2. [Activity recommendation]

### ⚠️ Warnings
- [Critical warnings if any]

### 📅 Follow-up
[When to return, what tests to do]"""),
            ("human", """PATIENT QUERY: {query}

PATIENT DATA:
{patient_data}

EVIDENCE FROM COGNITIVE LAYERS:

1. ML MODEL (Pattern Recognition):
{ml_evidence}

2. COUNCIL (Clinical Reasoning):
{council_evidence}

3. DATA RETRIEVAL (Medical Knowledge):
{data_evidence}

4. SAFETY CHECK:
{safety_evidence}

Now synthesize all evidence and provide your unified clinical judgment:""")
        ])
    
    def gather_evidence(self, 
                        query: str, 
                        patient_data: Dict,
                        ml_prediction: int = None,
                        ml_probability: float = None) -> Dict[str, ClinicalEvidence]:
        """
        Gather evidence from all cognitive layers in PARALLEL.
        """
        evidence = {}
        
        # 1. ML Model Evidence (Reflexive - instant)
        if ml_prediction is not None:
            pred_label = "CKD Positive" if ml_prediction == 1 else "CKD Negative"
            evidence['ml_model'] = ClinicalEvidence(
                source="ML Model (58k cases)",
                finding=f"Prediction: {pred_label}",
                confidence=ml_probability or 0.5,
                reasoning=f"Based on pattern matching from 58,000 historical cases"
            )
        else:
            evidence['ml_model'] = ClinicalEvidence(
                source="ML Model",
                finding="Insufficient data for prediction",
                confidence=0.0,
                reasoning="Patient data not complete enough for ML prediction"
            )
        
        # 3. Data Evidence (instant - no LLM call)
        data_summary = self._summarize_patient_data(patient_data)
        evidence['data'] = ClinicalEvidence(
            source="Patient Data",
            finding=data_summary,
            confidence=1.0,
            reasoning="Direct lab values and patient history"
        )
        
        # 2 & 4. Council + Safety in PARALLEL (these are slow)
        def get_council_evidence():
            try:
                from council import MedicalCouncil
                council = MedicalCouncil()
                opinions = council.consult(query, str(patient_data))
                synthesis = council.synthesize(query, opinions)
                return ClinicalEvidence(
                    source="LLM Council (3 specialists)",
                    finding=synthesis[:500],
                    confidence=0.85,
                    reasoning="Deliberation between nephrology, diagnosis, and pharmacology experts"
                )
            except Exception as e:
                return ClinicalEvidence(
                    source="Council",
                    finding=f"Council unavailable: {e}",
                    confidence=0.0,
                    reasoning="Error in council deliberation"
                )
        
        def get_safety_evidence():
            try:
                from safety_agent import SafetyGuardrailAgent
                safety = SafetyGuardrailAgent()
                check = safety.validate(query, patient_data)
                return ClinicalEvidence(
                    source="Safety System",
                    finding="PASSED" if check.passed else f"ISSUES: {', '.join(check.issues)}",
                    confidence=1.0 if check.passed else 0.7,
                    reasoning=f"Severity: {check.severity}"
                )
            except Exception as e:
                return ClinicalEvidence(
                    source="Safety",
                    finding="Safety check unavailable",
                    confidence=0.5,
                    reasoning=str(e)
                )
        
        # Run Council and Safety in parallel threads
        with ThreadPoolExecutor(max_workers=2) as executor:
            council_future = executor.submit(get_council_evidence)
            safety_future = executor.submit(get_safety_evidence)
            
            evidence['council'] = council_future.result()
            evidence['safety'] = safety_future.result()
        
        return evidence
    
    def _summarize_patient_data(self, patient_data: Dict) -> str:
        """Create clinical summary of patient data."""
        sc = patient_data.get('sc', 0)
        gfr = patient_data.get('grf', 0)
        htn = patient_data.get('htn', 'no')
        dm = patient_data.get('dm', 'no')
        age = patient_data.get('age', 0)
        hemo = patient_data.get('hemo', 0)
        
        summary = []
        if sc > 0:
            summary.append(f"Creatinine: {sc} mg/dL")
        if gfr > 0:
            summary.append(f"eGFR: {gfr} mL/min")
        if htn == 'yes':
            summary.append("Hypertension: Present")
        if dm == 'yes':
            summary.append("Diabetes: Present")
        if age > 0:
            summary.append(f"Age: {age}")
        if hemo > 0:
            summary.append(f"Hemoglobin: {hemo}")
        
        return " | ".join(summary) if summary else "Limited data available"
    
    def synthesize_judgment(self, 
                            query: str, 
                            patient_data: Dict,
                            evidence: Dict[str, ClinicalEvidence]) -> str:
        """
        Synthesize all evidence into unified clinical judgment.
        This is the "conscious" layer - final integration.
        """
        try:
            response = self.llm.invoke(
                self.synthesis_prompt.format_messages(
                    query=query,
                    patient_data=str(patient_data),
                    ml_evidence=f"{evidence['ml_model'].finding} (Confidence: {evidence['ml_model'].confidence:.0%})",
                    council_evidence=evidence['council'].finding,
                    data_evidence=evidence['data'].finding,
                    safety_evidence=evidence['safety'].finding
                )
            )
            return response.content
        except Exception as e:
            return f"Unable to synthesize judgment: {e}"
    
    def diagnose(self, 
                 query: str, 
                 patient_data: Dict,
                 ml_prediction: int = None,
                 ml_probability: float = None,
                 enable_critique: bool = True) -> str:
        """
        Full cognitive pipeline: gather evidence → synthesize → critique → improve → deliver.
        
        This mimics how an experienced doctor thinks:
        1. Pattern recognition (immediate ML assessment)
        2. Deep analysis (council deliberation)
        3. Knowledge retrieval (past cases, guidelines)
        4. Safety check (contraindications, warnings)
        5. Integration (unified clinical judgment)
        6. CRITIQUE & IMPROVE (quality check before delivery)
        """
        # Gather evidence from all cognitive layers
        evidence = self.gather_evidence(query, patient_data, ml_prediction, ml_probability)
        
        # Synthesize into unified judgment
        judgment = self.synthesize_judgment(query, patient_data, evidence)
        
        # === JUDGE FEEDBACK LOOP (Anterior Cingulate - Error Monitoring) ===
        if enable_critique:
            try:
                from judge_agent import JudgeAgent
                judge = JudgeAgent()
                
                # Score the initial response
                score = judge.score_response(judgment, query, str(patient_data))
                
                # If score is below threshold, critique and improve
                if score.overall < 7.0:
                    critique, improved = judge.critique_and_improve(judgment, query)
                    
                    # Return improved version with note
                    judgment = improved + f"\n\n---\n*Quality Score: {score.overall}/10 → Improved*"
                else:
                    # Add quality badge
                    judgment = judgment + f"\n\n---\n✅ *Quality Score: {score.overall}/10*"
                    
            except Exception as e:
                # If judge fails, return original judgment
                pass
        
        return judgment


# Singleton
_cortex = None

def get_cortex() -> CortexCoordinator:
    global _cortex
    if _cortex is None:
        _cortex = CortexCoordinator()
    return _cortex


# Test
if __name__ == "__main__":
    cortex = CortexCoordinator()
    
    test_data = {
        'age': 65, 'sc': 2.4, 'grf': 35, 
        'htn': 'yes', 'dm': 'yes', 'hemo': 11.5
    }
    
    judgment = cortex.diagnose(
        "Am I at risk for kidney disease?",
        test_data,
        ml_prediction=1,
        ml_probability=0.78
    )
    
    print(judgment)
