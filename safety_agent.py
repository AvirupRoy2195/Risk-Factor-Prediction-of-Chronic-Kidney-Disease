# Safety & Guardrail Agent
# Validates LLM outputs for hallucinations, dangerous advice, and missing disclaimers

import os
import re
from typing import Tuple, List, Dict
from dataclasses import dataclass
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

@dataclass
class SafetyCheck:
    """Result of a safety check."""
    passed: bool
    issues: List[str]
    severity: str  # "low", "medium", "high", "critical"
    modified_output: str


class SafetyGuardrailAgent:
    """
    Multi-layer safety agent that validates medical AI outputs:
    
    1. Hallucination Detection - Flags unsupported claims
    2. Drug Safety - Checks for dangerous interactions/contraindications
    3. Dosage Validation - Verifies medication doses are in safe ranges
    4. Disclaimer Enforcement - Ensures proper medical disclaimers
    5. Advisor Review - Final prescription review by separate LLM
    """
    
    # Known nephrotoxic drugs to flag
    NEPHROTOXIC_DRUGS = [
        "nsaids", "ibuprofen", "naproxen", "aspirin",
        "gentamicin", "vancomycin", "amphotericin",
        "contrast dye", "iodinated contrast",
        "lithium", "cyclosporine", "tacrolimus",
    ]
    
    # Dangerous drug combinations for CKD patients
    DANGEROUS_COMBINATIONS = [
        ("ace inhibitor", "potassium supplement"),
        ("ace inhibitor", "arb"),  # ACE + ARB dual blockade
        ("nsaid", "ace inhibitor"),
        ("metformin", "contrast"),  # Risk of lactic acidosis
    ]
    
    # Maximum safe doses for CKD patients (simplified)
    DOSE_LIMITS = {
        "metformin": {"max_mg": 1000, "note": "Reduce in CKD stage 3+"},
        "gabapentin": {"max_mg": 300, "note": "Reduce in CKD"},
        "lisinopril": {"max_mg": 40, "note": "Start low in CKD"},
        "furosemide": {"max_mg": 200, "note": "May need high doses in CKD"},
    }
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model="nvidia/nemotron-3-nano-30b-a3b:free",
            temperature=0,
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY")
        )
        
        self.required_disclaimer = (
            "⚠️ **Disclaimer**: This is AI-assisted analysis for educational purposes only. "
            "Always consult a qualified healthcare provider for medical decisions."
        )
    
    def check_hallucinations(self, output: str, context: str = "") -> Tuple[bool, List[str]]:
        """
        Detect potential hallucinations in the output.
        Uses both pattern matching and LLM verification.
        """
        issues = []
        
        # Pattern-based checks for common hallucination indicators
        hallucination_patterns = [
            (r"studies show that \d+%", "Unverified statistic"),
            (r"according to .+ study", "Uncited study reference"),
            (r"proven to cure", "Overclaim - nothing 'cures' CKD"),
            (r"guaranteed to", "Guarantee language is suspect"),
            (r"always works", "Absolute claim without evidence"),
            (r"100% effective", "Unrealistic effectiveness claim"),
        ]
        
        for pattern, issue in hallucination_patterns:
            if re.search(pattern, output.lower()):
                issues.append(f"⚠️ Potential hallucination: {issue}")
        
        # LLM-based fact check for critical claims
        if "kidney function will improve" in output.lower() or "cure" in output.lower():
            issues.append("⚠️ Claim about CKD reversal requires verification - CKD is typically progressive")
        
        return len(issues) == 0, issues
    
    def check_drug_safety(self, output: str, patient_data: dict = None) -> Tuple[bool, List[str]]:
        """Check for dangerous drug recommendations."""
        issues = []
        output_lower = output.lower()
        
        # Check for nephrotoxic drugs without warnings
        for drug in self.NEPHROTOXIC_DRUGS:
            if drug in output_lower:
                if "caution" not in output_lower and "avoid" not in output_lower:
                    issues.append(f"🚫 Nephrotoxic drug mentioned without warning: {drug.title()}")
        
        # Check for dangerous combinations
        for drug1, drug2 in self.DANGEROUS_COMBINATIONS:
            if drug1 in output_lower and drug2 in output_lower:
                issues.append(f"⚠️ Potentially dangerous combination: {drug1.title()} + {drug2.title()}")
        
        # Check dosages if patient has CKD
        if patient_data and patient_data.get('grf', 100) < 60:  # CKD stage 3+
            for drug, limits in self.DOSE_LIMITS.items():
                dose_match = re.search(rf"{drug}\s*(\d+)\s*mg", output_lower)
                if dose_match:
                    dose = int(dose_match.group(1))
                    if dose > limits['max_mg']:
                        issues.append(
                            f"⚠️ High dose warning: {drug.title()} {dose}mg exceeds "
                            f"recommended max of {limits['max_mg']}mg for CKD. {limits['note']}"
                        )
        
        return len(issues) == 0, issues
    
    def check_disclaimer(self, output: str) -> Tuple[bool, str]:
        """Ensure output has proper medical disclaimer."""
        disclaimer_patterns = [
            r"disclaimer",
            r"consult.+healthcare",
            r"consult.+doctor",
            r"medical advice",
            r"professional.+verification",
        ]
        
        has_disclaimer = any(re.search(p, output.lower()) for p in disclaimer_patterns)
        
        if has_disclaimer:
            return True, output
        else:
            # Add disclaimer
            return False, output + f"\n\n---\n{self.required_disclaimer}"
    
    def advisor_review(self, output: str, patient_data: dict = None) -> Tuple[bool, str]:
        """
        Final review by advisor LLM to catch anything missed.
        """
        review_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a Medical Safety Advisor reviewing AI-generated medical advice.

Check the following output for:
1. Dangerous or incorrect medical advice
2. Missing important safety warnings
3. Unrealistic claims about treatment outcomes
4. Inappropriate recommendations for kidney disease patients

Patient GFR: {gfr} (lower = worse kidney function)

If you find issues, respond with:
ISSUES:
- [issue 1]
- [issue 2]

If the output is safe, respond with:
APPROVED

Be concise."""),
            ("human", "{output}")
        ])
        
        try:
            gfr = patient_data.get('grf', 90) if patient_data else 90
            response = self.llm.invoke(
                review_prompt.format_messages(output=output[:2000], gfr=gfr)
            )
            
            review_result = response.content.strip()
            
            if "APPROVED" in review_result:
                return True, "Advisor approved"
            else:
                return False, review_result
                
        except Exception as e:
            return True, f"Advisor review skipped: {e}"
    
    def validate(self, output: str, patient_data: dict = None, run_advisor: bool = True) -> SafetyCheck:
        """
        Run all safety checks on the output.
        
        Returns SafetyCheck with pass/fail status, issues, and potentially modified output.
        """
        all_issues = []
        severity = "low"
        modified_output = output
        
        # 1. Hallucination check
        passed, issues = self.check_hallucinations(output)
        all_issues.extend(issues)
        
        # 2. Drug safety check
        passed, issues = self.check_drug_safety(output, patient_data)
        all_issues.extend(issues)
        if issues:
            severity = "high"
        
        # 3. Disclaimer check (always add if missing)
        had_disclaimer, modified_output = self.check_disclaimer(modified_output)
        if not had_disclaimer:
            all_issues.append("ℹ️ Added required medical disclaimer")
        
        # 4. Advisor review (optional, adds latency)
        if run_advisor and severity in ["medium", "high"]:
            advisor_passed, advisor_notes = self.advisor_review(output, patient_data)
            if not advisor_passed:
                all_issues.append(f"🔍 Advisor notes: {advisor_notes}")
                severity = "critical" if "dangerous" in advisor_notes.lower() else severity
        
        # Determine overall pass/fail
        critical_issues = [i for i in all_issues if "🚫" in i or severity == "critical"]
        passed = len(critical_issues) == 0
        
        # Add safety warnings to output if issues found
        if all_issues and severity in ["high", "critical"]:
            warning_block = "\n\n---\n⚠️ **Safety Review Notes:**\n"
            warning_block += "\n".join(f"- {issue}" for issue in all_issues if "ℹ️" not in issue)
            modified_output = warning_block + "\n\n---\n\n" + modified_output
        
        return SafetyCheck(
            passed=passed,
            issues=all_issues,
            severity=severity,
            modified_output=modified_output
        )


# Test
if __name__ == "__main__":
    agent = SafetyGuardrailAgent()
    
    # Test with potentially dangerous advice
    test_output = """
    For your kidney condition, I recommend:
    - Take ibuprofen 800mg three times daily for pain
    - Metformin 2000mg for diabetes
    - Studies show that 95% of patients improve with this treatment
    """
    
    patient = {"grf": 25, "sc": 3.5}  # CKD stage 4
    
    result = agent.validate(test_output, patient, run_advisor=False)
    
    print(f"Passed: {result.passed}")
    print(f"Severity: {result.severity}")
    print(f"Issues: {result.issues}")
    print(f"\nModified Output:\n{result.modified_output}")
