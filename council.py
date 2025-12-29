import os
import re
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

def clean_html(text: str) -> str:
    """Remove HTML artifacts from LLM output."""
    text = text.replace('<br>', '\n')
    text = text.replace('<br/>', '\n')
    text = text.replace('<br />', '\n')
    text = text.replace('&nbsp;', ' ')
    text = text.replace('&amp;', '&')
    text = text.replace('&lt;', '<')
    text = text.replace('&gt;', '>')
    # Remove any remaining HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    return text

class MedicalCouncil:
    def __init__(self):
        # 1. Dr. Nemotron (Clinical Nephrologist)
        # Strength: Nvidia's model, strong on medical reasoning
        self.agent_nephro = ChatOpenAI(
            model="nvidia/nemotron-3-nano-30b-a3b:free",
            temperature=0.3,
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY")
        )

        # 2. Dr. Mistral (Differential Diagnostician)
        # Using same reliable model for consistency
        self.agent_diag = ChatOpenAI(
            model="nvidia/nemotron-3-nano-30b-a3b:free",
            temperature=0.5,
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY")
        )

        # 3. Dr. GLM (Pharmacologist / Safety)
        # Using same reliable model for consistency
        self.agent_pharma = ChatOpenAI(
            model="nvidia/nemotron-3-nano-30b-a3b:free",
            temperature=0.2, # Low temp for safety/precision
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY")
        )

    def consult(self, case_description, patient_data_str):
        """
        Runs the Council Deliberation.
        Returns a dict with independent opinions and a synthesized report.
        """
        
        # --- PROMPTS ---
        prompt_nephro = f"""
        You are Dr. Nemotron, a Senior Clinical Nephrologist.
        Analyze this patient's case focusing STRICTLY on RENAL PHYSIOLOGY and CKD PROGRESSION.
        
        Patient Data: {patient_data_str}
        Case Notes: {case_description}
        
        Output:
        1. Estimated renal function status.
        2. Specific physiological risks (e.g. electrolyte imbalance).
        3. Recommended next diagnostic steps.
        """

        prompt_diag = f"""
        You are Dr. Mistral, a Differential Diagnostician. Your job is to play DEVIL'S ADVOCATE.
        Do NOT assume it is just CKD. Look for other causes (Cardiac, Autoimmune, Acute Injury).
        
        Patient Data: {patient_data_str}
        Case Notes: {case_description}
        
        Output:
        1. List 3 alternative diagnoses associated with these symptoms.
        2. Flag any inconsistencies in the data.
        """

        prompt_pharma = f"""
        You are Dr. GLM, a Clinical Pharmacologist.
        Focus on MEDICATION SAFETY, SAFETY WARNINGS, and LIFESTYLE.
        
        Patient Data: {patient_data_str}
        Case Notes: {case_description}
        
        Output:
        1. Warnings about potential nephrotoxic drugs (NSAIDs, etc).
        2. Determine if Metformin/Statin use is safe given the presumed kidney function.
        3. Simple lifestyle edits.
        """

        # --- PARALLEL EXECUTION with ThreadPoolExecutor ---
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        def call_nephro():
            return clean_html(self.agent_nephro.invoke([HumanMessage(content=prompt_nephro)]).content)
        
        def call_diag():
            return clean_html(self.agent_diag.invoke([HumanMessage(content=prompt_diag)]).content)
        
        def call_pharma():
            return clean_html(self.agent_pharma.invoke([HumanMessage(content=prompt_pharma)]).content)
        
        # Execute all 3 calls in parallel
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_nephro = executor.submit(call_nephro)
            future_diag = executor.submit(call_diag)
            future_pharma = executor.submit(call_pharma)
            
            response_nephro = future_nephro.result()
            response_diag = future_diag.result()
            response_pharma = future_pharma.result()

        return {
            "nephrologist": response_nephro,
            "diagnostician": response_diag,
            "pharmacologist": response_pharma
        }

    def synthesize(self, case_description, council_outputs):
        """
        Synthesizes the 3 opinions into a final consensus.
        """
        synth_prompt = f"""
        You are the Chief Medical Resident. Synthesize the following 3 specialist opinions into a single, cohesive patient report.
        
        Case: {case_description}
        
        opinion_nephro: {council_outputs['nephrologist']}
        opinion_diag: {council_outputs['diagnostician']}
        opinion_pharma: {council_outputs['pharmacologist']}
        
        IMPORTANT FORMATTING RULES:
        - Use proper Markdown formatting (headers with ##, bullets with -, numbered lists with 1.)
        - Use line breaks between sections
        - Use bullet points (-) for lists, NOT <br> tags
        - Use tables with | for comparisons
        
        Format the response EXACTLY like this:
        
        ## 🏥 Consensus Medical Report
        
        **Summary**: [One sentence summary]
        
        ---
        
        ### 🩺 Diagnostic Assessment
        
        **Primary Diagnosis**: [Main diagnosis]
        
        **Differential Considerations**:
        - [Alternative 1]
        - [Alternative 2]
        - [Alternative 3]
        
        ---
        
        ### 💊 Treatment & Safety Plan
        
        **Recommended Medications**:
        | Medication | Dosage | Notes |
        |------------|--------|-------|
        | [Med 1] | [Dose] | [Note] |
        
        **Avoid/Caution**:
        - [Drug/substance to avoid]
        
        **Lifestyle Modifications**:
        - [Lifestyle change 1]
        - [Lifestyle change 2]
        
        ---
        
        ### ⚠️ Critical Alerts
        
        > ⚠️ [Any urgent safety warning]
        
        ---
        
        ### 📋 Next Steps
        
        1. [Action item 1]
        2. [Action item 2]
        3. [Action item 3]
        """
        
        # We use the Nephrologist model (Nemotron) as the synthesizer as it's the domain expert
        final_report = self.agent_nephro.invoke([HumanMessage(content=synth_prompt)]).content
        
        # Clean up any HTML artifacts
        final_report = clean_html(final_report)
        
        # === SAFETY GUARDRAIL VALIDATION ===
        try:
            from safety_agent import SafetyGuardrailAgent
            safety = SafetyGuardrailAgent()
            
            # Run safety checks (advisor review disabled for speed, enable for critical cases)
            check = safety.validate(final_report, run_advisor=False)
            
            # Use the safety-modified output (includes disclaimers, warnings)
            final_report = check.modified_output
            
            # Log safety issues if any
            if check.issues:
                print(f"[SAFETY] Severity: {check.severity}, Issues: {check.issues}")
        except Exception as e:
            # If safety check fails, continue but add generic disclaimer
            final_report += "\n\n---\n⚠️ **Disclaimer**: This is AI-assisted analysis. Consult a healthcare provider."
        
        return final_report
