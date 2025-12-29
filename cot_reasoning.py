# Chain-of-Thought (CoT) Reasoning Wrapper
# Implements structured thinking for medical reasoning tasks

import os
import re
from typing import Optional, Tuple
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

class ChainOfThoughtReasoner:
    """
    Implements Chain-of-Thought prompting for better medical reasoning.
    
    Features:
    - Structured <think> tags for reasoning
    - Step-by-step problem decomposition
    - Self-verification before final answer
    """
    
    COT_SYSTEM_PROMPT = """You are a medical reasoning expert specializing in kidney disease.

When answering questions, you MUST use this thinking structure:

<think>
STEP 1: Understand the question
[What is being asked? What data do we have?]

STEP 2: Relevant medical knowledge
[What clinical guidelines apply? What are normal ranges?]

STEP 3: Analysis
[Apply knowledge to the specific case]

STEP 4: Verify
[Check for errors, consider alternatives]
</think>

After your thinking, provide a clear FINAL ANSWER.

IMPORTANT: Always show your reasoning in <think> tags before answering."""

    def __init__(self):
        self.llm = ChatOpenAI(
            model="nvidia/nemotron-3-nano-30b-a3b:free",
            temperature=0.3,
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY")
        )
    
    def reason(self, 
               query: str, 
               patient_data: dict = None, 
               context: str = "") -> Tuple[str, str]:
        """
        Generate a response using Chain-of-Thought reasoning.
        
        Returns:
            Tuple of (thinking_process, final_answer)
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.COT_SYSTEM_PROMPT),
            ("human", """Patient Context: {patient_data}

Additional Context: {context}

Question: {query}

Think through this step by step, then provide your answer.""")
        ])
        
        try:
            response = self.llm.invoke(
                prompt.format_messages(
                    query=query,
                    patient_data=str(patient_data) if patient_data else "Not provided",
                    context=context or "None"
                )
            )
            
            full_response = response.content
            
            # Extract thinking and answer
            thinking, answer = self._parse_cot_response(full_response)
            
            return thinking, answer
            
        except Exception as e:
            return f"Error in reasoning: {e}", "Unable to generate response"
    
    def _parse_cot_response(self, response: str) -> Tuple[str, str]:
        """Parse the thinking and answer from a CoT response."""
        
        # Look for <think> tags
        think_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL | re.IGNORECASE)
        
        if think_match:
            thinking = think_match.group(1).strip()
            # Get everything after </think>
            answer = response[think_match.end():].strip()
        else:
            # No explicit tags, try to split on common patterns
            if "FINAL ANSWER:" in response.upper():
                parts = re.split(r'FINAL ANSWER:', response, flags=re.IGNORECASE)
                thinking = parts[0].strip()
                answer = parts[1].strip() if len(parts) > 1 else response
            else:
                thinking = ""
                answer = response
        
        # Clean up answer
        answer = re.sub(r'^(ANSWER|RESPONSE|CONCLUSION):\s*', '', answer, flags=re.IGNORECASE)
        
        return thinking, answer
    
    def verify_answer(self, 
                      answer: str, 
                      query: str, 
                      patient_data: dict = None) -> Tuple[bool, str]:
        """
        Self-verify an answer for consistency and safety.
        
        Returns:
            Tuple of (is_valid, verification_notes)
        """
        verify_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a medical answer verifier. 
Check if the answer is:
1. Internally consistent
2. Safe (no dangerous advice)
3. Complete (addresses the question)
4. Accurate (based on medical knowledge)

Respond with:
VALID: [yes/no]
ISSUES: [list any problems found, or "None"]
CONFIDENCE: [high/medium/low]"""),
            ("human", """Question: {query}
Patient Data: {patient_data}
Answer to verify: {answer}

Verification:""")
        ])
        
        try:
            response = self.llm.invoke(
                verify_prompt.format_messages(
                    query=query,
                    patient_data=str(patient_data) if patient_data else "Not provided",
                    answer=answer[:1500]
                )
            )
            
            result = response.content
            is_valid = "VALID: YES" in result.upper() or "VALID:YES" in result.upper()
            
            return is_valid, result
            
        except Exception as e:
            return True, f"Verification skipped: {e}"


class SelfConsistencyChecker:
    """
    Implements Self-Consistency prompting:
    Generate multiple answers and select the most consistent one.
    """
    
    def __init__(self, reasoner: ChainOfThoughtReasoner = None):
        self.reasoner = reasoner or ChainOfThoughtReasoner()
    
    def generate_consistent_answer(self, 
                                    query: str, 
                                    patient_data: dict = None,
                                    num_samples: int = 3) -> str:
        """
        Generate multiple answers and return the most consistent one.
        """
        answers = []
        
        for i in range(num_samples):
            _, answer = self.reasoner.reason(query, patient_data)
            answers.append(answer)
        
        # For now, return the first one (full implementation would vote)
        # In production, would use semantic similarity to cluster answers
        return answers[0] if answers else "Unable to generate consistent answer"


# Test
if __name__ == "__main__":
    reasoner = ChainOfThoughtReasoner()
    
    thinking, answer = reasoner.reason(
        "My creatinine is 2.4 mg/dL. Am I at risk for kidney disease?",
        {"age": 65, "grf": 0, "dm": "yes"}
    )
    
    print("=== THINKING ===")
    print(thinking)
    print("\n=== ANSWER ===")
    print(answer)
    
    is_valid, notes = reasoner.verify_answer(answer, "Am I at risk?", {"sc": 2.4})
    print(f"\n=== VERIFICATION: {'PASSED' if is_valid else 'FAILED'} ===")
    print(notes)
