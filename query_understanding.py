# Query Understanding Agent - First layer of intelligence
# Refines and understands user queries before routing to other agents

import os
import json
from typing import Tuple, Dict, Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

class QueryUnderstandingAgent:
    """
    First layer agent that sits at the top of the cognitive architecture.
    
    Functions:
    1. INTENT CLASSIFICATION - What does the user really want?
    2. QUERY REFINEMENT - Optimize query for search/retrieval
    3. ENTITY EXTRACTION - Pull out medical terms
    4. CONTEXT BUILDING - Include relevant prior conversation
    
    This is like the "receptionist" of the doctor's office:
    - Understands what patient needs
    - Routes to right specialist
    - Provides context to the doctor
    """
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model="nvidia/nemotron-3-nano-30b-a3b:free",
            temperature=0.1,  # Low temp for precision
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY")
        )
        
        self.understanding_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a Query Understanding Agent for a Kidney Disease AI assistant.

Your job is to:
1. Understand the TRUE INTENT of the patient's question
2. Refine the query for optimal medical information retrieval
3. Extract key medical entities
4. Classify the query type

RESPOND IN THIS EXACT JSON FORMAT:
{
    "original_query": "the user's original question",
    "intent": "one of: diagnosis, prognosis, treatment, diet, medication, lifestyle, emergency, document_analysis, general_question",
    "refined_query": "a clearer, more medically-precise version of the question",
    "search_keywords": "3-5 keywords for web search, space-separated",
    "medical_entities": ["list", "of", "medical", "terms"],
    "requires_patient_data": true/false,
    "urgency": "low/medium/high/critical"
}

EXAMPLES:
Q: "how long does a patient survive after he is diagnosed with ckd"
{
    "original_query": "how long does a patient survive after he is diagnosed with ckd",
    "intent": "prognosis",
    "refined_query": "What is the life expectancy and survival rate for patients diagnosed with chronic kidney disease (CKD) by stage?",
    "search_keywords": "CKD survival rate prognosis life expectancy stage",
    "medical_entities": ["CKD", "chronic kidney disease", "survival", "prognosis"],
    "requires_patient_data": true,
    "urgency": "medium"
}

Q: "analyze the uploaded lab report"
{
    "original_query": "analyze the uploaded lab report",
    "intent": "document_analysis",
    "refined_query": "Interpret the clinical biomarkers and findings from the attached patient medical report in the context of kidney health.",
    "search_keywords": "medical report analysis lab values interpretation",
    "medical_entities": ["lab report", "analysis"],
    "requires_patient_data": true,
    "urgency": "medium"
}

Q: "provide me a food diet chart for an entire day"
{
    "original_query": "provide me a food diet chart for an entire day",
    "intent": "diet",
    "refined_query": "What is the recommended daily diet plan for CKD patients including breakfast, lunch, dinner and snacks with low potassium and sodium?",
    "search_keywords": "CKD renal diet meal plan low potassium sodium",
    "medical_entities": ["diet", "renal diet", "potassium", "sodium", "meal plan"],
    "requires_patient_data": true,
    "urgency": "low"
}"""),
            ("human", """Query: {query}

Patient Context (if available): {patient_context}
Recent Conversation: {conversation_history}

Understand this query and respond in JSON format:""")
        ])
    
    def understand(self, 
                   query: str, 
                   patient_context: str = "",
                   conversation_history: str = "") -> Dict:
        """
        Understand the user's query and return structured analysis.
        """
        try:
            response = self.llm.invoke(
                self.understanding_prompt.format_messages(
                    query=query,
                    patient_context=patient_context or "No patient data available",
                    conversation_history=conversation_history or "No prior conversation"
                )
            )
            
            # Parse JSON response
            content = response.content.strip()
            
            # Handle markdown code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            result = json.loads(content)
            return result
            
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            return {
                "original_query": query,
                "intent": "general_question",
                "refined_query": query,
                "search_keywords": query,
                "medical_entities": [],
                "requires_patient_data": False,
                "urgency": "low"
            }
        except Exception as e:
            return {
                "original_query": query,
                "intent": "general_question",
                "refined_query": query,
                "search_keywords": query,
                "medical_entities": [],
                "requires_patient_data": False,
                "urgency": "low",
                "error": str(e)
            }
    
    def get_optimized_search_query(self, query: str, patient_context: str = "") -> str:
        """
        Quick method to just get an optimized search query.
        """
        understanding = self.understand(query, patient_context)
        return understanding.get("search_keywords", query)
    
    def get_refined_query(self, query: str, patient_context: str = "") -> str:
        """
        Get a refined, clearer version of the query.
        """
        understanding = self.understand(query, patient_context)
        return understanding.get("refined_query", query)


# Singleton
_agent = None

def get_query_agent() -> QueryUnderstandingAgent:
    global _agent
    if _agent is None:
        _agent = QueryUnderstandingAgent()
    return _agent


# Test
if __name__ == "__main__":
    agent = QueryUnderstandingAgent()
    
    # Test queries
    test_queries = [
        "how long does a patient survive after he is diagnosed with ckd",
        "provide me a food diet chart for an entire day",
        "my creatinine is 2.4, am I in danger?"
    ]
    
    for q in test_queries:
        print(f"\nQuery: {q}")
        result = agent.understand(q)
        print(f"Intent: {result.get('intent')}")
        print(f"Refined: {result.get('refined_query')}")
        print(f"Keywords: {result.get('search_keywords')}")
