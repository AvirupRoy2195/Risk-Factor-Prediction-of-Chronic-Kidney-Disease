# Query Planner - Routes queries to the right tool
# SQL (data queries) | RAG (medical knowledge) | Council (diagnosis) | Hybrid (data + knowledge)

import os
import re
from typing import Literal, Tuple
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

QueryType = Literal["sql", "rag", "council", "simple", "hybrid"]

class QueryPlanner:
    """
    Intelligent query router that determines the best tool for each query:
    - SQL: Data queries (counts, averages, patient lookups)
    - RAG: Medical knowledge questions
    - Council: Complex diagnostic reasoning
    - Hybrid: Combines SQL data with RAG knowledge (e.g., "What's the avg creatinine and is it normal?")
    - Simple: Basic greetings/clarifications
    """
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model="nvidia/nemotron-3-nano-30b-a3b:free",
            temperature=0,
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY")
        )
        
        # Pattern-based shortcuts for fast routing
        self.sql_patterns = [
            r'\b(how many|count|average|sum|total|statistics|patients with|list all)\b',
            r'\b(highest|lowest|maximum|minimum|top \d+|bottom \d+)\b',
            r'\b(compare|distribution|percentage|ratio)\b',
            r'\b(in the database|in our data|from the records)\b',
        ]
        
        self.council_patterns = [
            r'\b(diagnos|prescri|treat|recommend|should I|what to do)\b',
            r'\b(medication|drug|therapy|intervention)\b',
            r'\b(my creatinine|my gfr|my blood pressure|my results)\b',
            r'\b(am I at risk|is this dangerous|is this normal)\b',
            # Patient descriptions that need analysis
            r'\b(patient|patient is|years? old|yrs old|male|female)\b',
            r'\b(history of|with history|diabetes|hypertension|dialysis)\b',
            r'\b(analyze|assess|evaluate|check|review)\b',
        ]
        
        self.simple_patterns = [
            r'^(hi|hello|hey|thanks|ok|yes|no|bye)\b',
            r'\b(who are you|what can you do|help)\b',
        ]
        
        # Hybrid patterns: needs both data and knowledge
        self.hybrid_patterns = [
            r'\b(average|count|how many).+\b(normal|abnormal|healthy|dangerous|risk)\b',
            r'\b(patients with).+\b(should|recommend|treatment)\b',
            r'\b(compare|correlation).+\b(outcome|prognosis|guidelines)\b',
            r'\b(data|database|records).+\b(explain|interpret|meaning)\b',
        ]
    
    def _pattern_match(self, query: str) -> QueryType | None:
        """Fast pattern-based routing."""
        query_lower = query.lower()
        
        for pattern in self.simple_patterns:
            if re.search(pattern, query_lower):
                return "simple"
        
        # Check hybrid BEFORE sql/rag (hybrid has both data + knowledge components)
        for pattern in self.hybrid_patterns:
            if re.search(pattern, query_lower):
                return "hybrid"
        
        for pattern in self.sql_patterns:
            if re.search(pattern, query_lower):
                return "sql"
        
        for pattern in self.council_patterns:
            if re.search(pattern, query_lower):
                return "council"
        
        return None
    
    def plan(self, query: str, patient_data: dict = None) -> Tuple[QueryType, str]:
        """
        Determine the best tool for the query.
        
        Returns:
            Tuple of (query_type, reasoning)
        """
        # Try fast pattern matching first
        pattern_result = self._pattern_match(query)
        if pattern_result:
            return pattern_result, f"Pattern match: {pattern_result}"
        
        # Fall back to LLM classification
        classification_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a query router for a medical AI system.
Classify the user's query into ONE of these categories:

1. **sql** - Data queries about patient records, statistics, counts, averages
   Examples: "How many patients have stage 4 CKD?", "Average creatinine level?"
   
2. **rag** - General medical knowledge questions
   Examples: "What is GFR?", "Symptoms of kidney failure?", "What does albumin mean?"
   
3. **council** - Personalized diagnosis/treatment questions requiring expert reasoning
   Examples: "What should I do about my high creatinine?", "Is my kidney function normal?"
   
4. **simple** - Greetings, thanks, basic clarifications
   Examples: "Hello", "Thanks", "What can you do?"

Patient has data: {has_data}

Respond with ONLY the category name (sql/rag/council/simple) and nothing else."""),
            ("human", "{query}")
        ])
        
        try:
            response = self.llm.invoke(
                classification_prompt.format_messages(
                    query=query,
                    has_data="yes" if patient_data and patient_data.get('sc', 0) > 0 else "no"
                )
            )
            category = response.content.strip().lower()
            
            if category in ["sql", "rag", "council", "simple"]:
                return category, f"LLM classified as: {category}"
            else:
                return "rag", f"Default to RAG (unclear classification: {category})"
                
        except Exception as e:
            return "rag", f"Default to RAG (error: {e})"
    
    def get_execution_plan(self, query: str, patient_data: dict = None) -> dict:
        """
        Get a full execution plan with tool selection and parameters.
        """
        query_type, reasoning = self.plan(query, patient_data)
        
        plan = {
            "query": query,
            "tool": query_type,
            "reasoning": reasoning,
            "requires_deep_think": query_type == "council",
            "show_sources": query_type in ["sql", "rag"],
        }
        
        # Add tool-specific parameters
        if query_type == "sql":
            plan["sql_context"] = "Query patient database for statistics"
        elif query_type == "rag":
            plan["rag_context"] = "Search medical knowledge base"
        elif query_type == "council":
            plan["council_context"] = "Invoke medical council for expert opinion"
        
        return plan


# Test
if __name__ == "__main__":
    planner = QueryPlanner()
    
    test_queries = [
        "Hello!",
        "How many patients have CKD stage 5?",
        "What is creatinine?",
        "My creatinine is 2.4, am I at risk?",
        "Average GFR of patients with diabetes?",
        "What medications should I avoid with kidney disease?",
    ]
    
    for q in test_queries:
        result, reason = planner.plan(q)
        print(f"Query: {q}")
        print(f"  → {result} ({reason})\n")
