import os
import json
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from feedback_agent import FeedbackAgent
from rag_engine import RAGEngine

class OrchestratorAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            model="nvidia/nemotron-3-nano-30b-a3b:free", 
            temperature=0.3, 
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY")
        )
        self.feedback_agent = FeedbackAgent()
        self.rag_engine = RAGEngine()

    def plan_analysis(self, patient_data, prediction):
        """
        Decides what aspects of the patient's health need detailed analysis.
        Returns a list of tasks (questions) to answer.
        """
        system_prompt = """You are a Clinical Case Manager.
        Based on the patient's data and the prediction model's result, plan a comprehensive analysis report.
        
        Identify the top 3 most critical areas to investigate.
        Examples:
        - If BP is high -> "Analyze impact of Hypertension on Kidney function"
        - If Diabetic -> "Explain Diabetic Nephropathy risks"
        - If CKD Positive -> "Suggest immediate lifestyle modifications for Stage {stage}"
        
        Return a JSON object with a list of 'tasks'.
        Example: {"tasks": ["Question 1", "Question 2", "Question 3"]}
        """
        
        user_input = f"""
        Prediction: {prediction}
        Patient Data: {patient_data}
        """

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", user_input)
        ])
        
        chain = prompt | self.llm | JsonOutputParser()
        
        try:
            plan = chain.invoke({})
            return plan.get("tasks", ["Analyze general kidney health guidelines."])
        except Exception as e:
            print(f"Planning Error: {e}")
            return ["Analyze general kidney health and preventive measures."]

    def _process_task(self, task, patient_data, report_vector_store):
        """Helper to run RAG + Feedback for a single task (for parallel execution)."""
        # RAG Retrieval with Context
        context_enhanced_query = f"""
        Patient Data: {patient_data}
        In light of this patient data, address the following task: {task}
        """
        
        # Pass the report index if available
        rag_response = self.rag_engine.chat_reasoning(
            context_enhanced_query, 
            chat_history=[],
            report_vector_store=report_vector_store
        )
        
        # Feedback Loop
        critique = self.feedback_agent.critique_and_refine(rag_response, context=task)
        
        # Parse refined response
        if "[REFINED_RESPONSE]" in critique:
            final_content = critique.split("[REFINED_RESPONSE]")[1].strip()
        else:
            final_content = critique 
        
        return {
            "task": task,
            "raw_response": rag_response,
            "critique": critique,
            "final_content": final_content
        }

    def execute_workflow(self, patient_data, prediction, report_vector_store=None, status_callback=None):
        """Runs the Planner -> Executor -> Feedback loop in PARALLEL."""
        import concurrent.futures
        
        # 1. PLAN
        if status_callback: status_callback("🧠 Clinical Agent: Analying patient data and planning...")
        tasks = self.plan_analysis(patient_data, prediction)
        
        if status_callback: status_callback(f"🚀 Parallel Execution: Running {len(tasks)} research agents...")
        
        # 2. EXECUTE (Parallel)
        report_sections = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            # Create a future for each task
            future_to_task = {
                executor.submit(self._process_task, task, patient_data, report_vector_store): task 
                for task in tasks
            }
            
            for future in concurrent.futures.as_completed(future_to_task):
                try:
                    section = future.result()
                    report_sections.append(section)
                    if status_callback: status_callback(f"✅ Completed: {section['task']}")
                except Exception as exc:
                    print(f"Task generated an exception: {exc}")
        
        return report_sections

if __name__ == "__main__":
    # Test
    orch = OrchestratorAgent()
    # Dummy data
    res = orch.execute_workflow({"age": 60, "bp": 140, "sc": 2.5}, "Positive (CKD)")
    print(res)
