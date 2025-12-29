# Centralized Pipeline Coordinator
# Implements ReAct pattern (Observe -> Plan -> Act) and coordinates all agents

import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

class AgentType(Enum):
    SQL = "sql"
    RAG = "rag"
    COUNCIL = "council"
    SAFETY = "safety"
    JUDGE = "judge"

@dataclass
class PipelineStep:
    """Represents a single step in the ReAct loop."""
    step_type: str  # observe, plan, act
    agent: Optional[AgentType]
    input_data: Dict
    output_data: Optional[Dict] = None
    reasoning: str = ""

@dataclass
class PipelineResult:
    """Final result from the pipeline."""
    response: str
    steps: List[PipelineStep]
    quality_score: Optional[float]
    safety_passed: bool
    metadata: Dict


class PipelineCoordinator:
    """
    ReAct-based pipeline coordinator that manages the full workflow:
    
    OBSERVE: Gather context from conversation history, patient data, uploaded docs
    PLAN: Determine which agents to invoke and in what order
    ACT: Execute agents and collect results
    
    Features:
    - Maintains conversation memory across requests
    - Initializes SQL database on first data query
    - Coordinates safety and judge agents
    - Logs pipeline execution for debugging
    """
    
    def __init__(self):
        self._rag_engine = None
        self._sql_agent = None
        self._council = None
        self._safety_agent = None
        self._judge_agent = None
        
        self.conversation_memory = []
        self.pipeline_history: List[PipelineResult] = []
        self._sql_initialized = False
    
    # === LAZY LOADING (Initialize agents only when needed) ===
    
    @property
    def rag_engine(self):
        if self._rag_engine is None:
            from rag_engine import RAGEngine
            self._rag_engine = RAGEngine()
        return self._rag_engine
    
    @property
    def sql_agent(self):
        if self._sql_agent is None:
            from sql_agent import SQLAgent
            self._sql_agent = SQLAgent()
            self._sql_initialized = True
        return self._sql_agent
    
    @property
    def council(self):
        if self._council is None:
            from council import MedicalCouncil
            self._council = MedicalCouncil()
        return self._council
    
    @property
    def safety_agent(self):
        if self._safety_agent is None:
            from safety_agent import SafetyGuardrailAgent
            self._safety_agent = SafetyGuardrailAgent()
        return self._safety_agent
    
    @property
    def judge_agent(self):
        if self._judge_agent is None:
            from judge_agent import JudgeAgent
            self._judge_agent = JudgeAgent()
        return self._judge_agent
    
    # === OBSERVE PHASE ===
    
    def observe(self, 
                query: str, 
                patient_data: Dict, 
                uploaded_docs: List = None) -> PipelineStep:
        """
        Gather all available context for the current query.
        """
        context = {
            "query": query,
            "patient_data": patient_data,
            "has_uploaded_docs": bool(uploaded_docs),
            "conversation_length": len(self.conversation_memory),
            "patient_has_data": patient_data.get('sc', 0) > 0,
        }
        
        # Summarize conversation history
        if self.conversation_memory:
            recent = self.conversation_memory[-4:]  # Last 2 turns
            context["recent_context"] = " | ".join([f"{m['role']}: {m['content'][:50]}..." for m in recent])
        
        return PipelineStep(
            step_type="observe",
            agent=None,
            input_data={"raw_query": query},
            output_data=context,
            reasoning="Gathered patient context and conversation history"
        )
    
    # === PLAN PHASE ===
    
    def plan(self, observation: PipelineStep) -> PipelineStep:
        """
        Determine which agents to invoke based on observation.
        """
        context = observation.output_data
        query = context["query"]
        
        # Use query planner for routing
        from query_planner import QueryPlanner
        planner = QueryPlanner()
        plan = planner.get_execution_plan(query, context.get("patient_data"))
        
        # Build execution plan
        agents_to_run = []
        
        if plan['tool'] == 'sql':
            agents_to_run = [AgentType.SQL, AgentType.SAFETY]
        elif plan['tool'] == 'council':
            agents_to_run = [AgentType.COUNCIL, AgentType.SAFETY, AgentType.JUDGE]
        elif plan['tool'] == 'rag':
            agents_to_run = [AgentType.RAG, AgentType.SAFETY]
        else:
            agents_to_run = []  # Simple response
        
        return PipelineStep(
            step_type="plan",
            agent=None,
            input_data=context,
            output_data={
                "route": plan['tool'],
                "agents": [a.value for a in agents_to_run],
                "requires_safety": AgentType.SAFETY in agents_to_run,
                "requires_judge": AgentType.JUDGE in agents_to_run,
            },
            reasoning=plan['reasoning']
        )
    
    # === ACT PHASE ===
    
    def act(self, plan_step: PipelineStep, raw_query: str, patient_data: Dict) -> PipelineStep:
        """
        Execute the planned agents and collect results.
        """
        route = plan_step.output_data["route"]
        results = {}
        
        try:
            if route == "sql":
                sql_query, answer, df = self.sql_agent.query(raw_query)
                results["sql_query"] = sql_query
                results["answer"] = answer
                results["row_count"] = len(df) if df is not None else 0
                
            elif route == "council":
                patient_str = str(patient_data)
                opinions = self.council.consult(raw_query, patient_str)
                response = self.council.synthesize(raw_query, opinions)
                results["answer"] = response
                results["opinions"] = opinions
                
            elif route == "rag":
                # Use conversation history
                response = self.rag_engine.chat_reasoning(
                    raw_query, 
                    self.conversation_memory[-10:],  # Last 5 turns
                    None
                )
                results["answer"] = response
                
            else:  # simple
                results["answer"] = self._simple_response(raw_query)
            
            # Safety check
            if plan_step.output_data.get("requires_safety"):
                safety_check = self.safety_agent.validate(
                    results.get("answer", ""),
                    patient_data,
                    run_advisor=False
                )
                results["safety_passed"] = safety_check.passed
                results["safety_severity"] = safety_check.severity
                results["answer"] = safety_check.modified_output
            
            # Judge evaluation
            if plan_step.output_data.get("requires_judge"):
                score = self.judge_agent.score_response(
                    results.get("answer", ""),
                    raw_query,
                    str(patient_data)
                )
                results["judge_score"] = score.overall
                results["judge_reasoning"] = score.reasoning
                
        except Exception as e:
            results["error"] = str(e)
            results["answer"] = f"An error occurred: {e}"
        
        return PipelineStep(
            step_type="act",
            agent=AgentType(route) if route in ["sql", "rag", "council"] else None,
            input_data={"query": raw_query},
            output_data=results,
            reasoning=f"Executed {route} agent"
        )
    
    def _simple_response(self, query: str) -> str:
        """Handle simple greetings/meta queries."""
        return "Hello! I'm NephroAI, your kidney health assistant. 🩺\n\nAsk me anything about kidney health, lab values, or patient data. I'll route your question to the right tool automatically."
    
    # === MAIN PIPELINE ===
    
    def run(self, 
            query: str, 
            patient_data: Dict, 
            uploaded_docs: List = None) -> PipelineResult:
        """
        Run the full ReAct pipeline: Observe -> Plan -> Act
        """
        steps = []
        
        # 1. OBSERVE
        observe_step = self.observe(query, patient_data, uploaded_docs)
        steps.append(observe_step)
        
        # 2. PLAN
        plan_step = self.plan(observe_step)
        steps.append(plan_step)
        
        # 3. ACT
        act_step = self.act(plan_step, query, patient_data)
        steps.append(act_step)
        
        # Update conversation memory
        self.conversation_memory.append({"role": "user", "content": query})
        self.conversation_memory.append({"role": "assistant", "content": act_step.output_data.get("answer", "")})
        
        # Trim memory
        if len(self.conversation_memory) > 20:
            self.conversation_memory = self.conversation_memory[-20:]
        
        # Build result
        result = PipelineResult(
            response=act_step.output_data.get("answer", ""),
            steps=steps,
            quality_score=act_step.output_data.get("judge_score"),
            safety_passed=act_step.output_data.get("safety_passed", True),
            metadata={
                "route": plan_step.output_data.get("route"),
                "sql_query": act_step.output_data.get("sql_query"),
                "safety_severity": act_step.output_data.get("safety_severity"),
            }
        )
        
        self.pipeline_history.append(result)
        return result


# Singleton instance for use across app
_coordinator = None

def get_coordinator() -> PipelineCoordinator:
    """Get or create the singleton pipeline coordinator."""
    global _coordinator
    if _coordinator is None:
        _coordinator = PipelineCoordinator()
    return _coordinator


# Test
if __name__ == "__main__":
    coord = get_coordinator()
    
    result = coord.run(
        "What is the average creatinine in the database?",
        {"sc": 0, "age": 0}
    )
    
    print(f"Route: {result.metadata['route']}")
    print(f"Response: {result.response[:200]}...")
    print(f"Steps: {len(result.steps)}")
