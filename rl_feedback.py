"""
Agentic Reinforcement Learning Feedback System
Uses LLM agents to analyze quality scores, identify failure patterns,
and generate improvement strategies for continuous model enhancement.
"""
import json
import os
from datetime import datetime
from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, List, Tuple
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

FEEDBACK_LOG_PATH = "rl_feedback_log.jsonl"
IMPROVEMENT_LOG_PATH = "rl_improvements.jsonl"

@dataclass
class FeedbackEntry:
    """Single feedback entry for RL training."""
    timestamp: str
    query: str
    response: str
    patient_context: str
    quality_score: float
    accuracy: float
    safety: float
    helpfulness: float
    evidence: float
    reasoning: str
    suggestions: List[str] = field(default_factory=list)
    
@dataclass
class ImprovementSuggestion:
    """Agent-generated improvement suggestion."""
    timestamp: str
    failure_pattern: str
    root_cause: str
    improved_prompt: str
    priority: str  # high, medium, low
    estimated_impact: float


class RLFeedbackAgent:
    """
    Agentic RL Feedback System that actively learns from quality scores.
    
    Capabilities:
    1. Pattern Recognition: Identifies common failure modes
    2. Root Cause Analysis: Determines why responses scored poorly
    3. Prompt Improvement: Generates better prompts for low-scoring queries
    4. Reward Shaping: Calculates nuanced reward signals for RL training
    5. Self-Improvement: Suggests architectural/prompt modifications
    """
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model="nvidia/nemotron-3-nano-30b-a3b:free",
            temperature=0.3,
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY")
        )
        
        # Analysis prompts
        self.pattern_analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an AI improvement specialist analyzing feedback patterns.
            
Your task is to identify recurring failure patterns in AI responses and suggest improvements.

For each pattern you identify, provide:
1. FAILURE_PATTERN: Brief description of what went wrong
2. ROOT_CAUSE: Why this failure occurs
3. IMPROVED_PROMPT: A better system prompt or instruction that would fix this
4. PRIORITY: high/medium/low based on frequency and severity
5. ESTIMATED_IMPACT: Score 0-1 of how much this fix would improve overall quality

Format your response as JSON:
{{
    "patterns": [
        {{
            "failure_pattern": "...",
            "root_cause": "...",
            "improved_prompt": "...",
            "priority": "high",
            "estimated_impact": 0.8
        }}
    ],
    "summary": "Brief summary of main issues"
}}"""),
            ("human", """Analyze these low-scoring responses and identify improvement patterns:

{feedback_entries}

Current scoring statistics:
- Average Score: {avg_score}
- Common Issues: {common_issues}

Identify the top 3 most impactful improvement patterns.""")
        ])
        
        self.reward_shaping_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a reward engineer for medical AI systems.
            
Calculate a nuanced reward signal based on the quality dimensions.
Consider:
- Safety is CRITICAL (weight: 0.4)
- Accuracy is very important (weight: 0.3)
- Helpfulness matters (weight: 0.2)
- Evidence-based reasoning (weight: 0.1)

Also apply penalties for:
- Missing disclaimers (-0.2)
- Overconfident claims (-0.15)
- Hallucinated drug names (-0.5)

Output format:
{{
    "base_reward": <float 0-1>,
    "safety_bonus": <float -0.5 to 0.5>,
    "accuracy_bonus": <float -0.3 to 0.3>,
    "penalties": [<list of applied penalties>],
    "final_reward": <float -1 to 1>,
    "reasoning": "Brief explanation"
}}"""),
            ("human", """Calculate reward for this medical AI response:

Query: {query}
Response: {response}
Patient Context: {context}

Quality Scores:
- Overall: {overall}/10
- Safety: {safety}/10
- Accuracy: {accuracy}/10
- Helpfulness: {helpfulness}/10
- Evidence: {evidence}/10

Judge Feedback: {reasoning}""")
        ])
        
        self.prompt_improvement_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a prompt engineer specializing in medical AI.
            
Given a query that resulted in a low-quality response, generate an improved
version of the system prompt that would produce better results.

Focus on:
1. Clearer medical context requirements
2. Explicit safety instructions
3. Evidence-based reasoning requirements
4. Appropriate uncertainty language

Output the improved prompt directly, ready to use."""),
            ("human", """The following query received a low score ({score}/10):

Original Query: {query}
Response Issues: {issues}
Patient Context: {context}

Generate an improved system prompt that would produce a better response.""")
        ])
    
    def store_feedback(
        self,
        query: str,
        response: str,
        patient_context: str,
        score_obj,
        append_mode: bool = True
    ) -> Tuple[bool, Optional[float]]:
        """
        Store feedback and calculate reward signal.
        
        Returns:
            Tuple of (success, reward_signal)
        """
        try:
            entry = FeedbackEntry(
                timestamp=datetime.now().isoformat(),
                query=query,
                response=response[:2000],
                patient_context=patient_context[:500],
                quality_score=score_obj.overall if score_obj else 0.0,
                accuracy=getattr(score_obj, 'accuracy', 0.0) if score_obj else 0.0,
                safety=getattr(score_obj, 'safety', 0.0) if score_obj else 0.0,
                helpfulness=getattr(score_obj, 'helpfulness', 0.0) if score_obj else 0.0,
                evidence=getattr(score_obj, 'evidence', 0.0) if score_obj else 0.0,
                reasoning=getattr(score_obj, 'reasoning', '') if score_obj else '',
                suggestions=getattr(score_obj, 'suggestions', [])[:3] if score_obj else []
            )
            
            mode = 'a' if append_mode else 'w'
            with open(FEEDBACK_LOG_PATH, mode, encoding='utf-8') as f:
                f.write(json.dumps(asdict(entry), ensure_ascii=False) + '\n')
            
            # Calculate reward signal for this entry
            reward = self._calculate_reward_signal(entry)
            
            # If score is low, trigger improvement analysis
            if entry.quality_score < 6.0:
                self._queue_for_improvement(entry)
            
            return True, reward
            
        except Exception as e:
            print(f"[RL Feedback] Storage error: {e}")
            return False, None
    
    def _calculate_reward_signal(self, entry: FeedbackEntry) -> float:
        """
        Calculate a nuanced reward signal using weighted dimensions.
        Fast local calculation (no LLM call for speed).
        """
        # Weights for different quality dimensions
        weights = {
            'safety': 0.40,
            'accuracy': 0.30,
            'helpfulness': 0.20,
            'evidence': 0.10
        }
        
        # Normalize scores to 0-1 range
        safety_norm = entry.safety / 10.0
        accuracy_norm = entry.accuracy / 10.0
        helpfulness_norm = entry.helpfulness / 10.0
        evidence_norm = entry.evidence / 10.0
        
        # Weighted base reward
        base_reward = (
            weights['safety'] * safety_norm +
            weights['accuracy'] * accuracy_norm +
            weights['helpfulness'] * helpfulness_norm +
            weights['evidence'] * evidence_norm
        )
        
        # Apply penalties
        penalties = 0.0
        response_lower = entry.response.lower()
        
        # Missing disclaimer penalty
        if 'disclaimer' not in response_lower and 'consult' not in response_lower:
            penalties -= 0.1
        
        # Overconfidence penalty (phrases like "definitely", "certainly", "100%")
        overconfident_phrases = ['definitely', 'certainly', '100%', 'guaranteed', 'always works']
        for phrase in overconfident_phrases:
            if phrase in response_lower:
                penalties -= 0.05
                break
        
        # Safety bonus for explicit warnings
        if 'warning' in response_lower or 'caution' in response_lower:
            penalties += 0.05
        
        # Final reward in range [-1, 1]
        final_reward = max(-1.0, min(1.0, (base_reward * 2 - 1) + penalties))
        
        return final_reward
    
    def _queue_for_improvement(self, entry: FeedbackEntry):
        """Queue low-scoring entries for batch improvement analysis."""
        try:
            with open(IMPROVEMENT_LOG_PATH, 'a', encoding='utf-8') as f:
                f.write(json.dumps({
                    'timestamp': entry.timestamp,
                    'query': entry.query[:500],
                    'score': entry.quality_score,
                    'issues': entry.suggestions[:2],
                    'reasoning': entry.reasoning[:200]
                }, ensure_ascii=False) + '\n')
        except Exception:
            pass
    
    def analyze_failure_patterns(self, min_entries: int = 5) -> Dict:
        """
        Analyze accumulated feedback to identify improvement patterns.
        Uses LLM to find root causes and suggest fixes.
        """
        entries = self._load_low_scoring_entries(threshold=6.0)
        
        if len(entries) < min_entries:
            return {"status": "insufficient_data", "count": len(entries)}
        
        # Prepare summary for LLM
        common_issues = {}
        for entry in entries:
            for suggestion in entry.get('suggestions', []):
                common_issues[suggestion] = common_issues.get(suggestion, 0) + 1
        
        top_issues = sorted(common_issues.items(), key=lambda x: x[1], reverse=True)[:5]
        
        avg_score = sum(e.get('quality_score', 0) for e in entries) / len(entries)
        
        # Format entries for LLM
        formatted_entries = "\n\n".join([
            f"Query: {e['query'][:200]}\nScore: {e['quality_score']}/10\nIssues: {e.get('reasoning', 'N/A')[:150]}"
            for e in entries[:10]  # Limit to 10 for context window
        ])
        
        try:
            chain = self.pattern_analysis_prompt | self.llm | StrOutputParser()
            result = chain.invoke({
                "feedback_entries": formatted_entries,
                "avg_score": f"{avg_score:.1f}",
                "common_issues": str(top_issues)
            })
            
            # Parse JSON response
            import re
            json_match = re.search(r'\{[\s\S]*\}', result)
            if json_match:
                return json.loads(json_match.group())
            return {"status": "parse_error", "raw": result}
            
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def generate_improved_prompt(self, query: str, context: str, issues: List[str], score: float) -> str:
        """
        Generate an improved system prompt for a low-scoring query.
        """
        try:
            chain = self.prompt_improvement_prompt | self.llm | StrOutputParser()
            return chain.invoke({
                "query": query,
                "context": context,
                "issues": ", ".join(issues),
                "score": score
            })
        except Exception as e:
            return f"Error generating improvement: {e}"
    
    def _load_low_scoring_entries(self, threshold: float = 6.0) -> List[Dict]:
        """Load entries below the quality threshold."""
        entries = []
        if not os.path.exists(FEEDBACK_LOG_PATH):
            return entries
        
        with open(FEEDBACK_LOG_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    if entry.get('quality_score', 10) < threshold:
                        entries.append(entry)
                except json.JSONDecodeError:
                    continue
        
        return entries
    
    def get_stats(self) -> Dict:
        """Get statistics about collected feedback."""
        entries = []
        if os.path.exists(FEEDBACK_LOG_PATH):
            with open(FEEDBACK_LOG_PATH, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        entries.append(json.loads(line.strip()))
                    except:
                        continue
        
        if not entries:
            return {"total": 0, "avg_score": 0, "high_quality": 0, "low_quality": 0}
        
        scores = [e.get('quality_score', 0) for e in entries]
        return {
            "total": len(entries),
            "avg_score": sum(scores) / len(scores),
            "high_quality": len([s for s in scores if s >= 7.0]),
            "low_quality": len([s for s in scores if s < 5.0]),
            "safety_avg": sum(e.get('safety', 0) for e in entries) / len(entries),
            "accuracy_avg": sum(e.get('accuracy', 0) for e in entries) / len(entries),
            "improvement_queue": self._count_improvement_queue()
        }
    
    def _count_improvement_queue(self) -> int:
        """Count entries in improvement queue."""
        if not os.path.exists(IMPROVEMENT_LOG_PATH):
            return 0
        with open(IMPROVEMENT_LOG_PATH, 'r') as f:
            return sum(1 for _ in f)
    
    def export_training_data(self, output_path: str = "rl_training_data.json", min_score: float = 7.0) -> int:
        """Export high-quality entries for fine-tuning."""
        entries = []
        if os.path.exists(FEEDBACK_LOG_PATH):
            with open(FEEDBACK_LOG_PATH, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        if entry.get('quality_score', 0) >= min_score:
                            entries.append(entry)
                    except:
                        continue
        
        # Format for instruction fine-tuning
        training_data = []
        for entry in entries:
            training_data.append({
                "instruction": entry['query'],
                "input": entry['patient_context'],
                "output": entry['response'],
                "reward": self._calculate_reward_signal(FeedbackEntry(**entry))
            })
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, indent=2, ensure_ascii=False)
        
        return len(training_data)


# Singleton instance
_agent = None

def get_rl_agent() -> RLFeedbackAgent:
    """Get singleton RL feedback agent."""
    global _agent
    if _agent is None:
        _agent = RLFeedbackAgent()
    return _agent

def store_feedback(query: str, response: str, patient_context: str, score_obj) -> bool:
    """Convenience function for storing feedback."""
    agent = get_rl_agent()
    success, _ = agent.store_feedback(query, response, patient_context, score_obj)
    return success


if __name__ == "__main__":
    # Test the agentic RL system
    agent = RLFeedbackAgent()
    
    # Check stats
    stats = agent.get_stats()
    print(f"Current Stats: {stats}")
    
    # If we have enough data, analyze patterns
    if stats['total'] >= 5:
        print("\nAnalyzing failure patterns...")
        patterns = agent.analyze_failure_patterns()
        print(json.dumps(patterns, indent=2))
