"""
Reinforcement Learning Feedback Storage
Stores quality scores from the Judge Agent for future model improvement.
"""
import json
import os
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional, Dict, List

FEEDBACK_LOG_PATH = "rl_feedback_log.jsonl"

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
    suggestions: List[str]
    
def store_feedback(
    query: str,
    response: str,
    patient_context: str,
    score_obj,  # JudgeScore object from judge_agent
    append_mode: bool = True
) -> bool:
    """
    Store feedback entry for RL training data collection.
    
    Args:
        query: The user's query
        response: The AI's response
        patient_context: Patient data string
        score_obj: JudgeScore dataclass from judge_agent
        append_mode: If True, append to existing file. If False, overwrite.
    
    Returns:
        True if successfully stored, False otherwise.
    """
    try:
        entry = FeedbackEntry(
            timestamp=datetime.now().isoformat(),
            query=query,
            response=response[:2000],  # Truncate to avoid huge files
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
        
        return True
    except Exception as e:
        print(f"[RL Feedback] Storage error: {e}")
        return False

def load_feedback_entries(min_score: float = 0.0) -> List[Dict]:
    """
    Load feedback entries, optionally filtered by minimum quality score.
    Useful for creating training data with only high-quality examples.
    
    Args:
        min_score: Minimum quality score to include (0-10)
    
    Returns:
        List of feedback entries as dictionaries.
    """
    entries = []
    if not os.path.exists(FEEDBACK_LOG_PATH):
        return entries
    
    with open(FEEDBACK_LOG_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                if entry.get('quality_score', 0) >= min_score:
                    entries.append(entry)
            except json.JSONDecodeError:
                continue
    
    return entries

def get_feedback_stats() -> Dict:
    """
    Get statistics about collected feedback for monitoring.
    """
    entries = load_feedback_entries()
    if not entries:
        return {"total": 0, "avg_score": 0, "high_quality": 0, "low_quality": 0}
    
    scores = [e.get('quality_score', 0) for e in entries]
    return {
        "total": len(entries),
        "avg_score": sum(scores) / len(scores),
        "high_quality": len([s for s in scores if s >= 7.0]),
        "low_quality": len([s for s in scores if s < 5.0]),
        "safety_avg": sum(e.get('safety', 0) for e in entries) / len(entries),
        "accuracy_avg": sum(e.get('accuracy', 0) for e in entries) / len(entries)
    }

def export_training_data(output_path: str = "rl_training_data.json", min_score: float = 7.0) -> int:
    """
    Export high-quality feedback entries as training data for fine-tuning.
    
    Args:
        output_path: Path to save the training data
        min_score: Only include entries with score >= this value
    
    Returns:
        Number of entries exported.
    """
    entries = load_feedback_entries(min_score)
    
    # Format for instruction fine-tuning
    training_data = []
    for entry in entries:
        training_data.append({
            "instruction": entry['query'],
            "input": entry['patient_context'],
            "output": entry['response']
        })
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(training_data, f, indent=2, ensure_ascii=False)
    
    return len(training_data)


if __name__ == "__main__":
    # Test
    from dataclasses import dataclass
    
    @dataclass
    class MockScore:
        overall: float = 7.5
        accuracy: float = 8.0
        safety: float = 9.0
        helpfulness: float = 7.0
        evidence: float = 6.5
        reasoning: str = "Good response with some room for improvement"
        suggestions: List[str] = None
        
        def __post_init__(self):
            self.suggestions = ["Add more citations", "Include disclaimer"]
    
    # Store test entry
    store_feedback(
        query="What are the symptoms of Stage 3 CKD?",
        response="Stage 3 CKD symptoms include fatigue, swelling, and changes in urination.",
        patient_context="Age: 65, Creatinine: 2.1",
        score_obj=MockScore()
    )
    
    # Check stats
    stats = get_feedback_stats()
    print(f"Feedback Stats: {stats}")
