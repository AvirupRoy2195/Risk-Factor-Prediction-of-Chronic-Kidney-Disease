# Intelligent Memory Module - Hippocampus of the Brain
# Stores and retrieves key facts from conversation

from typing import Dict, List, Optional
from dataclasses import dataclass, field
import re

@dataclass  
class ConversationMemory:
    """
    Intelligent memory that extracts and stores key facts from conversation.
    
    Like the Hippocampus in the brain:
    - Stores important facts (patient values, diagnoses)
    - Retrieves relevant context for follow-up questions
    - Forgets irrelevant details
    """
    
    # Key facts extracted from conversation
    extracted_facts: Dict[str, str] = field(default_factory=dict)
    
    # Recent conversation (last N turns)
    recent_messages: List[Dict] = field(default_factory=list)
    max_messages: int = 10
    
    # Important medical values mentioned
    medical_values: Dict[str, float] = field(default_factory=dict)
    
    # Diagnosis/findings from previous turns
    findings: List[str] = field(default_factory=list)
    
    def add_message(self, role: str, content: str):
        """Add a message to memory and extract facts."""
        self.recent_messages.append({"role": role, "content": content})
        
        # Trim to max size
        if len(self.recent_messages) > self.max_messages:
            self.recent_messages = self.recent_messages[-self.max_messages:]
        
        # Extract facts from user messages
        if role == "user":
            self._extract_medical_values(content)
        
        # Extract findings from assistant messages
        if role == "assistant":
            self._extract_findings(content)
    
    def _extract_medical_values(self, content: str):
        """Extract medical values from user input."""
        content_lower = content.lower()
        
        # Creatinine patterns
        creat_match = re.search(r'creatinine\s*[:=]?\s*(\d+\.?\d*)', content_lower)
        if creat_match:
            self.medical_values['creatinine'] = float(creat_match.group(1))
            self.extracted_facts['creatinine'] = creat_match.group(1)
        
        # Age patterns
        age_match = re.search(r'(\d+)\s*(?:years?\s*old|yrs?\s*old|y\.?o\.?)', content_lower)
        if age_match:
            self.medical_values['age'] = float(age_match.group(1))
            self.extracted_facts['age'] = age_match.group(1)
        
        # GFR patterns
        gfr_match = re.search(r'(?:gfr|egfr)\s*[:=]?\s*(\d+)', content_lower)
        if gfr_match:
            self.medical_values['gfr'] = float(gfr_match.group(1))
            self.extracted_facts['gfr'] = gfr_match.group(1)
        
        # Diabetes/Hypertension
        if 'diabetes' in content_lower or 'diabetic' in content_lower:
            self.extracted_facts['diabetes'] = 'yes'
        if 'hypertension' in content_lower or 'high blood pressure' in content_lower:
            self.extracted_facts['hypertension'] = 'yes'
    
    def _extract_findings(self, content: str):
        """Extract key findings from assistant responses."""
        content_lower = content.lower()
        
        # CKD Stage
        stage_match = re.search(r'stage\s*(\d|i+v?)', content_lower)
        if stage_match:
            self.findings.append(f"CKD Stage {stage_match.group(1)}")
        
        # Risk level
        if 'high risk' in content_lower:
            self.findings.append("High CKD risk")
        elif 'low risk' in content_lower:
            self.findings.append("Low CKD risk")
    
    def get_context_summary(self) -> str:
        """Get a summary of important context for the next query."""
        summary_parts = []
        
        if self.medical_values:
            values = ", ".join([f"{k}={v}" for k, v in self.medical_values.items()])
            summary_parts.append(f"Known values: {values}")
        
        if self.extracted_facts:
            facts = ", ".join([f"{k}={v}" for k, v in self.extracted_facts.items()])
            summary_parts.append(f"Patient facts: {facts}")
        
        if self.findings:
            summary_parts.append(f"Prior findings: {', '.join(self.findings[-3:])}")
        
        return " | ".join(summary_parts) if summary_parts else "No prior context"
    
    def get_recent_conversation(self, n: int = 5) -> str:
        """Get last N messages as formatted string."""
        recent = self.recent_messages[-n*2:]  # N turns = 2N messages
        
        if not recent:
            return ""
        
        formatted = []
        for msg in recent:
            role = "Patient" if msg["role"] == "user" else "Doctor"
            content = msg["content"][:150] + "..." if len(msg["content"]) > 150 else msg["content"]
            formatted.append(f"{role}: {content}")
        
        return "\n".join(formatted)
    
    def clear(self):
        """Clear all memory."""
        self.extracted_facts = {}
        self.recent_messages = []
        self.medical_values = {}
        self.findings = []


# Singleton factory
def get_memory(session_state) -> ConversationMemory:
    """Get or create conversation memory from session state."""
    if 'conversation_memory' not in session_state:
        session_state.conversation_memory = ConversationMemory()
    return session_state.conversation_memory
