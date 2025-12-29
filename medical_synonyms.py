# Medical Query Expansion Module
# Expands queries with medical synonyms and abbreviations for better retrieval

from typing import List, Dict, Tuple
import re

class MedicalQueryExpander:
    """
    Expands user queries with medical synonyms and abbreviations.
    Improves SQL and RAG retrieval by matching various ways terms are expressed.
    """
    
    # Medical term synonyms and abbreviations
    SYNONYMS = {
        # Kidney disease terms
        "ckd": ["chronic kidney disease", "kidney failure", "renal failure", "renal disease"],
        "chronic kidney disease": ["ckd", "kidney disease", "renal insufficiency"],
        "kidney": ["renal", "nephro"],
        "renal": ["kidney"],
        "esrd": ["end stage renal disease", "end-stage kidney disease", "kidney failure"],
        "aki": ["acute kidney injury", "acute renal failure"],
        "dialysis": ["hemodialysis", "peritoneal dialysis", "renal replacement therapy"],
        
        # Lab values
        "creatinine": ["creat", "sc", "serum creatinine", "scr"],
        "gfr": ["glomerular filtration rate", "egfr", "estimated gfr"],
        "egfr": ["gfr", "estimated glomerular filtration rate"],
        "bun": ["blood urea nitrogen", "urea"],
        "albumin": ["alb", "serum albumin"],
        "hemoglobin": ["hgb", "hb", "haemoglobin"],
        "potassium": ["k+", "serum potassium"],
        "sodium": ["na+", "serum sodium"],
        "phosphorus": ["phosphate", "phos"],
        "calcium": ["ca2+", "serum calcium"],
        
        # Conditions
        "diabetes": ["dm", "diabetes mellitus", "diabetic", "t2dm", "type 2 diabetes"],
        "hypertension": ["htn", "high blood pressure", "elevated bp", "hypertensive"],
        "proteinuria": ["protein in urine", "albuminuria", "urine protein"],
        "anemia": ["low hemoglobin", "low hgb", "anaemia"],
        "edema": ["swelling", "fluid retention", "oedema"],
        
        # Symptoms
        "fatigue": ["tiredness", "weakness", "lethargy"],
        "nausea": ["vomiting", "nauseous"],
        "itching": ["pruritus", "uremic pruritus"],
        
        # Medications
        "ace inhibitor": ["acei", "lisinopril", "enalapril", "ramipril"],
        "arb": ["angiotensin receptor blocker", "losartan", "valsartan"],
        "diuretic": ["lasix", "furosemide", "water pill"],
        "statin": ["atorvastatin", "simvastatin", "cholesterol medication"],
        "epo": ["erythropoietin", "epogen", "procrit", "aranesp"],
        
        # Stages
        "stage 1": ["stage i", "early ckd", "mild"],
        "stage 2": ["stage ii"],
        "stage 3": ["stage iii", "stage 3a", "stage 3b", "moderate"],
        "stage 4": ["stage iv", "severe"],
        "stage 5": ["stage v", "kidney failure", "esrd"],
    }
    
    # Common medical abbreviations to expand
    ABBREVIATIONS = {
        "bp": "blood pressure",
        "hr": "heart rate",
        "bmi": "body mass index",
        "htn": "hypertension",
        "dm": "diabetes mellitus",
        "cad": "coronary artery disease",
        "chf": "congestive heart failure",
        "copd": "chronic obstructive pulmonary disease",
        "uti": "urinary tract infection",
        "wbc": "white blood cell",
        "rbc": "red blood cell",
        "plt": "platelet",
        "hct": "hematocrit",
        "pcv": "packed cell volume",
        "sg": "specific gravity",
    }
    
    def __init__(self):
        # Build reverse lookup
        self._build_reverse_lookup()
    
    def _build_reverse_lookup(self):
        """Build reverse synonym mapping."""
        self.reverse_synonyms = {}
        for term, synonyms in self.SYNONYMS.items():
            for syn in synonyms:
                if syn.lower() not in self.reverse_synonyms:
                    self.reverse_synonyms[syn.lower()] = []
                self.reverse_synonyms[syn.lower()].append(term)
    
    def expand_query(self, query: str) -> Tuple[str, List[str]]:
        """
        Expand a query with medical synonyms.
        
        Returns:
            Tuple of (expanded_query, list_of_expansions)
        """
        query_lower = query.lower()
        expansions = []
        expanded_terms = set()
        
        # Find and expand terms
        for term, synonyms in self.SYNONYMS.items():
            if term in query_lower and term not in expanded_terms:
                expansions.extend(synonyms[:3])  # Limit to top 3
                expanded_terms.add(term)
        
        # Check reverse synonyms (if user used abbreviation)
        for abbrev, full_terms in self.reverse_synonyms.items():
            if abbrev in query_lower and abbrev not in expanded_terms:
                expansions.extend(full_terms[:2])
                expanded_terms.add(abbrev)
        
        # Expand abbreviations
        for abbrev, full in self.ABBREVIATIONS.items():
            pattern = rf"\b{abbrev}\b"
            if re.search(pattern, query_lower):
                expansions.append(full)
        
        # Build expanded query
        if expansions:
            unique_expansions = list(set(expansions))[:10]  # Limit total
            expanded_query = f"{query} (related: {', '.join(unique_expansions)})"
            return expanded_query, unique_expansions
        
        return query, []
    
    def expand_for_sql(self, query: str) -> str:
        """
        Expand query specifically for SQL searches.
        Returns query with OR conditions for synonyms.
        """
        expanded, terms = self.expand_query(query)
        return expanded
    
    def expand_for_rag(self, query: str) -> str:
        """
        Expand query for RAG vector search.
        Adds context about medical terminology.
        """
        expanded, terms = self.expand_query(query)
        
        if terms:
            return f"{query}\n\nMedical context: This query relates to {', '.join(terms[:5])}."
        return query
    
    def get_related_terms(self, term: str) -> List[str]:
        """Get related medical terms for a given term."""
        term_lower = term.lower()
        
        related = []
        
        # Check direct synonyms
        if term_lower in self.SYNONYMS:
            related.extend(self.SYNONYMS[term_lower])
        
        # Check reverse lookup
        if term_lower in self.reverse_synonyms:
            related.extend(self.reverse_synonyms[term_lower])
        
        return list(set(related))


# Singleton instance
_expander = None

def get_expander() -> MedicalQueryExpander:
    """Get singleton expander instance."""
    global _expander
    if _expander is None:
        _expander = MedicalQueryExpander()
    return _expander


def expand_query(query: str) -> Tuple[str, List[str]]:
    """Convenience function for query expansion."""
    return get_expander().expand_query(query)


# Test
if __name__ == "__main__":
    expander = MedicalQueryExpander()
    
    test_queries = [
        "What is my GFR?",
        "How many patients have CKD stage 4?",
        "Is my creatinine normal?",
        "Patients with diabetes and hypertension",
        "Average hemoglobin in the database",
    ]
    
    for q in test_queries:
        expanded, terms = expander.expand_query(q)
        print(f"Original: {q}")
        print(f"Expanded: {expanded}")
        print(f"Terms: {terms}\n")
