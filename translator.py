# Translation Layer for Multi-Language Support
# Translates any language to English for processing

import os
from typing import Tuple, Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

class Translator:
    """
    Translates text from any language to English.
    Uses LLM for translation with language detection.
    """
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model="nvidia/nemotron-3-nano-30b-a3b:free",
            temperature=0,
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY")
        )
        
        self.translation_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a medical translator. 
1. Detect if the input is in English.
2. If NOT English, translate to English.
3. Preserve medical terminology accurately.

Respond in this format:
LANGUAGE: [detected language]
ENGLISH: [translated text, or original if already English]"""),
            ("human", "{text}")
        ])
    
    def detect_and_translate(self, text: str) -> Tuple[str, str, bool]:
        """
        Detect language and translate to English if needed.
        
        Returns:
            Tuple of (english_text, detected_language, was_translated)
        """
        # Quick check for common English patterns
        if self._is_likely_english(text):
            return text, "English", False
        
        try:
            response = self.llm.invoke(
                self.translation_prompt.format_messages(text=text)
            )
            
            result = response.content
            
            # Parse response
            language = "Unknown"
            english_text = text
            
            if "LANGUAGE:" in result:
                parts = result.split("ENGLISH:")
                language = parts[0].replace("LANGUAGE:", "").strip()
                if len(parts) > 1:
                    english_text = parts[1].strip()
            else:
                english_text = result
            
            was_translated = language.lower() != "english"
            return english_text, language, was_translated
            
        except Exception as e:
            return text, "Unknown", False
    
    def _is_likely_english(self, text: str) -> bool:
        """Quick heuristic check for English text."""
        # Common English words
        english_words = ['the', 'is', 'are', 'what', 'how', 'my', 'patient', 
                        'kidney', 'creatinine', 'blood', 'with', 'have', 'for']
        
        text_lower = text.lower()
        word_count = sum(1 for word in english_words if word in text_lower)
        
        # If 2+ common English words, likely English
        return word_count >= 2
    
    def translate_batch(self, texts: list) -> list:
        """Translate a batch of texts."""
        return [self.detect_and_translate(t)[0] for t in texts]


# Singleton
_translator = None

def get_translator() -> Translator:
    global _translator
    if _translator is None:
        _translator = Translator()
    return _translator


def translate_to_english(text: str) -> Tuple[str, str, bool]:
    """Convenience function."""
    return get_translator().detect_and_translate(text)


# Test
if __name__ == "__main__":
    translator = Translator()
    
    tests = [
        "What is my creatinine level?",  # English
        "मेरी क्रिएटिनिन क्या है?",  # Hindi
        "Was ist mein Kreatinin?",  # German
        "Qual é a minha creatinina?",  # Portuguese
    ]
    
    for text in tests:
        english, lang, translated = translator.detect_and_translate(text)
        print(f"Original: {text}")
        print(f"Language: {lang}, Translated: {translated}")
        print(f"English: {english}\n")
