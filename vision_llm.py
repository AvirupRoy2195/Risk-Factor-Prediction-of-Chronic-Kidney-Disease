# Vision LLM - GPT-4V Integration via OpenRouter
# Provides multimodal reasoning for medical images

import os
import base64
from dotenv import load_dotenv

load_dotenv()

class VisionLLM:
    """GPT-4V integration for multimodal medical reasoning."""
    
    def __init__(self):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.base_url = "https://openrouter.ai/api/v1"
        # Using free/low-cost vision models on OpenRouter
        self.model = "nvidia/nemotron-nano-12b-v2-vl:free"  # NVIDIA Nemotron Nano 2 VL - OCR & document intelligence
    
    def analyze_medical_image(self, image_base64: str, query: str, patient_context: str = "") -> str:
        """
        Analyze a medical image using direct API call.
        
        Args:
            image_base64: Base64-encoded image string
            query: User's question about the image
            patient_context: Optional patient data for context
        
        Returns:
            Clinical interpretation as markdown string
        """
        import requests
        import base64
        import imghdr
        
        # Detect image format
        try:
            image_data = base64.b64decode(image_base64)
            img_format = imghdr.what(None, h=image_data) or "png"
            mime_type = f"image/{img_format}"
        except:
            mime_type = "image/png"
        
        system_prompt = """You are Dr. VisionAI, a senior radiologist and nephrologist with expertise in:
- Kidney imaging (X-ray, Ultrasound, CT)
- Lab report interpretation
- Pattern recognition for CKD markers

When analyzing medical images:
1. Describe what you observe objectively
2. Note any abnormalities relevant to kidney health
3. Correlate with patient data if provided
4. Suggest differential diagnoses
5. Recommend follow-up tests if needed

IMPORTANT: Always state that this is AI-assisted analysis and recommend professional verification."""

        # Build the request payload using OpenAI-compatible format
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Patient Context: {patient_context}\n\nQuery: {query}"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{image_base64}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 1500,
            "temperature": 0.3
        }
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://nephroai.local",
            "X-Title": "NephroAI Medical Assistant"
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                error_detail = response.json() if response.text else {"error": response.status_code}
                return f"⚠️ Vision analysis error: {response.status_code} - {error_detail}. The model may not support image input."
                
        except requests.exceptions.Timeout:
            return "⚠️ Vision analysis timed out. The model may be overloaded."
        except Exception as e:
            return f"⚠️ Vision analysis error: {str(e)}. The model may not support image input."
    
    def extract_lab_values(self, image_base64: str) -> dict:
        """
        Extract lab values from a lab report image using OCR + LLM.
        
        Returns dict with extracted values like:
        {"creatinine": 1.4, "bun": 25, "gfr": 60, ...}
        """
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage, SystemMessage
        import json
        
        llm = ChatOpenAI(
            model=self.model,
            openai_api_key=self.api_key,
            openai_api_base=self.base_url,
            temperature=0.1,
            max_tokens=1000
        )
        
        extraction_prompt = """You are a medical data extraction specialist. 
Extract all lab values from this image and return them as JSON.

Expected format:
{
    "creatinine": <number or null>,
    "bun": <number or null>,
    "gfr": <number or null>,
    "hemoglobin": <number or null>,
    "albumin": <number or null>,
    "potassium": <number or null>,
    "sodium": <number or null>,
    "glucose": <number or null>,
    "other_findings": "<any other relevant text>"
}

If a value is not clearly visible, use null. Return ONLY valid JSON."""

        content = [
            {"type": "text", "text": extraction_prompt},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
            }
        ]
        
        try:
            response = llm.invoke([HumanMessage(content=content)])
            # Parse JSON from response
            json_str = response.content.strip()
            if json_str.startswith("```"):
                json_str = json_str.split("```")[1].replace("json", "").strip()
            return json.loads(json_str)
        except Exception as e:
            return {"error": str(e), "raw_response": response.content if 'response' in dir() else None}


# Quick test
if __name__ == "__main__":
    vision = VisionLLM()
    print(f"VisionLLM ready. Using model: {vision.model}")
