import os
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.vectorstores import FAISS
import json

# Load environment variables
load_dotenv()

class PDFAnalyzer:
    def __init__(self):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not found in .env file")
        
        # Initialize embeddings for semantic chunking (using HF instead of Google)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        self.text_splitter = SemanticChunker(self.embeddings)
        
        # Initialize LLM for entity extraction
        # Using Gemini Flash 2.0 (via OpenRouter) for superior context handling and reliable JSON
        self.llm = ChatOpenAI(
            model="google/gemini-2.0-flash-exp:free", 
            temperature=0, 
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key
        )

    def extract_text_from_pdf(self, pdf_path):
        """Extracts text from each page of the PDF."""
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text

    def extract_text_from_bytes(self, pdf_bytes):
        """Extracts text from PDF bytes stream."""
        from io import BytesIO
        reader = PdfReader(BytesIO(pdf_bytes))
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text

    def get_semantic_chunks(self, text):
        """Splits text into semantically meaningful chunks."""
        chunks = self.text_splitter.create_documents([text])
        return [chunk.page_content for chunk in chunks]

    def create_report_index(self, text, index_path="vector_store/temp_report_index"):
        """
        Chunks the text and builds a temporary FAISS index for RAG.
        Returns the vector store object.
        """
        chunks = self.text_splitter.create_documents([text])
        if not chunks:
            return None
            
        vector_store = FAISS.from_documents(chunks, self.embeddings)
        vector_store.save_local(index_path)
        return vector_store

    def extract_clinical_entities(self, text):
        """Uses LLM to extract clinical parameters from the text."""
        prompt = ChatPromptTemplate.from_template("""
        You are a medical data extraction agent. Extract clinical parameters and patient metadata from the medical report text.
        
        CRITICAL INSTRUCTIONS:
        1. Look for Header Info like "Customer Name : Mr. X", "Age/Gender : 69/Male", "Date : 10/Dec/2025".
        2. Handle table rows where values might be separated by whitespace (e.g. "Creatinine 1.2 0.5-1.5").
        3. Convert units if necessary to standard formats (mg/dL for Creatinine).
        4. If a value is NOT found, return null.

        Expected Parameters (JSON Keys must match EXACTLY):
        - name (Patient Name - extract full name found, remove titles like Mr./Mrs. if possible)
        - gender (Male/Female)
        - report_date (Date of report)
        - report_id (Lab ID or Order ID)
        - sample_type (Sample type if mentioned)
        - age (Age in years, integer)
        - bp (Diastolic Blood Pressure - integer, e.g., 80. If given as 120/80, take 80)
        - sg (Specific Gravity: 1.005-1.030)
        - al (Albumin: 0-5 scale. If text says 'Trace', 'Nil', map to 0. If '+', '++', map to 1, 2)
        - su (Sugar: 0-5 scale. 'Nil'->0, 'Trace'->0, '+'->1)
        - rbc (Red Blood Cells: 'normal' or 'abnormal')
        - pc (Pus Cell: 'normal' or 'abnormal')
        - pcc (Pus Cell Clumps: 'present' or 'notpresent')
        - ba (Bacteria: 'present' or 'notpresent')
        - bgr (Blood Glucose Random)
        - bu (Blood Urea)
        - sc (Serum Creatinine)
        - sod (Sodium)
        - pot (Potassium)
        - hemo (Hemoglobin)
        - pcv (Packed Cell Volume)
        - wbcc (White Blood Cell Count)
        - rbcc (Red Blood Cell Count)
        - htn (Hypertension: 'yes' or 'no')
        - dm (Diabetes Mellitus: 'yes' or 'no')
        - cad (Coronary Artery Disease: 'yes' or 'no')
        - appet (Appetite: 'good' or 'poor')
        - pe (Pedal Edema: 'yes' or 'no')
        - ane (Anemia: 'yes' or 'no')
        - grf (GFR / eGFR)
        
        Text to Analyze:
        {text}

        Return the result ONLY as a JSON object.
        """)

        chain = prompt | self.llm | JsonOutputParser()
        try:
            result = chain.invoke({"text": text})
            return result
        except Exception as e:
            print(f"Extraction Error: {e}")
            return {}

if __name__ == "__main__":
    # Quick test if needed
    pass
