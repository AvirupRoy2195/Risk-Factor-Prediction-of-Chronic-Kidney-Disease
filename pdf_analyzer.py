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
        self.llm = ChatOpenAI(
            model="nvidia/nemotron-3-nano-30b-a3b:free", 
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
        You are a medical data extraction agent. Extract the following clinical parameters from the medical report text provided.
        If a value is not found, return null. 
        Ensure specific gravity (sg) is one of [1.005, 1.010, 1.015, 1.020, 1.025].
        Ensure binary values (rbc, pc, pcc, ba, htn, dm, cad, appet, pe, ane) are mapped to the expected categories.
        
        Expected Parameters (JSON Keys must match EXACTLY):
        - name (Patient Name - extract full name found)
        - gender (Male/Female)
        - report_date (Date of report)
        - report_id (Lab ID or Order ID)
        - sample_type (Sample type if mentioned)
        - age (Age in years)
        - bp (Diastolic Blood Pressure - integer, e.g., 80)
        - sg (Specific Gravity: 1.005, 1.010, 1.015, 1.020, 1.025)
        - al (Albumin: 0, 1, 2, 3, 4, 5)
        - su (Sugar: 0, 1, 2, 3, 4, 5)
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
        
        Text:
        {text}

        Return the result ONLY as a JSON object. Ensure values like 'normal' are lowercase.
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
