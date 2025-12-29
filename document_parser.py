# Advanced Document Parser with Semantic Chunking
# Uses BPE-aware tokenization for optimal chunk sizing

import os
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class DocumentChunk:
    """A semantically meaningful chunk of a document."""
    content: str
    metadata: Dict[str, Any]
    token_count: int
    chunk_type: str  # "text", "table", "image_caption", "header"


class SemanticDocumentParser:
    """
    Advanced document parser with:
    - Semantic chunking (respects sentence/paragraph boundaries)
    - BPE-aware token counting for optimal chunk sizing
    - Table and image extraction from PDFs
    - Medical entity recognition
    """
    
    def __init__(self, 
                 max_chunk_tokens: int = 512,
                 overlap_tokens: int = 50,
                 model_name: str = "gpt-3.5-turbo"):
        self.max_chunk_tokens = max_chunk_tokens
        self.overlap_tokens = overlap_tokens
        self._tokenizer = None
        self.model_name = model_name
        
    def _get_tokenizer(self):
        """Lazy load tiktoken BPE tokenizer."""
        if self._tokenizer is None:
            try:
                import tiktoken
                self._tokenizer = tiktoken.encoding_for_model(self.model_name)
            except ImportError:
                # Fallback: approximate token count
                self._tokenizer = None
        return self._tokenizer
    
    def count_tokens(self, text: str) -> int:
        """Count tokens using BPE tokenizer or approximation."""
        tokenizer = self._get_tokenizer()
        if tokenizer:
            return len(tokenizer.encode(text))
        else:
            # Approximation: 1 token ≈ 4 characters for English
            return len(text) // 4
    
    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using a simpler approach."""
        # Simple sentence split on period followed by space and uppercase
        # This avoids problematic variable-width lookbehinds
        
        # First, protect common abbreviations
        protected = text
        abbrevs = ['Dr.', 'Mr.', 'Mrs.', 'Ms.', 'Prof.', 'Jr.', 'Sr.', 'vs.', 'etc.', 'e.g.', 'i.e.', 'et al.']
        for abbr in abbrevs:
            protected = protected.replace(abbr, abbr.replace('.', '<<<DOT>>>'))
        
        # Split on sentence boundaries
        sentences = re.split(r'[.!?]+\s+', protected)
        
        # Restore abbreviations and clean
        sentences = [s.replace('<<<DOT>>>', '.').strip() for s in sentences if s.strip()]
        return sentences
    
    def semantic_chunk(self, text: str, metadata: Dict = None) -> List[DocumentChunk]:
        """
        Split text into semantically meaningful chunks.
        
        Strategy:
        1. Split by paragraphs first
        2. If paragraph too large, split by sentences
        3. Combine small sentences until near max_chunk_tokens
        4. Add overlap for context continuity
        """
        if metadata is None:
            metadata = {}
        
        chunks = []
        
        # Step 1: Split by double newlines (paragraphs)
        paragraphs = re.split(r'\n\n+', text)
        
        current_chunk = ""
        current_tokens = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            para_tokens = self.count_tokens(para)
            
            # If paragraph fits in current chunk
            if current_tokens + para_tokens <= self.max_chunk_tokens:
                current_chunk += ("\n\n" if current_chunk else "") + para
                current_tokens += para_tokens
            
            # Paragraph too large - split by sentences
            elif para_tokens > self.max_chunk_tokens:
                # Save current chunk first
                if current_chunk:
                    chunks.append(DocumentChunk(
                        content=current_chunk,
                        metadata=metadata.copy(),
                        token_count=current_tokens,
                        chunk_type="text"
                    ))
                    current_chunk = ""
                    current_tokens = 0
                
                # Split paragraph by sentences
                sentences = self.split_into_sentences(para)
                for sentence in sentences:
                    sent_tokens = self.count_tokens(sentence)
                    
                    if current_tokens + sent_tokens <= self.max_chunk_tokens:
                        current_chunk += (" " if current_chunk else "") + sentence
                        current_tokens += sent_tokens
                    else:
                        # Save and start new chunk
                        if current_chunk:
                            chunks.append(DocumentChunk(
                                content=current_chunk,
                                metadata=metadata.copy(),
                                token_count=current_tokens,
                                chunk_type="text"
                            ))
                        current_chunk = sentence
                        current_tokens = sent_tokens
            
            # Current chunk full, start new one
            else:
                chunks.append(DocumentChunk(
                    content=current_chunk,
                    metadata=metadata.copy(),
                    token_count=current_tokens,
                    chunk_type="text"
                ))
                current_chunk = para
                current_tokens = para_tokens
        
        # Don't forget the last chunk
        if current_chunk:
            chunks.append(DocumentChunk(
                content=current_chunk,
                metadata=metadata.copy(),
                token_count=current_tokens,
                chunk_type="text"
            ))
        
        # Add overlap for context continuity
        chunks = self._add_overlap(chunks)
        
        return chunks
    
    def _add_overlap(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """Add overlapping context between chunks."""
        if len(chunks) <= 1:
            return chunks
        
        for i in range(1, len(chunks)):
            prev_content = chunks[i-1].content
            prev_sentences = self.split_into_sentences(prev_content)
            
            # Take last 1-2 sentences as overlap
            overlap_sentences = prev_sentences[-2:] if len(prev_sentences) > 1 else prev_sentences[-1:]
            overlap_text = " ".join(overlap_sentences)
            
            # Prepend with marker
            chunks[i].content = f"[...] {overlap_text}\n\n{chunks[i].content}"
            chunks[i].token_count = self.count_tokens(chunks[i].content)
        
        return chunks
    
    def parse_pdf(self, pdf_bytes: bytes) -> List[DocumentChunk]:
        """Parse PDF with text, table, and image extraction."""
        chunks = []
        
        try:
            from pypdf import PdfReader
            from io import BytesIO
            
            reader = PdfReader(BytesIO(pdf_bytes))
            
            full_text = ""
            for page_num, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""
                full_text += f"\n\n[Page {page_num + 1}]\n{page_text}"
            
            # Semantic chunk the extracted text
            chunks = self.semantic_chunk(full_text, metadata={"source": "pdf"})
            
            # Try to extract tables (if tabula available)
            try:
                import tabula
                tables = tabula.read_pdf(BytesIO(pdf_bytes), pages='all', silent=True)
                for i, table in enumerate(tables):
                    if not table.empty:
                        table_text = table.to_markdown()
                        chunks.append(DocumentChunk(
                            content=f"[TABLE {i+1}]\n{table_text}",
                            metadata={"source": "pdf", "type": "table"},
                            token_count=self.count_tokens(table_text),
                            chunk_type="table"
                        ))
            except ImportError:
                pass  # tabula not installed
            except Exception:
                pass  # Table extraction failed
                
        except Exception as e:
            # Fallback: just return error info
            chunks.append(DocumentChunk(
                content=f"PDF parsing error: {e}",
                metadata={"source": "pdf", "error": str(e)},
                token_count=10,
                chunk_type="text"
            ))
        
        return chunks
    
    def parse_image(self, image_bytes: bytes) -> List[DocumentChunk]:
        """Extract text from image using OCR."""
        chunks = []
        
        try:
            from PIL import Image
            from io import BytesIO
            
            # Try pytesseract for OCR
            try:
                import pytesseract
                image = Image.open(BytesIO(image_bytes))
                ocr_text = pytesseract.image_to_string(image)
                
                if ocr_text.strip():
                    chunks = self.semantic_chunk(ocr_text, metadata={"source": "image_ocr"})
            except ImportError:
                # No pytesseract - just note that we have an image
                chunks.append(DocumentChunk(
                    content="[Image uploaded - OCR not available]",
                    metadata={"source": "image"},
                    token_count=10,
                    chunk_type="image_caption"
                ))
                
        except Exception as e:
            chunks.append(DocumentChunk(
                content=f"Image parsing error: {e}",
                metadata={"source": "image", "error": str(e)},
                token_count=10,
                chunk_type="text"
            ))
        
        return chunks
    
    def extract_medical_entities(self, text: str) -> Dict[str, Any]:
        """Extract medical entities from text using regex patterns."""
        entities = {}
        
        # Lab values
        patterns = {
            'creatinine': r'(?:creatinine|creat|sc)\s*[:=]?\s*(\d+\.?\d*)\s*(?:mg/dL)?',
            'gfr': r'(?:e?gfr)\s*[:=]?\s*(\d+\.?\d*)',
            'hemoglobin': r'(?:hemo(?:globin)?|hgb|hb)\s*[:=]?\s*(\d+\.?\d*)',
            'albumin': r'(?:albumin|alb)\s*[:=]?\s*(\d+\.?\d*)',
            'potassium': r'(?:potassium|k\+?)\s*[:=]?\s*(\d+\.?\d*)',
            'sodium': r'(?:sodium|na\+?)\s*[:=]?\s*(\d+\.?\d*)',
            'bun': r'(?:bun|blood\s*urea\s*nitrogen)\s*[:=]?\s*(\d+\.?\d*)',
            'bp_systolic': r'(?:bp|blood\s*pressure)\s*[:=]?\s*(\d+)\s*/\s*\d+',
            'bp_diastolic': r'(?:bp|blood\s*pressure)\s*[:=]?\s*\d+\s*/\s*(\d+)',
        }
        
        for entity_name, pattern in patterns.items():
            match = re.search(pattern, text.lower())
            if match:
                try:
                    entities[entity_name] = float(match.group(1))
                except ValueError:
                    pass
        
        return entities


# Quick test
if __name__ == "__main__":
    parser = SemanticDocumentParser(max_chunk_tokens=256)
    
    test_text = """
    Patient is a 69-year-old male with a history of hypertension and diabetes mellitus type 2.
    
    Lab Results:
    Creatinine: 2.4 mg/dL (elevated)
    eGFR: 28 mL/min/1.73m²
    Hemoglobin: 10.2 g/dL
    Albumin: 3.1 g/dL
    
    The patient presents with symptoms consistent with Stage 4 CKD. Recommend nephrology consultation
    and initiation of pre-dialysis education. Consider EPO therapy for anemia of CKD.
    """
    
    chunks = parser.semantic_chunk(test_text, metadata={"patient_id": "test"})
    print(f"Created {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks):
        print(f"\n--- Chunk {i+1} ({chunk.token_count} tokens) ---")
        print(chunk.content[:200] + "...")
    
    entities = parser.extract_medical_entities(test_text)
    print(f"\nExtracted entities: {entities}")
