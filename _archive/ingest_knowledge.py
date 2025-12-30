import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

def ingest():
    pdf_path = "Kidney-Beginnings.pdf"
    index_path = "vector_store/faiss_index"

    if not os.path.exists(pdf_path):
        print(f"❌ Error: {pdf_path} not found. Please download it first.")
        return

    print(f"📖 Reading {pdf_path}...")
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
        print(f"  - Extracted {len(text)} characters from {len(reader.pages)} pages.")
    except Exception as e:
        print(f"❌ Failed to read PDF: {e}")
        return

    # Chunking
    print("✂️ Chunking text...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    docs = splitter.create_documents([text], metadatas=[{"source": "Kidney-Beginnings-Book", "type": "knowledge_base"} for _ in range(1)]) # metadata needs to match list length? No, create_documents takes list of texts.
    # Actually create_documents needs metadatas list same length as texts list if passed. 
    # Better: split_text then create document objects.
    
    docs = splitter.create_documents([text])
    for doc in docs:
        doc.metadata = {"source": "Kidney-Beginnings-Book", "type": "knowledge_base"}
        
    print(f"  - Created {len(docs)} knowledge chunks.")

    # Embeddings
    print("🧠 Initializing Embeddings (BAAI/bge-small-en-v1.5)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    # Load existing index or create new
    if os.path.exists(index_path) and os.path.exists(os.path.join(index_path, "index.faiss")):
        print("📂 Loading existing RAG index...")
        try:
            vector_store = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
            print("➕ Adding new documents to index...")
            vector_store.add_documents(docs)
        except Exception as e:
            print(f"⚠️ Could not load existing index: {e}. Creating new one.")
            vector_store = FAISS.from_documents(docs, embeddings)
    else:
        print("✨ Creating NEW RAG index...")
        vector_store = FAISS.from_documents(docs, embeddings)

    # Save
    print(f"💾 Saving updated index to {index_path}...")
    vector_store.save_local(index_path)
    print("✅ Knowledge Base Successfully Updated with Kidney-Beginnings Book!")

if __name__ == "__main__":
    ingest()
