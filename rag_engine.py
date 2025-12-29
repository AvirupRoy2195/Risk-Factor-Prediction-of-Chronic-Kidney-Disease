import os
import pandas as pd
from datasets import load_dataset
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough, RunnableBranch
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from tqdm import tqdm

class RAGEngine:
    def __init__(self, index_path="vector_store/faiss_index"):
        self.index_path = index_path
        # Use BGE embeddings as requested
        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        self.vector_store = None
        # Initialize OpenRouter GPT-5.2 for Chat
        self.llm = ChatOpenAI(
            model="nvidia/nemotron-3-nano-30b-a3b:free", 
            temperature=0.3,
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY")
        )

    def build_index(self, max_samples=2000):
        """Downloads a subset of the dataset and indexes kidney-related clinical reasoning."""
        print(f"Loading OpenMed/Medical-Reasoning-SFT-GPT-OSS-120B dataset (subset of {max_samples})...")
        
        # Load small subset for indexing
        ds = load_dataset("OpenMed/Medical-Reasoning-SFT-GPT-OSS-120B", split=f"train[:{max_samples}]")
        
        documents = []
        print("Filtering and preparing documents...")
        for item in tqdm(ds):
            # item is a list of messages [user, assistant]
            # We want to index the assistant's clinical reasoning (which contains the <think> tag or structured logic)
            user_msg = ""
            assistant_msg = ""
            
            for msg in item['messages']:
                if msg['role'] == 'user':
                    user_msg = msg['content']
                elif msg['role'] == 'assistant':
                    assistant_msg = msg['content']
            
            # Combine for context
            full_context = f"Patient Scenario: {user_msg}\n\nClinical Reasoning: {assistant_msg}"
            
            # Filter for kidney related content to keep the index relevant and efficient
            keywords = ['kidney', 'renal', 'nephro', 'creatinine', 'albumin', 'gfr', 'ckd', 'urine']
            if any(key in full_context.lower() for key in keywords):
                doc = Document(
                    page_content=full_context,
                    metadata={"source": "OpenMed-Reasoning", "type": "clinical_reasoning"}
                )
                documents.append(doc)
        
        print(f"Indexing {len(documents)} relevant documents...")
        if documents:
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
            os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
            self.vector_store.save_local(self.index_path)
            print(f"Index saved to {self.index_path}")
        else:
            print("No relevant documents found in the sampled subset.")

    def load_index(self):
        """Loads the FAISS index from local storage."""
        if os.path.exists(self.index_path):
            try:
                self.vector_store = FAISS.load_local(self.index_path, self.embeddings, allow_dangerous_deserialization=True)
                return True
            except Exception as e:
                print(f"Error loading index: {e}")
                return False
        return False

    def query_reasoning(self, query, k=3):
        """Retrieves top k clinical reasoning traces for a given query."""
        if not self.vector_store:
             # Try reloading or rebuilding
             if not self.load_index():
                 self.build_index()
        
        results = self.vector_store.similarity_search(query, k=k)
        return results

    def chat_reasoning(self, query, chat_history, report_vector_store=None):
        """
        Conversational RAG using Gemini/OpenAI.
        If report_vector_store is provided, it retrieves context from BOTH the medical knowledge base
        and the patient's specific uploaded report.
        """
        if not self.vector_store:
             if not self.load_index():
                 self.build_index()

        retriever_kb = self.vector_store.as_retriever(search_kwargs={"k": 3})
        
        # --- 1. Query Transformation (History Aware) ---
        contextualize_q_system_prompt = """Given a chat history and the latest user question 
        which might reference context in the chat history, formulate a standalone question 
        which can be understood without the chat history. Do NOT answer the question, 
        just reformulate it if needed or otherwise return it as is."""
        
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        
        contextualize_q_chain = contextualize_q_prompt | self.llm | StrOutputParser()

        transform_query = RunnableBranch(
            (lambda x: bool(x.get("chat_history")), contextualize_q_chain),
            (lambda x: x["input"])
        )

        # --- 2. Dual Retrieval Logic ---
        def retrieve_combined(query):
            # 1. Knowledge Base Docs
            docs = retriever_kb.invoke(query)
            
            # 2. Report Docs (if available)
            if report_vector_store:
                report_retriever = report_vector_store.as_retriever(search_kwargs={"k": 3})
                report_docs = report_retriever.invoke(query)
                # Label them clearly
                for d in report_docs:
                    d.page_content = f"[PATIENT REPORT]: {d.page_content}"
                docs.extend(report_docs)
            
            return docs

        # --- 3. Retrieval & Answer Generation ---
        qa_system_prompt = """You are an expert Nephrology AI Assistant. 
        Use the following pieces of retrieved medical context (Knowledge Base + Patient Report) to answer the question. 
        If specific patient info differs from general guidelines, prioritize the patient's report.
        If you don't know the answer, say that you don't know. 
        
        Context: {context}"""
        
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", qa_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        rag_chain = (
            RunnablePassthrough.assign(
                # Transform query -> Retrieve Combined -> Format -> context
                context=transform_query | retrieve_combined | format_docs
            )
            | qa_prompt
            | self.llm
            | StrOutputParser()
        )
        
        response = rag_chain.invoke({"input": query, "chat_history": chat_history})
        return response

if __name__ == "__main__":
    # Initialize and build index
    engine = RAGEngine()
    if not engine.load_index():
        try:
            engine.build_index()
        except:
            pass # Dataset might require login or internet, skip for import test
    
    # Simple test
    # test_query = "High serum creatinine level and hypertension"
    # hits = engine.query_reasoning(test_query)
    # if not isinstance(hits, str):
    #     for i, hit in enumerate(hits):
    #         print(f"\n--- Hit {i+1} ---\n{hit.page_content[:500]}...")
