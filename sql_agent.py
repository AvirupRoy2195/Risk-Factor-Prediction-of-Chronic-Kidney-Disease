# Text2SQL Agent for CKD Dataset Queries
# Converts natural language questions to SQL and executes against patient data

import os
import sqlite3
import pandas as pd
from typing import Tuple, Optional
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import ChatPromptTemplate

class SQLAgent:
    """
    Text2SQL agent for querying CKD datasets.
    Loads CSVs into SQLite and uses LLM to convert natural language to SQL.
    """
    
    def __init__(self, db_path: str = "ckd_data.db"):
        self.db_path = db_path
        self.db = None
        self.llm = ChatOpenAI(
            model="nvidia/nemotron-3-nano-30b-a3b:free",
            temperature=0,
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY")
        )
        self._initialize_db()
    
    def _initialize_db(self):
        """Load CSVs into SQLite database."""
        conn = sqlite3.connect(self.db_path)
        
        # Load available datasets
        datasets = {
            'ckd_patients': 'kaggle_ckd.csv',      # CKD patient data
            'esrd_records': 'kaggle_esrd.csv',     # End-stage renal disease  
            'health_data': 'kaggle_health.csv'    # General health data
        }
        
        for table_name, csv_file in datasets.items():
            if os.path.exists(csv_file):
                try:
                    df = pd.read_csv(csv_file)
                    df.to_sql(table_name, conn, if_exists='replace', index=False)
                    print(f"✓ Loaded {table_name}: {len(df)} rows")
                except Exception as e:
                    print(f"✗ Error loading {csv_file}: {e}")
        
        conn.close()
        
        # Initialize LangChain SQLDatabase
        self.db = SQLDatabase.from_uri(f"sqlite:///{self.db_path}")
    
    def get_schema(self) -> str:
        """Get database schema for context."""
        return self.db.get_table_info()
    
    def query(self, question: str) -> Tuple[str, str, Optional[pd.DataFrame]]:
        """
        Convert natural language to SQL and execute.
        
        Returns:
            Tuple of (sql_query, natural_language_answer, result_dataframe)
        """
        # Create SQL generation prompt with medical context
        sql_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a medical data analyst expert in SQL.
Convert the user's question to a SQL query for a CKD (Chronic Kidney Disease) database.

Database Schema:
{schema}

IMPORTANT RULES:
1. Only use SELECT queries (no INSERT, UPDATE, DELETE)
2. Limit results to 20 rows unless specifically asked for more
3. Use column aliases for readability
4. For aggregations, always include GROUP BY
5. Handle NULL values appropriately

Return ONLY the SQL query, nothing else."""),
            ("human", "{question}")
        ])
        
        try:
            # Expand query with medical synonyms
            from medical_synonyms import get_expander
            expander = get_expander()
            expanded_question, related_terms = expander.expand_query(question)
            
            # Generate SQL with expanded context
            chain = sql_prompt | self.llm
            schema = self.get_schema()
            sql_response = chain.invoke({"schema": schema, "question": expanded_question})
            sql_query = sql_response.content.strip()
            
            # Clean up the query (remove markdown formatting if present)
            if sql_query.startswith("```"):
                sql_query = sql_query.split("```")[1]
                if sql_query.startswith("sql"):
                    sql_query = sql_query[3:]
            sql_query = sql_query.strip()
            
            # Execute query
            conn = sqlite3.connect(self.db_path)
            df = pd.read_sql_query(sql_query, conn)
            conn.close()
            
            # Generate natural language answer
            answer_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a medical data analyst. 
Given the SQL query results, provide a clear, concise answer to the user's question.
Format numbers appropriately and highlight key findings.
If the results are empty, explain what that means clinically."""),
                ("human", """Question: {question}
                
SQL Query: {sql}

Results (first 10 rows):
{results}

Total rows: {total_rows}

Provide a natural language answer:""")
            ])
            
            answer_chain = answer_prompt | self.llm
            answer = answer_chain.invoke({
                "question": question,
                "sql": sql_query,
                "results": df.head(10).to_markdown() if not df.empty else "No results",
                "total_rows": len(df)
            }).content
            
            return sql_query, answer, df
            
        except Exception as e:
            return "", f"❌ Query failed: {str(e)}", None
    
    def get_table_stats(self) -> dict:
        """Get summary statistics for all tables."""
        stats = {}
        conn = sqlite3.connect(self.db_path)
        
        for table in self.db.get_usable_table_names():
            cursor = conn.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            stats[table] = count
        
        conn.close()
        return stats


# Example usage
if __name__ == "__main__":
    agent = SQLAgent()
    print("\nDatabase Schema:")
    print(agent.get_schema()[:1000] + "...")
    
    print("\nTable Stats:")
    print(agent.get_table_stats())
    
    # Test query
    question = "How many patients have CKD stage 5?"
    sql, answer, df = agent.query(question)
    print(f"\nQuestion: {question}")
    print(f"SQL: {sql}")
    print(f"Answer: {answer}")
