from Custom_Functions.faiss_query_engine import FaissQueryEngine
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import os
from dotenv import load_dotenv

# Load OPENAI_API_KEY from .env if available
load_dotenv()

# ---- Step 1: Load or build your FAISS index from CSV ----
faiss_qe = FaissQueryEngine(
    "/Users/jkr/Desktop/FutureGoal/pythonprojects/jay-genai-portfolio/Simple-Python/SampleSuperstore.csv",
    embedded_columns=["City", "ProductName"]
)

# ---- Step 2: Build a tiny RAG chain using the engine's results ----
SYSTEM_PROMPT = """You are a concise assistant.
Use ONLY the provided rows from the CSV to answer the question.
If the answer is not in the rows, say "I don't know".
"""

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "Question: {question}\n\nRows:\n{rows}")
])

def format_rows(df):
    return "\n".join(f"{r.City} - {r.ProductName}" for _, r in df.iterrows())

def answer_with_csv(query: str, k: int = 5):
    # Retrieve top-k relevant rows
    df = faiss_qe.semantic_search(query, k=k)
    if df.empty:
        return "No relevant rows found."
    retriever_output = format_rows(df)

    llm = ChatOpenAI(model="gpt-4o-mini")  # or any other model you have access to
    chain = (
        {"rows": RunnablePassthrough(), "question": RunnablePassthrough()}
        | PROMPT
        | llm
        | StrOutputParser()
    )
    return chain.invoke({"rows": retriever_output, "question": query})

if __name__ == "__main__":
    q = "Most furniture sold in New York"
    print(answer_with_csv(q, k=5))