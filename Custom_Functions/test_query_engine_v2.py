# test_query_engine_v2.py
import pytest
from Custom_Functions.faiss_query_engine import FaissQueryEngine

#faiss_qe = FaissQueryEngine("/Users/jkr/Desktop/FutureGoal/pythonprojects/jay-genai-portfolio/Simple-Python/students_performance.csv", embedded_columns=["Name", "Subject", "Remarks"])
faiss_qe = FaissQueryEngine("/Users/jkr/Desktop/FutureGoal/pythonprojects/jay-genai-portfolio/Simple-Python/SampleSuperstore.csv", embedded_columns=["City", "ProductName"])

def test_preview_not_empty():
    preview_df = faiss_qe.preview()
    assert not preview_df.empty, "Preview should return non-empty DataFrame"

def test_semantic_search_result():
    query = "furniture sold in New York"
    result_df = faiss_qe.semantic_search(query)
    assert not result_df.empty, "Semantic search should return results"