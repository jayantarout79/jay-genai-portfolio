import sys
import os
import pytest


# Add root directory to path to allow absolute imports like 'Custom_Functions.xxx'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from faiss_query_engine import FaissQueryEngine

# Initialize FAISSQueryEngine (embed both Name and Score columns for this example)
faiss_qe = FaissQueryEngine("/Users/jkr/Desktop/FutureGoal/pythonprojects/jay-genai-portfolio/Simple-Python/ students_performance.csv", embedded_columns=["Name", "Subject", "Remarks"])

# Preview Data
print("\n--- Preview ---")
print(faiss_qe.preview())

# Semantic Search
print("\n--- Semantic Search: 'math competition winner' ---")
print(faiss_qe.semantic_search("math competition winner"))

print("\n--- Semantic Search: 'student struggling with lab reports' ---")
print(faiss_qe.semantic_search("student struggling with lab reports"))