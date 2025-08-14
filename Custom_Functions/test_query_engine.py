#import pytest
from Custom_Functions.faiss_query_engine import FaissQueryEngine

faiss_qe = FaissQueryEngine("/Users/jkr/Desktop/FutureGoal/pythonprojects/jay-genai-portfolio/Simple-Python/SampleSuperstore.csv", embedded_columns=["City", "ProductName"])

if __name__ == "__main__":
    print("Preview of the dataset:")
    print(faiss_qe.preview().head(5))
    
    query = "furniture sold in New York"
    result_df = faiss_qe.semantic_search(query, k=5)
    print("\nSemantic search results:")
    print(result_df[['City', 'ProductName', 'score']])