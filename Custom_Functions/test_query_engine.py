from query_engine import QueryEngine
qe= QueryEngine("/Users/jkr/Desktop/FutureGoal/pythonprojects/jay-genai-portfolio/Simple-Python/sample_data.csv")

print("\n--- Preview ---")
print(qe.preview())

# Search for a keyword
print("\n--- Search Results for 'Alice' ---")
print(qe.search("Alice"))

print("\n--- Search Results for '92' ---")
print(qe.search("92"))
