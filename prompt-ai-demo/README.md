# 🚀 Jay's GenAI Prompt Engineering Portfolio

This project demonstrates a clean GenAI prompt-processing pipeline using Python and OpenAI's GPT-4 API. It includes modular utilities to clean user input, apply context, and generate responses from OpenAI.

---

## 📁 Project Structure

jay-genai-portfolio/
├── Custom_Functions/                  # Modular, reusable Python components
│   ├── faiss_query_engine.py           # Semantic search engine using FAISS + OpenAI embeddings
│   ├── query_engine.py                 # Keyword-based CSV query engine
│   ├── test_faiss_query_engine.py      # Test script for FAISSQueryEngine
│
├── SampleSuperstore.csv                # Example dataset for query engine testing
├── llm_query.py                        # Script to query GPT-4 with processed prompts
├── prompt_utils.py                     # Prompt cleaning and context utilities
├── query_engine.py                     # (Optional duplicate / refactored version in root)
├── requirements.txt                    # Dependencies for the project
├── README.md                           # Project documentation
├── .env.example                        # Example env file (no real keys)
└── .env                                # Local environment variables (ignored in Git)

---

## ⚙️ Setup Instructions

### 1. Clone the Repo

```bash
git clone https://github.com/yourusername/jay-genai-portfolio.git
cd jay-genai-portfolio

2. Install Dependencies

pip install -r requirements.txt

3. Setup Your OpenAI API Key

Create a .env file:

OPENAI_API_KEY=your_openai_api_key_here


⸻

🧠 How to Use

➤ Run the script

python llm_query.py

You’ll be prompted to enter a prompt and additional info. The script:
	•	Cleans your input
	•	Adds intelligent context
	•	Sends it to GPT-4
	•	Returns a smart response

⸻

💡 Example

Input: How can I identify churn risk in customer data?

Output: To identify churn risk, consider training a classification model on customer behavior and engagement metrics...


⸻

🔐 Security Notes
	•	✅ .env is used to keep your API key secure
	•	❌ Do not commit .env to GitHub — only .env.example should be public
	•	✅ Compatible with OpenAI SDK v1.x+

⸻

📌 Author

Jayanta Kumar Rout
Lead Data Engineer | Building AI-Augmented Data Systems
🌐 LinkedIn | ✨ Exploring GenAI for real-world use cases
