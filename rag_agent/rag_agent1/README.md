🧠 Google ADK + Pinecone + OpenAI RAG Agent

This project demonstrates a Retrieval-Augmented Generation (RAG) pipeline built using
Google ADK, Pinecone Vector Database, and OpenAI Embeddings.

It’s a lightweight, retrieval-only chatbot that answers strictly from Pinecone’s stored knowledge base — without fabricating any information.

⸻

🚀 Features
	•	Vector Search with Pinecone → Retrieves relevant context based on embeddings
	•	OpenAI Embeddings → Converts input text into high-dimensional vectors
	•	Google ADK Agent Framework → Handles user prompts, retrieval calls, and responses
	•	Retrieval-Only Logic → The model cannot invent — it must answer only from the stored data
	•	Customizable Knowledge Base → Easily upsert your own domain-specific documents

rag_agent1/
│
├── agent.py               # Core ADK agent connecting to Pinecone + OpenAI
├── Create_index.py        # Creates Pinecone index
├── Upsert.py              # Embeds and upserts docs into Pinecone
├── .env           # Example environment variables (no keys)
├── requirements.txt       # Dependencies
└── README.md              # (this file)

# 1️⃣ Clone repository
git clone https://github.com/jayantarout79/jay-genai-portfolio.git
cd jay-genai-portfolio/rag_agent1

# 2️⃣ Create virtual environment
python -m venv venv
source venv/bin/activate

# 3️⃣ Install dependencies
pip install -r requirements.txt

🔐 Environment Setup

Create a .env file in the project root:

OPENAI_API_KEY=your_openai_key_here
PINECONE_API_KEY=your_pinecone_key_here
PINECONE_INDEX=rag-demo-openai-embed
EMBEDDING_MODEL=text-embedding-3-large

🧩 Create & Populate the Vector Database

python Create_index.py

2️⃣ Insert your data

python Upsert.py

🤖 Run the Agent

adk run rag_agent1
adk web --port 8000

You’ll see an interactive prompt:

[user]: what is supervised learning?
[root_agent]: Supervised learning uses labeled data...

💡 How It Works
	1.	Query Embedding – Converts user query into a vector using OpenAI embeddings.
	2.	Vector Search – Retrieves the most relevant chunks from Pinecone.
	3.	Response Generation – The Gemini model responds using only the retrieved context.

If the context doesn’t exist, it replies:

“I’m sorry, I don’t have information about that in my current knowledge base.”

👨‍💻 Author

Jayanta Kumar Rout (Jay)
	•	🌐 GitHub
	•	🎥 YouTube – The AI Crush
	•	💼 LinkedIn
