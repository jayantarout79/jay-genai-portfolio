# 🧠 Pinecone Assistant Demo – Simple RAG System with PDF Upload

This project demonstrates how to create a **Retrieval-Augmented Generation (RAG)** system using the **Pinecone Assistant API** — in just a few lines of Python code.  
You can upload any PDF or document, and the Pinecone Assistant automatically indexes it for semantic retrieval.  
Then, you can ask natural language questions about your data.

---

## 🚀 Features
- Create and manage a **Pinecone Assistant** programmatically.
- Upload documents (PDF, TXT, CSV, etc.) for instant vectorization.
- Query and chat with your data using Pinecone’s retrieval engine.
- No manual embedding or chunking — handled automatically by Pinecone.
- Fully compatible with **OpenAI**, **Gemini**, and other LLMs for response generation.

---

## 🧩 Project Structure

pinecone_Assistant_demo/
│
├── pinecone_Assistant_creation_upload.py     # Creates assistant & uploads the file
├── TechNovareport.pdf                        # Sample company dataset (6 pages)
├── .env.example                              # Example environment variables
└── README.md                                 # Documentation

---

## ⚙️ Setup Instructions

### 1. Clone this repository
```bash
git clone https://github.com/jayantarout79/jay-genai-portfolio.git
cd pinecone_Assistant_demo

2. Create a Python virtual environment

python3 -m venv venv
source venv/bin/activate

3. Install dependencies

pip install pinecone python-dotenv

4. Add your Pinecone API key

Create a .env file in this directory:

PINECONE_API_KEY=your_real_pinecone_api_key

5. Run the assistant creation script
python pinecone_Assistant_creation_upload.py

This script:
	•	Creates a new Pinecone Assistant.
	•	Uploads TechNovareport.pdf with metadata.
	•	Makes it instantly ready for question-answering.

To try asking question to the assistant ; 

https://app.pinecone.io/organizations/-OcJg3XhEGYLLakNIb9a/projects/bd4610bd-b3ba-412f-84d8-0f36b9767c35/assistant/My-pinecone-assistant/

💬 Example Queries

Once uploaded, you can query your assistant:
	•	“What was TechNova’s total revenue for Q3 2024?”
	•	“Which product line saw the largest increase in sales?”
	•	“Summarize the report in 3 bullet points.”

🔗 References
	•	📘 Pinecone Assistant Documentation
	•	🧩 Pinecone Python SDK
	•	🌐 Pinecone Website


🧠 Author

Jayanta Kumar Rout (Jay)
LinkedIn • GitHub
