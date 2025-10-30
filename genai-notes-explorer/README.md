📘 GenAI Notes Explorer

A lightweight RAG (Retrieval-Augmented Generation) pipeline to explore personal notes, ML projects, and PDFs with embeddings + FAISS + LLM answers.

This repo lets you:
	•	🔄 Rebuild an index from Markdown/PDF study notes
	•	❓ Ask questions against your notes with grounded answers + citations
	•	📊 Use embeddings (text-embedding-3-small) + FAISS similarity search
	•	🤖 Generate concise, bullet-style responses from an LLM (OpenAI GPT models)
    
🚀 Quickstart

1. Clone & Install

git clone <this-repo-url>
cd genai-notes-explorer
pip install -r requirements.txt

OPENAI_API_KEY=sk-...
EMBEDDING_MODEL=text-embedding-3-small
CHAT_MODEL=gpt-4o-mini

python3 src/cli.py rebuild \
  --data data \
  --chunks index/chunks.jsonl \
  --out index \
  --size 1000 --overlap 200 \
  --embed_model text-embedding-3-small --batch 128

  This will:
	•	Load docs from data/
	•	Split them into overlapping chunks
	•	Embed with OpenAI
	•	Save FAISS index + metadata into index/

4. Ask Questions

python3 src/cli.py ask \
  --index index \
  --query "What metrics did we optimize and why?" \
  --k 5

  === ANSWER ===
- Precision was prioritized to reduce wasted discounts.
- Recall was tracked but allowed to be lower.
- F1 and ROC-AUC monitored for balance.

=== SOURCES ===
- retail_project_summary.pdf#chunk0
- project_doc.md#chunk0
- random_forest_notes.md#chunk0

📂 Project Structure

├── data/                  # Input notes (md/pdf)
│   ├── glossary.md
│   ├── project_doc.md
│   ├── random_forest_notes.md
│   ├── retail_project_summary.pdf
│   └── week_plan_excerpt.md
│
├── index/                 # Auto-generated index files
│   ├── chunks.jsonl
│   ├── faiss.index
│   ├── meta.jsonl
│   └── model.json
│
├── src/                   # Core RAG pipeline
│   ├── loaders.py         # Load PDFs/Markdown
│   ├── chunker.py         # Split docs into chunks
│   ├── embedder.py        # Embeddings + FAISS index
│   ├── retriever.py       # Similarity search
│   ├── rag_pipeline.py    # Orchestrates retrieval + LLM answer
│   ├── llm_client.py      # OpenAI client abstraction
│   ├── prompts.py         # Prompt templates
│   └── cli.py             # Main CLI (rebuild | ask)
│
├── requirements.txt       # Python dependencies
├── .env                   # API keys (not committed)
├── .gitignore             # Ignore index + secrets
└── README.md              # This file


🧠 Concepts Used
	•	Embeddings – text → vector representation (text-embedding-3-small)
	•	Chunking – long documents split into overlapping pieces
	•	Vector search – FAISS similarity search for top-k chunks
	•	RAG – retrieved context fed into LLM for grounded answers
	•	Citations – inline [source:file#chunk] references

📌 Example Use Cases
	•	Study notes Q&A (ML/AI concepts)
	•	Portfolio project documentation search
	•	Meeting notes → semantic retrieval
	•	Technical papers exploration

🛠️ Requirements
	•	Python 3.9+
	•	OpenAI API key
	•	FAISS library (faiss-cpu)

pip install -r requirements.txt

🔮 Future Enhancements
	•	Streamlit/Gradio UI for interactive Q&A
	•	Support for image + multimodal docs
	•	Improved prompt engineering & summarization

📜 License

MIT License © 2025 Jayanta Kumar Rout




****Testing Commands *******
python3 src/loaders.py data
python3 src/chunker.py data --out index/chunks.jsonl --size 1000 --overlap 200
python3 src/embedder.py build --chunks index/chunks.jsonl --out index/ --model text-embedding-3-small --batch 128
python3 src/embedder.py stats --out index/
python3 src/retriever.py --index index/ \
  --query "What metrics did we optimize in the retail purchase project and why?" --k 5

python3 src/cli.py rebuild \
  --data data --chunks index/chunks.jsonl --out index \
  --size 1000 --overlap 200 --embed_model text-embedding-3-small --batch 128

  python3 src/cli.py ask \
  --index index --k 5 \
  --query "What is the plan for week 4-5?"

 python3 src/cli.py ask \
  --index index --k 5 \
  --query "What is the Account number?"

   python3 src/cli.py ask \
  --index index --k 5 \
  --query "how much has been spent for City of Austin?"

Testing :
python3 src/cli.py ask --index index --k 3 --query "What metrics did we optimize and why?"
python3 src/cli.py ask --index index --k 3 --query "Next steps in Week 4–5 plan?"
python3 src/cli.py ask --index index --k 3 --query "How does Random Forest reduce overfitting?"
python3 src/cli.py ask --index index --k 3 --query "What metrics did we optimize in the retail purchase project and why?"
python3 src/cli.py ask --index index --k 3 --query "how much has been spent for City of Austin?"
python3 src/cli.py ask --index index --k 3 --query "What is the Account number?"