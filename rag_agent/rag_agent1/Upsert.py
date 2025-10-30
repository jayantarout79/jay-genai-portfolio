import os
from dotenv import load_dotenv
from pinecone import Pinecone
from openai import OpenAI

# --- load .env file ---
load_dotenv()

# --- config ---
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
INDEX_NAME = "rag-demo-openai-embed"

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

def embed_text(text: str):
    client = OpenAI(api_key=OPENAI_API_KEY)
    resp = client.embeddings.create(
        model="text-embedding-3-large",
        input=text
    )
    return resp.data[0].embedding

docs = [
    {"id": "rec1", "text": "Supervised learning uses labeled data to train models that predict outcomes for unseen inputs.", "category": "machine learning"},
    {"id": "rec2", "text": "Unsupervised learning discovers hidden patterns in unlabeled data through clustering or dimensionality reduction.", "category": "machine learning"},
    {"id": "rec3", "text": "Reinforcement learning trains agents to make decisions by rewarding desired behaviors over time.", "category": "AI control"},
    {"id": "rec4", "text": "A neural network is composed of layers of interconnected nodes that transform inputs into outputs through weighted computations.", "category": "deep learning"},
    {"id": "rec5", "text": "Large Language Models like GPT-4 or Gemini use transformer architectures trained on massive text corpora.", "category": "LLMs"},
    {"id": "rec6", "text": "Vector databases such as Pinecone store high-dimensional embeddings to enable semantic search and retrieval.", "category": "vector database"},
    {"id": "rec7", "text": "RAG combines retrieval-based context from databases with generative AI to produce grounded, accurate responses.", "category": "AI architecture"},
    {"id": "rec8", "text": "Fine-tuning allows models to specialize on specific domains by training further on curated datasets.", "category": "model tuning"},
]

# build vectors for all docs
vectors_to_upsert = []
for d in docs:
    vec = embed_text(d["text"])  # 3072-dim list[float]
    vectors_to_upsert.append((
        d["id"],          # id
        vec,              # the actual embedding vector
        {                 # metadata, totally your choice
            "text": d["text"],
            "category": d["category"]
        }
    ))

# now upsert the vectors
index.upsert(
    vectors=vectors_to_upsert,
    namespace="__default__"
)

print("✅ Upsert complete.")