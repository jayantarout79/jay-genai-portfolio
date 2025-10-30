import os
from dotenv import load_dotenv
from pinecone import Pinecone

# Load env vars from .env file (PINECONE_API_KEY, etc.)
load_dotenv()

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index_name = "rag-demo-openai-embed"

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=3072,
        metric="cosine",
        spec={
            "serverless": {
                "cloud": "aws",
                "region": "us-east-1"
            }
        }
    )