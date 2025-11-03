from pinecone import Pinecone
from os import getenv
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Get the API key securely
api_key = os.getenv("PINECONE_API_KEY")

# Initialize Pinecone client
pc = Pinecone(api_key=api_key)

# Create an assistant.
assistant = pc.assistant.create_assistant(
    assistant_name="My-pinecone-assistant", 
    instructions="Use American English for spelling and grammar.", # Description or directive for the assistant to apply to all responses.
    region="us", # Region to deploy assistant. Options: "us" (default) or "eu".    
    timeout=30 # Maximum seconds to wait for assistant status to become "Ready" before timing out.
)

# Upload a file to your assistant.
response = assistant.upload_file(
    file_path="/Users/jkr/Desktop/FutureGoal/pythonprojects/jay-genai-portfolio/pinecone_Assistant_demo/TechNovareport.pdf",
    metadata={"company": "TechNova", "document_type": "sales"},
    timeout=None
)
