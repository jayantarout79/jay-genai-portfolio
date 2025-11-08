import os
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load the API key from .env
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("Missing GEMINI_API_KEY in .env file")

client = genai.Client(api_key=api_key)

# Create a File Search Store
store = client.file_search_stores.create()

# Upload your file
upload_name = client.file_search_stores.upload_to_file_search_store(
    file_search_store_name=store.name,
    file='TechNovaSalesreport.pdf'
)

# Wait for upload to complete
while True:
    op = client.operations.get(upload_name)
    if op.done:
        break
    time.sleep(5)

# Use the file search store in a generation call
response = client.models.generate_content(
    model='gemini-2.5-flash',
    contents='What are the total sales by region?',
    config=types.GenerateContentConfig(
        tools=[types.Tool(
            file_search=types.FileSearch(
                file_search_store_names=[store.name]
            )
        )]
    )
)

print(response.text)

# Print grounding sources
grounding = response.candidates[0].grounding_metadata
sources = {c.retrieved_context.title for c in grounding.grounding_chunks}
print('Sources:', *sources)