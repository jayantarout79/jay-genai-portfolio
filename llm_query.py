from dotenv import load_dotenv
import os
import openai
import prompt_utils # Importing the prompt_utils module for utility functions


# Ensure the environment variable for OpenAI API key is set
load_dotenv()  # Load variables from .env
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY is not set in the environment variables.")

user_prompt=input("Enter your prompt: ")  # Get user input for the prompt
cleaned_prompt = prompt_utils.clean_user_input(user_prompt)  # Clean the user input
additional_info = input("Enter additional context (optional): ")  # Get additional context from the user
prompt = prompt_utils.add_context(cleaned_prompt, additional_info)  # Add context to the prompt

# Set up OpenAI API client
openai.api_key = api_key
from openai import OpenAI
client = OpenAI()

response = client.responses.create(
    model="gpt-4",
    input=prompt,
)

print(response.output_text)
# This code processes a user prompt, cleans it, adds context, and sends it to the OpenAI API for a response.
# Ensure the environment variable for OpenAI API key is set
# The response is printed to the console.