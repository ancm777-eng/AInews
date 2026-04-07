import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

print("Listing all available models/agents...")
try:
    models = client.models.list()
    for model in models:
        print(f"- {model.name} : {model.display_name}")
except Exception as e:
    print(f"Error: {e}")
