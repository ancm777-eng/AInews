import os
from google import genai
from dotenv import load_dotenv

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

print("Listing models...")
try:
    for m in client.models.list():
        print(f"Model: {m.name}")
        print(f"  DisplayName: {m.display_name}")
        print(f"  Methods: {m.supported_generation_methods}")
except Exception as e:
    print(f"Error: {e}")
