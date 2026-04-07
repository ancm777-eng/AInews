import os
from google import genai
from dotenv import load_dotenv

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

print("Listing models...")
try:
    for m in client.models.list():
        print(f"Name: {m.name} | Display: {m.display_name}")
        # print(f"  Methods: {m.supported_methods}") # Just in case it's different
except Exception as e:
    print(f"Error: {e}")
