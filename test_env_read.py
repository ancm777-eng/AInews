import os
from dotenv import load_dotenv
from github_utils import fetch_prompt_from_github

# Load .env
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
prompt_url = os.getenv("PROMPT_URL")

print("--- .env Read Test ---")
if api_key:
    print(f"GEMINI_API_KEY: Found (starts with {api_key[:5]}...)")
else:
    print("GEMINI_API_KEY: Not found!")

if prompt_url:
    print(f"PROMPT_URL: {prompt_url}")
    print("\nAttempting to fetch prompt content...")
    content = fetch_prompt_from_github(prompt_url)
    if content:
        print("Success! Content fetched from GitHub.")
        print("First 100 characters of prompt:")
        print("-" * 30)
        print(content[:100])
        print("-" * 30)
    else:
        print("Failed to fetch content from GitHub. Please check the URL.")
else:
    print("PROMPT_URL: Not found!")
