import os
import requests
import json
from dotenv import load_dotenv

def test_claude_via_requests():
    load_dotenv()
    api_key = os.getenv("CLAUDE_API_KEY")
    
    if not api_key:
        print("Error: CLAUDE_API_KEY not found in .env")
        return

    print("--- Claude API Connection Test (via standard requests) ---")
    url = "https://api.anthropic.com/v1/messages"
    
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    
    payload = {
        "model": "claude-4-6-sonnet-latest",
        "max_tokens": 10,
        "messages": [
            {"role": "user", "content": "Hi"}
        ]
    }
    
    try:
        print("Sending request via HTTP POST...")
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        print(f"HTTP Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Success! Claude responded: {result['content'][0]['text']}")
        else:
            print(f"Failed with response: {response.text}")
    except Exception as e:
        print(f"API Connection Failed: {e}")

if __name__ == "__main__":
    test_claude_via_requests()
