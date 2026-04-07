import os
import anthropic
from dotenv import load_dotenv

def test_claude_minimal():
    load_dotenv()
    api_key = os.getenv("CLAUDE_API_KEY")
    
    if not api_key:
        print("Error: CLAUDE_API_KEY not found in .env")
        return

    print("--- Claude API Minimal Connection Test ---")
    client = anthropic.Anthropic(api_key=api_key)
    
    try:
        # Extremely small prompt to minimize token cost (1-2 tokens)
        print("Sending 1-word request to Claude...")
        message = client.messages.create(
            model="claude-3-5-sonnet-latest",
            max_tokens=10,
            messages=[
                {"role": "user", "content": "Hi"}
            ]
        )
        print(f"Success! Claude responded: {message.content[0].text}")
    except Exception as e:
        print(f"API Connection Failed: {e}")

if __name__ == "__main__":
    test_claude_minimal()
