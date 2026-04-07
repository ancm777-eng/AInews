import os
from google import genai
from dotenv import load_dotenv

def test_gemini_minimal():
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        print("Error: GEMINI_API_KEY not found in env")
        return

    print("--- Gemini API Minimal Connection Test ---")
    # Generative AI client (not Interactions) for cheap/fast test
    client = genai.Client(api_key=api_key)
    
    try:
        print("Sending 1-word request to Gemini...")
        # Use a standard model for minimal cost
        response = client.models.generate_content(
            model="gemini-2.0-flash", 
            contents="hi"
        )
        print(f"Success! Gemini responded: {response.text}")
    except Exception as e:
        print(f"API Connection Failed: {e}")

if __name__ == "__main__":
    test_gemini_minimal()
