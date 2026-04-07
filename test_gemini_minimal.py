import os
from google import genai
from dotenv import load_dotenv
from main import get_latest_pro_model

def test_gemini_discovery_and_conn():
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        print("Error: GEMINI_API_KEY not found in env")
        return

    client = genai.Client(api_key=api_key, http_options={'timeout': 60.0})
    
    print("--- Gemini Model Discovery Test ---")
    # 1. Test Agent Discovery (Phase 1 logic)
    research_agent = get_latest_pro_model(client, require_agent=True)
    print(f"Discovered Research Agent: {research_agent}")
    
    # 2. Test Standard Model Discovery (Phase 5 logic)
    gen_model = get_latest_pro_model(client, require_agent=False)
    print(f"Discovered Generative Model: {gen_model}")

    print("\n--- Gemini Connection Test (Using Discovered Model) ---")
    try:
        print(f"Sending minimal request to {gen_model}...")
        response = client.models.generate_content(
            model=gen_model, 
            contents="hi"
        )
        print(f"Success! Gemini responded: {response.text}")
    except Exception as e:
        print(f"API Connection Failed: {e}")

if __name__ == "__main__":
    test_gemini_discovery_and_conn()
