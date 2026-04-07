import os
import datetime
from main import validate_with_claude, generate_html
from github_utils import fetch_prompt_from_github
from dotenv import load_dotenv

def test_pipeline():
    load_dotenv()
    
    print("--- Phase 1: Deep Research (Skipped, using mock trial/1.txt) ---")
    with open("trial/1.txt", "r", encoding="utf-8") as f:
        initial_result = f.read()
        
    print(f"Loaded {len(initial_result)} characters from trial/1.txt")
    
    print("\n--- Phase 2: Claude Validation ---")
    feedback = validate_with_claude(initial_result, custom_prompt="검증해주세요")
    if feedback:
        print("Feedback generated successfully.")
        os.makedirs("trial", exist_ok=True)
        with open("trial/feedback.txt", "w", encoding="utf-8") as f:
            f.write(feedback)
            
        refined_result = initial_result + "\n\n(피드백이 반영되어 내용이 개선됨)"
    else:
        print("Claude validation skipped.")
        refined_result = initial_result
        
    print("\n--- Phase 4: Claude Audit 2 & Translation ---")
    translation_prompt = "구조를 유지한 채 최상의 영어 문장으로 최종 수정 및 번역해주세요."
    translated_result = validate_with_claude(refined_result, custom_prompt=translation_prompt)
    
    final_content_for_html = translated_result if translated_result else refined_result
    print(f"Final translated length: {len(final_content_for_html)}")
    
    print("\n--- Phase 5: HTML Generation ---")
    html_prompt_url = os.getenv("HTML_PROMPT_URL")
    if html_prompt_url:
        html_prompt = fetch_prompt_from_github(html_prompt_url)
        if html_prompt:
            generate_html(final_content_for_html, html_prompt, "index.html")
            print("Successfully generated index.html!")
        else:
            print("Failed to fetch HTML prompt.")
    else:
        print("HTML_PROMPT_URL missing.")

if __name__ == "__main__":
    test_pipeline()
