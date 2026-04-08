import os
import time
import argparse
import re
from dotenv import load_dotenv
from google import genai
from google.genai import types
import anthropic
from tenacity import retry, stop_after_attempt, wait_exponential
import datetime
from github_utils import fetch_prompt_from_github

# Load environment variables
load_dotenv()

def return_none_on_error(retry_state):
    """Callback for tenacity when all retry attempts fail."""
    print(f"\nAction ultimately failed after {retry_state.attempt_number} attempts: {retry_state.outcome.exception()}")
    return None

def get_recent_archives(days=7):
    """
    Reads news content from the data/ directory for the last N days.
    """
    archive_data = ""
    archive_dir = "data"
    if not os.path.exists(archive_dir):
        return ""
    
    try:
        # Get list of files and filter by date-like names (YYYY-MM-DD.txt)
        files = [f for f in os.listdir(archive_dir) if re.match(r'\d{4}-\d{2}-\d{2}\.txt', f)]
        files.sort(reverse=True) # Most recent first
        
        today = datetime.datetime.now()
        count: int = 0
        for filename in files:
            file_date_str = filename.replace(".txt", "")
            file_date = datetime.datetime.strptime(file_date_str, "%Y-%m-%d")
            
            # Check if within window
            if (today - file_date).days <= days:
                file_path = os.path.join(archive_dir, filename)
                with open(file_path, "r", encoding="utf-8") as f:
                    archive_data += f"\n--- Report from {file_date_str} ---\n"
                    archive_data += f.read() + "\n"
                count = count + 1
            if count >= days:
                break
                
        if archive_data:
            return f"\n[REPORTS FROM LAST {days} DAYS - DO NOT REPEAT UNLESS THERE IS NEW PROGRESS]\n{archive_data}\n"
        return ""
    except Exception as e:
        print(f"Warning: Could not read archives: {e}")
        return ""

def get_latest_gemini_model(client):
    """
    Dynamically finds the best available Gemini Pro model.
    Deep Research agents and Flash models are completely excluded.
    """
    try:
        models = client.models.list()
        all_names = [m.name for m in models]

        # 불안정하거나 목적에 맞지 않는 모델 키워드 배제
        bad_keywords = ["flash", "nano", "vision", "latest", "customtools", "experimental",
                        "deep-research", "live", "tts", "embedding", "imagen", "aqa", "preview"]
        
        # Pro 모델 검색
        pro_models = [
            n for n in all_names
            if "gemini" in n.lower() and "pro" in n.lower()
            and not any(bad in n.lower() for bad in bad_keywords)
        ]
        
            if pro_models:
                pro_models.sort(reverse=True)
                latest = pro_models[0].replace("models/", "")
                print(f"Automatically selected latest Pro model: {latest}")
                return latest

            # Fallback: API 조회 실패 시 현재 최신 플래그십 모델로 대체
            return "gemini-3.1-pro-preview"
    except Exception as e:
        print(f"Warning: Could not list models automatically: {e}")
        return "gemini-3.1-pro-preview"

def get_latest_claude_model(client):
    """
    Dynamically finds the latest available Claude Sonnet model
    by querying the Anthropic models API.
    """
    try:
        env_model = os.getenv("CLAUDE_MODEL")
        if env_model and env_model != "latest-sonnet":
            return env_model
        # Dynamic discovery via API
        models_page = client.models.list(limit=100)
        sonnet_models = [m.id for m in models_page.data if "sonnet" in m.id.lower()]
        if sonnet_models:
            sonnet_models.sort(reverse=True)
            latest = sonnet_models[0]
            print(f"Discovered latest Claude Sonnet model: {latest}")
            return latest
        print("Warning: No Sonnet models found via API, using fallback.")
        return "claude-sonnet-4-20250514"
    except Exception as e:
        print(f"Warning: Claude model discovery failed ({e}), using fallback.")
        return "claude-sonnet-4-20250514"

@retry(
    stop=stop_after_attempt(3), 
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry_error_callback=return_none_on_error
)
def run_claude_chat(client, model, messages):
    """
    Executes a chat completion with Claude, retaining the provided message history.
    """
    print(f"Starting Claude action (Model: {model})...")
    message = client.messages.create(
        model=model,
        max_tokens=8192,
        messages=messages
    )
    return message.content[0].text

@retry(
    stop=stop_after_attempt(3), 
    wait=wait_exponential(multiplier=2, min=10, max=60),
    retry_error_callback=return_none_on_error
)
def run_grounded_research(prompt, output_file="research_result.md"):
    """
    Phase 1: Performs web-grounded research using Google Search tool with the Pro model.
    Returns (result_text, chat_session) to maintain the chat context for Phase 3.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "YOUR_API_KEY_HERE":
        print("Error: GEMINI_API_KEY not found in .env file.")
        return None, None

    client = genai.Client(api_key=api_key)
    model_id = get_latest_gemini_model(client)

    print(f"Starting Grounded Research (Model: {model_id}, Tool: google_search)...")
    try:
        # Chat Session 생성하여 문맥 유지 및 Google Search 도구 장착
        chat = client.chats.create(
            model=model_id,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
                temperature=0.7, # Pro 모델에 맞게 온도를 약간 안정적으로 조정
            )
        )
        response = chat.send_message(prompt)
        result_text = response.text

        if not result_text:
            print("Warning: Empty response from grounded research.")
            return None, chat

        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(result_text)
        print(f"Grounded research complete. Saved to {output_file} (Total length: {len(result_text)})")
        
        return result_text, chat
    except Exception as e:
        print(f"An error occurred during grounded research: {e}")
        raise e  

def main():
    parser = argparse.ArgumentParser(description="AI News Pipeline with Gemini Pro & Claude Context")
    parser.add_argument("--url", help="GitHub URL of the prompt file")
    default_output = os.getenv("OUTPUT_FILE") or "trial/1.txt"
    parser.add_argument("--output", default=default_output, help="Output file name")
    
    args = parser.parse_args()
    
    url = args.url or os.getenv("PROMPT_URL")
    prompt_content = None
    
    if url:
        print(f"Using prompt from URL: {url}")
        prompt_content = fetch_prompt_from_github(url)
    else:
        local_news_path = "prompt/news.txt"
        if os.path.exists(local_news_path):
            print(f"Using local prompt file: {local_news_path}")
            with open(local_news_path, "r", encoding="utf-8") as f:
                prompt_content = f.read()
        else:
            print("Error: No PROMPT_URL set and no local prompt/news.txt found.")
            return
    
    if prompt_content:
        archives = get_recent_archives(days=7)
        if archives:
            print("Injecting recent archives for duplicate filtering...")
            prompt_content = archives + "\n" + prompt_content

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # Phase 1: Initial Research (Gemini Pro Chat Session 시작)
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        initial_result, gemini_chat = run_grounded_research(prompt_content, args.output)
        
        if initial_result:
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # Phase 2: Claude Validation (Claude Chat Context 시작)
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            print("\n--- Phase 2: Claude Validation ---")
            
            claude_api_key = os.getenv("CLAUDE_API_KEY")
            if not claude_api_key:
                print("Warning: CLAUDE_API_KEY not found. Skipping validation.")
                feedback = None
            else:
                cclient = anthropic.Anthropic(api_key=claude_api_key, timeout=120.0) 
                claude_model = get_latest_claude_model(cclient)
                
                # Claude 대화 이력 배열
                claude_messages = [
                    {
                        "role": "user", 
                        "content": f"다음은 작성된 AI 뉴스 리서치 초안입니다.\n\n[초안]\n{initial_result}\n\n이 초안에 대해 엄격하게 검증해주세요."
                    }
                ]
                
                feedback = run_claude_chat(cclient, claude_model, claude_messages)
                
                if feedback:
                    claude_messages.append({"role": "assistant", "content": feedback})
            
            if feedback:
                feedback_file = os.getenv("FEEDBACK_FILE") or "trial/feedback.txt"
                feedback_dir = os.path.dirname(feedback_file)
                if feedback_dir and not os.path.exists(feedback_dir):
                    os.makedirs(feedback_dir, exist_ok=True)
                with open(feedback_file, "w", encoding="utf-8") as f:
                    f.write(feedback)
                print(f"Feedback saved to {feedback_file}")
                
                # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                # Phase 3: Gemini Refinement (Gemini Pro 채팅 세션 유지)
                # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                print("\n--- Phase 3: Gemini Refinement ---")
                refined_output = os.getenv("REFINED_OUTPUT_FILE") or "trial/2.txt"
                
                refine_prompt = (
                    f"방금 네가 작성한 리서치 초안에 대해 다음과 같은 검증 피드백이 도착했어.\n\n"
                    f"[피드백]\n{feedback}\n\n"
                    f"지시사항:\n"
                    f"1. 추가 인터넷 검색 없이 기존 대화(네가 작성한 원문) 안의 정보만 사용하세요.\n"
                    f"2. 피드백을 반영하여 날짜·사실이 불확실하거나 오류가 있는 뉴스 항목은 삭제하세요.\n"
                    f"3. 삭제로 항목이 부족해지면, 초안 작성 시 확보했던 정보 중 주요 뉴스로 다루지 않은 다른 내용을 찾아 새 항목으로 채우세요.\n"
                    f"4. 처음 요구했던 구조(섹션, 마크다운 형식)를 그대로 유지하여 최종본을 작성하세요."
                )
                
                max_retries = 3
                refined_result = None
                
                for attempt in range(max_retries):
                    try:
                        print(f"Refining with Gemini Chat Session (Attempt {attempt + 1}/{max_retries})...")
                        refine_response = gemini_chat.send_message(refine_prompt)
                        refined_result = refine_response.text
                        
                        refined_dir = os.path.dirname(refined_output)
                        if refined_dir and not os.path.exists(refined_dir):
                            os.makedirs(refined_dir, exist_ok=True)
                        with open(refined_output, "w", encoding="utf-8") as f:
                            f.write(refined_result)
                        print(f"Refinement complete. Saved to {refined_output} (Total length: {len(refined_result)})")
                        break
                        
                    except Exception as e:
                        print(f"Warning: Refinement attempt {attempt + 1} failed ({e}).")
                        if attempt < max_retries - 1:
                            print("Sleeping for 20 seconds before retrying...")
                            time.sleep(20)
                        else:
                            print("All refinement attempts failed. Using initial research as fallback.")
                            refined_result = initial_result
            else:
                print("Claude validation failed or skipped. Using initial research as refined result.")
                refined_result = initial_result
                refined_output = args.output
            
            if refined_result:
                # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                # Phase 4: Claude Audit 2 & English Translation (대화 이력 이어가기)
                # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                print("\n--- Phase 4: Claude Audit 2 & English Translation ---")
                
                if feedback and claude_api_key:
                    claude_messages.append({
                        "role": "user", 
                        "content": f"다음은 당신의 피드백을 반영하여 새롭게 업데이트된 최종 AI 뉴스 보고서입니다.\n\n[수정된 최종본]\n{refined_result}\n\n초기 초안의 검증 내역과 이 최종본을 바탕으로, 기존 구조를 완벽히 유지한 채 최상의 영어 비즈니스 문장으로 최종 번역을 수행해주세요."
                    })
                    translated_result = run_claude_chat(cclient, claude_model, claude_messages)
                else:
                    translated_result = None

                final_content_for_html = translated_result if translated_result else refined_result
                
                if final_content_for_html:
                    today = datetime.datetime.now().strftime("%Y-%m-%d")
                    archive_file = f"data/{today}.txt"
                    archive_dir = os.path.dirname(archive_file)
                    if archive_dir and not os.path.exists(archive_dir):
                        os.makedirs(archive_dir, exist_ok=True)
                    
                    with open(archive_file, "w", encoding="utf-8") as f:
                        f.write(final_content_for_html)
                    print(f"Archived content to {archive_file}")

                    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                    # Phase 5: HTML Generation
                    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                    print("\n--- Phase 5: HTML Generation ---")
                    
                    html_prompt_url = os.getenv("HTML_PROMPT_URL")
                    html_prompt = None
                    
                    if html_prompt_url:
                        print(f"Using HTML prompt from URL: {html_prompt_url}")
                        html_prompt = fetch_prompt_from_github(html_prompt_url)
                    else:
                        local_html_path = "prompt/html.txt"
                        if os.path.exists(local_html_path):
                            print(f"Using local HTML prompt file: {local_html_path}")
                            with open(local_html_path, "r", encoding="utf-8") as f:
                                html_prompt = f.read()
                        else:
                            print("Warning: HTML prompt source not found.")
                            
                    if html_prompt:
                        generate_html(final_content_for_html, html_prompt, "index.html")
                    else:
                        print("Error: No content available for HTML generation.")
                else:
                    print("Error: No content available for translation/archiving.")
    else:
        print("Failed to fetch prompt.")

@retry(
    stop=stop_after_attempt(3), 
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry_error_callback=return_none_on_error
)
def generate_html(content, prompt, output_file="index.html"):
    """
    Generates index.html using Gemini Pro.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)
    model_id = get_latest_gemini_model(client)
    
    print(f"Generating HTML using model: {model_id}...")
    response = client.models.generate_content(
        model=model_id,
        contents=[f"{prompt}\n{content}"]
    )
    html_content = response.text
    if "```html" in html_content:
        html_content = html_content.split("```html")[1].split("```")[0].strip()
    elif "```" in html_content:
        html_content = html_content.split("```")[1].split("```")[0].strip()
        
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"HTML generation complete. Saved to {output_file}")

if __name__ == "__main__":
    main()
