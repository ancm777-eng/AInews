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

def get_latest_gemini_model(client, require_agent=False):
    """
    Dynamically finds the best available Gemini model.
    If require_agent=True, selects deep-research agents.
    If require_agent=False, prefers Gemini Flash models over Pro models.
    """
    try:
        models = client.models.list()
        all_names = [m.name for m in models]

        # Priority 1: Pick Deep Research Agent
        if require_agent:
            dr_models = [m.name for m in models if "deep-research-pro" in m.name.lower()]
            if dr_models:
                dr_models.sort(reverse=True)
                latest_dr = dr_models[0].replace("models/", "")
                print(f"Found latest specialized Deep Research agent: {latest_dr}")
                return latest_dr
            return "deep-research-pro-preview-12-2025"

        # Priority 2: Pick Gemini Flash model
        # 'preview' 키워드 추가하여 503 에러가 잦은 불안정한 프리뷰 모델 배제
        bad_keywords = ["nano", "vision", "latest", "customtools", "experimental",
                        "deep-research", "live", "tts", "embedding", "imagen", "aqa", "preview"]
        flash_models = [
            n for n in all_names
            if "gemini" in n.lower() and "flash" in n.lower()
            and not any(bad in n.lower() for bad in bad_keywords)
        ]
        if flash_models:
            flash_models.sort(reverse=True)
            latest = flash_models[0].replace("models/", "")
            print(f"Automatically selected latest Flash model: {latest}")
            return latest

        # Fallback to any Pro model
        pro_models = [
            n for n in all_names
            if "gemini" in n.lower() and "pro" in n.lower()
            and not any(bad in n.lower() for bad in bad_keywords + ["flash", "deep-research"])
        ]
        if pro_models:
            pro_models.sort(reverse=True)
            latest = pro_models[0].replace("models/", "")
            print(f"Automatically selected Pro model (Flash not found): {latest}")
            return latest

        return "gemini-3.0-flash"
    except Exception as e:
        print(f"Warning: Could not list models automatically: {e}")
        return "deep-research-pro-preview-12-2025" if require_agent else "gemini-3.0-flash"

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
def validate_with_claude(content, custom_prompt="검증해주세요"):
    """
    Validates the research content using Claude.
    """
    api_key = os.getenv("CLAUDE_API_KEY")
    if not api_key:
        print("Warning: CLAUDE_API_KEY not found. Skipping validation.")
        return None
    
    # Initialize client with a generous timeout for large content
    client = anthropic.Anthropic(api_key=api_key, timeout=120.0) 
    model = get_latest_claude_model(client)
    
    print(f"Starting Claude action (Model: {model}, Prompt: {custom_prompt})...")
    message = client.messages.create(
        model=model,
        max_tokens=8192,  # Increased for translation
        messages=[
            {"role": "user", "content": f"{content}\n\n{custom_prompt}"}
        ]
    )
    return message.content[0].text

@retry(
    stop=stop_after_attempt(3), # 최대 3번까지 재시도
    wait=wait_exponential(multiplier=2, min=10, max=60), # 실패할 때마다 10초, 20초, 40초 대기
    retry_error_callback=return_none_on_error
)
def run_grounded_research(prompt, output_file="research_result.md"):
    """
    Phase 1: Performs fast web-grounded research using Google Search tool.
    Returns (result_text, None) to maintain the same interface as run_deep_research.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "YOUR_API_KEY_HERE":
        print("Error: GEMINI_API_KEY not found in .env file.")
        return None, None

    client = genai.Client(api_key=api_key)
    model_id = get_latest_gemini_model(client, require_agent=False)

    print(f"Starting Grounded Research (Model: {model_id}, Tool: google_search)...")
    try:
        response = client.models.generate_content(
            model=model_id,
            contents=[prompt],
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
                temperature=1.0,
            )
        )
        result_text = response.text

        if not result_text:
            print("Warning: Empty response from grounded research.")
            return None, None

        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(result_text)
        print(f"Grounded research complete. Saved to {output_file} (Total length: {len(result_text)})")
        return result_text, None
    except Exception as e:
        print(f"An error occurred during grounded research: {e}")
        raise e  # @retry 가 잡아서 재시도할 수 있도록 에러를 다시 던집니다.

@retry(
    stop=stop_after_attempt(3), 
    wait=wait_exponential(multiplier=2, min=10, max=60), 
    retry_error_callback=return_none_on_error
)
def run_deep_research(prompt, output_file="research_result.md", agent_id=None, previous_interaction_id=None):
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "YOUR_API_KEY_HERE":
        print("Error: GEMINI_API_KEY not found in .env file.")
        return None, None

    client = genai.Client(api_key=api_key)
    research_agent = agent_id or os.getenv("RESEARCH_AGENT")
    
    if not research_agent or research_agent.lower() == "latest-pro":
        research_agent = get_latest_gemini_model(client, require_agent=True)
    
    print(f"Starting Gemini Deep Research (Agent: {research_agent}, Continued: {previous_interaction_id is not None})...")
    
    try:
        if "deep-research" in research_agent.lower():
            interaction = client.interactions.create(
                agent=research_agent, 
                input=prompt, 
                background=True,
                previous_interaction_id=previous_interaction_id
            )
        else:
            interaction = client.interactions.create(
                model=research_agent, 
                input=prompt, 
                background=True,
                previous_interaction_id=previous_interaction_id
            )
        print(f"Interaction ID: {interaction.id}")
        
        start_time = time.time()
        while True:
            try:
                interaction = client.interactions.get(interaction.id)
            except Exception as get_err:
                print(f"\nWarning: Error polling status ({get_err}), retrying in 30s...")
                time.sleep(30)
                continue
                
            if interaction.status == "completed":
                print("\nResearch Turn Completed!")
                result_text = "\n".join([o.text for o in interaction.outputs if o.text])
                
                if not result_text:
                    print("Warning: API returned completed status but no text outputs were found.")
                    result_text = "No content returned from research agent."

                output_dir = os.path.dirname(output_file)
                if output_dir and not os.path.exists(output_dir):
                    os.makedirs(output_dir, exist_ok=True)
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(result_text)
                print(f"Results saved to {output_file} (Total length: {len(result_text)})")
                return result_text, interaction.id
            elif interaction.status == "failed":
                print(f"\nResearch failed: {interaction.error}")
                return None, interaction.id
            else:
                elapsed = int(time.time() - start_time)
                print(f"\rStatus: {interaction.status} (Elapsed: {elapsed}s)...", end="", flush=True)
                time.sleep(10)
    except Exception as e:
        print(f"An error occurred during API interaction: {e}")
        raise e

def main():
    parser = argparse.ArgumentParser(description="Gemini Deep Research with GitHub Prompt")
    parser.add_argument("--url", help="GitHub URL of the prompt file")
    parser.add_argument("--agent", help="Gemini Research Agent ID")
    default_output = os.getenv("OUTPUT_FILE") or "trial/1.txt"
    parser.add_argument("--output", default=default_output, help="Output file name")
    
    args = parser.parse_args()
    
    # 1. 뉴스 프롬프트 가져오기 (URL 우선, 없으면 로컬 파일)
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

        # Step 1: Initial Research (자동 재시도 로직 적용됨)
        initial_result, _ = run_grounded_research(prompt_content, args.output)
        
        if initial_result:
            # Step 2: Claude Validation
            print("\n--- Phase 2: Claude Validation ---")
            feedback = validate_with_claude(initial_result)
            
            if feedback:
                feedback_file = os.getenv("FEEDBACK_FILE") or "trial/feedback.txt"
                feedback_dir = os.path.dirname(feedback_file)
                if feedback_dir and not os.path.exists(feedback_dir):
                    os.makedirs(feedback_dir, exist_ok=True)
                with open(feedback_file, "w", encoding="utf-8") as f:
                    f.write(feedback)
                print(f"Feedback saved to {feedback_file}")
                
                # Step 3: Refinement using feedback (503 방어 루프 적용)
                print("\n--- Phase 3: Gemini Refinement ---")
                refined_output = os.getenv("REFINED_OUTPUT_FILE") or "trial/2.txt"
                
                api_key = os.getenv("GEMINI_API_KEY")
                client_gemini = genai.Client(api_key=api_key)
                gemini_model_id = get_latest_gemini_model(client_gemini, require_agent=False)
                
                refine_prompt = (
                    f"아래는 AI 뉴스 보고서 원문(Phase 1 리서치 전체 내용)과 검증 피드백입니다.\n\n"
                    f"[원문]\n{initial_result}\n\n"
                    f"[피드백]\n{feedback}\n\n"
                    f"지시사항:\n"
                    f"1. 추가 인터넷 검색 없이 '원문' 안의 정보만 사용하세요.\n"
                    f"2. 날짜·사실이 불확실하거나 오류가 있는 뉴스 항목은 삭제하세요.\n"
                    f"3. 삭제로 항목이 부족해지면, 원문에서 언급됐지만 주요 뉴스로 다루지 않은 다른 내용을 찾아 새 항목으로 채우세요. 원문에는 최종 보고서보다 훨씬 많은 소스가 담겨 있습니다.\n"
                    f"4. 원문의 구조(섹션, 마크다운 형식)를 그대로 유지하세요."
                )
                
                max_retries = 3
                refined_result = None
                
                for attempt in range(max_retries):
                    try:
                        print(f"Refining with model: {gemini_model_id} (Attempt {attempt + 1}/{max_retries})...")
                        refine_response = client_gemini.models.generate_content(
                            model=gemini_model_id,
                            contents=[refine_prompt]
                        )
                        refined_result = refine_response.text
                        
                        refined_dir = os.path.dirname(refined_output)
                        if refined_dir and not os.path.exists(refined_dir):
                            os.makedirs(refined_dir, exist_ok=True)
                        with open(refined_output, "w", encoding="utf-8") as f:
                            f.write(refined_result)
                        print(f"Refinement complete. Saved to {refined_output} (Total length: {len(refined_result)})")
                        break # 성공 시 루프 탈출
                        
                    except Exception as e:
                        print(f"Warning: Refinement attempt {attempt + 1} failed ({e}).")
                        if attempt < max_retries - 1:
                            print("Sleeping for 20 seconds before retrying...")
                            time.sleep(20) # 20초 대기 후 재시도
                        else:
                            print("All refinement attempts failed. Using initial research as fallback.")
                            refined_result = initial_result
            else:
                print("Claude validation failed or skipped. Using initial research as refined result.")
                refined_result = initial_result
                refined_output = args.output
            
            if refined_result:
                # Phase 4: Claude Audit 2 & English Translation
                print("\n--- Phase 4: Claude Audit 2 & English Translation ---")
                translation_prompt = "구조를 유지한 채 최상의 영어 문장으로 최종 수정 및 번역해주세요."
                translated_result = validate_with_claude(refined_result, custom_prompt=translation_prompt)
                
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

                    # Phase 5: HTML Generation
                    print("\n--- Phase 5: HTML Generation ---")
                    
                    # HTML 프롬프트 가져오기 (URL 우선, 없으면 로컬 파일)
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
    Generates index.html using Gemini.
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
