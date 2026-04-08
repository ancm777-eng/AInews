```python
import os
import time
import sys
import argparse
import re
import datetime
from dotenv import load_dotenv
from google import genai
from google.genai import types
import anthropic
from tenacity import retry, stop_after_attempt, wait_exponential
from github_utils import fetch_prompt_from_github

# 실시간 로그 출력을 위한 설정 (GitHub Actions에서 로그가 멈춰보이는 현상 방지)
sys.stdout.reconfigure(line_buffering=True)

# 환경 변수 로드
load_dotenv()

def return_none_on_error(retry_state):
    """모든 재시도 실패 시 호출되는 콜백"""
    print(f"\nAction ultimately failed after {retry_state.attempt_number} attempts: {retry_state.outcome.exception()}")
    return None

def get_recent_archives(days=7):
    """중복 필터링을 위해 최근 N일간의 리포트를 읽어옴"""
    archive_data = ""
    archive_dir = "data"
    if not os.path.exists(archive_dir):
        return ""
    
    try:
        files = [f for f in os.listdir(archive_dir) if re.match(r'\d{4}-\d{2}-\d{2}\.txt', f)]
        files.sort(reverse=True)
        
        today = datetime.datetime.now()
        count = 0
        for filename in files:
            file_date_str = filename.replace(".txt", "")
            file_date = datetime.datetime.strptime(file_date_str, "%Y-%m-%d")
            
            if (today - file_date).days <= days:
                file_path = os.path.join(archive_dir, filename)
                with open(file_path, "r", encoding="utf-8") as f:
                    archive_data += f"\n--- Report from {file_date_str} ---\n"
                    archive_data += f.read() + "\n"
                count += 1
            if count >= days:
                break
                
        if archive_data:
            return f"\n[REPORTS FROM LAST {days} DAYS - DO NOT REPEAT UNLESS THERE IS NEW PROGRESS]\n{archive_data}\n"
        return ""
    except Exception as e:
        print(f"Warning: Could not read archives: {e}")
        return ""

def get_latest_gemini_model(client):
    """최신 Gemini Pro 모델 동적 탐색 (3.0-pro 우선)"""
    try:
        models = client.models.list()
        all_names = [m.name for m in models]
        
        bad_keywords = ["flash", "nano", "vision", "latest", "customtools", 
                        "deep-research", "live", "tts", "embedding", "imagen", "aqa"]
        
        for name in all_names:
            if "gemini-3.0-pro" in name.lower() and not any(bad in name.lower() for bad in bad_keywords):
                target = name.replace("models/", "")
                print(f"Automatically selected Gemini Pro model: {target}")
                return target
        
        pro_models = [
            n for n in all_names
            if "gemini" in n.lower() and "pro" in n.lower()
            and not any(bad in n.lower() for bad in bad_keywords)
        ]
        
        if pro_models:
            pro_models.sort(reverse=True)
            latest = pro_models[0].replace("models/", "")
            print(f"Automatically selected Gemini Pro model: {latest}")
            return latest
            
        return "gemini-3.0-pro"
    except Exception as e:
        print(f"Warning: Gemini model discovery failed ({e}), using default Pro.")
        return "gemini-3.0-pro"

def get_latest_claude_model(client):
    """최신 Claude Sonnet 모델 동적 탐색"""
    try:
        models_page = client.models.list(limit=50)
        sonnet_models = [m.id for m in models_page.data if "sonnet" in m.id.lower()]
        if sonnet_models:
            sonnet_models.sort(reverse=True)
            return sonnet_models[0]
        return "claude-3-5-sonnet-20241022"
    except Exception as e:
        print(f"Warning: Claude model discovery failed ({e}), using fallback.")
        return "claude-3-5-sonnet-20241022"

@retry(
    stop=stop_after_attempt(3), 
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry_error_callback=return_none_on_error
)
def run_claude_chat(client, model, messages):
    """Claude API 호출 (문맥 유지형)"""
    print(f"Starting Claude action (Model: {model})...")
    message = client.messages.create(
        model=model,
        max_tokens=8192,
        messages=messages
    )
    return message.content[0].text

@retry(
    stop=stop_after_attempt(2), 
    wait=wait_exponential(multiplier=2, min=10, max=30), # 타임아웃 발생 시 10~30초 쉬었다가 재시도
    retry_error_callback=return_none_on_error
)
def run_grounded_research(client, model_id, prompt, output_file="research_result.md"):
    """Phase 1: Google Search 기반 리서치 및 채팅 세션 생성"""
    print(f"Starting Grounded Research (Model: {model_id})...")
    try:
        chat = client.chats.create(
            model=model_id,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
                temperature=0.1,
            )
        )
        response = chat.send_message(prompt)
        result_text = response.text

        if not result_text:
            return None

        os.makedirs(os.path.dirname(output_file), exist_ok=True) if os.path.dirname(output_file) else None
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(result_text)
        print(f"Phase 1 complete. Saved to {output_file}")
        return result_text, chat
    except Exception as e:
        print(f"Error in Phase 1: {e}")
        raise e

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", help="GitHub URL for news prompt")
    parser.add_argument("--output", default="trial/1.txt")
    args = parser.parse_args()

    # 1. API 클라이언트 초기화
    g_api_key = os.getenv("GEMINI_API_KEY")
    c_api_key = os.getenv("CLAUDE_API_KEY")
    
    if not g_api_key:
        print("Missing GEMINI_API_KEY")
        sys.exit(1)

    # 타임아웃 한도를 150초로 넉넉히 주어 무거운 검색도 기다려줌
    g_client = genai.Client(api_key=g_api_key, http_options={'timeout': 150.0})
    g_model = get_latest_gemini_model(g_client)
    
    c_client = anthropic.Anthropic(api_key=c_api_key, timeout=120.0) if c_api_key else None
    c_model = get_latest_claude_model(c_client) if c_client else None

    # 2. 프롬프트 준비
    url = args.url or os.getenv("PROMPT_URL")
    if url:
        prompt_content = fetch_prompt_from_github(url)
    else:
        try:
            with open("prompt/news.txt", "r", encoding="utf-8") as f:
                prompt_content = f.read()
        except FileNotFoundError:
            try:
                with open("news.txt", "r", encoding="utf-8") as f:
                    prompt_content = f.read()
            except FileNotFoundError:
                print("Prompt source not found.")
                sys.exit(1)

    # 🔥 [수정됨] KST 시간 주입 및 오버라이드 룰 강제 적용
    current_kst = datetime.datetime.now().strftime("%Y-%m-%d %H:%M KST")
    system_instr = (
        f"[🔥 SYSTEM OVERRIDE & ALERT: 매우 중요한 지시사항 🔥]\n"
        f"1. 현재 한국 표준시(KST)는 {current_kst} 입니다. 이를 '오늘(TODAY)'의 기준으로 삼으십시오.\n"
        f"2. 사용자의 프롬프트 원문에 있는 '쿼리 0: today's date 검색' 및 '날짜 확인 실패 시 Briefing aborted 출력' 규칙을 **완벽하게 무시**하십시오. 시스템이 이미 날짜를 제공했으므로 절대 중단해서는 안 됩니다. 즉시 최신 AI 뉴스 리서치부터 시작하십시오.\n"
        f"3. 문장 내 인라인 LaTeX($) 사용은 절대 금지하며, 수식이나 변수(예: v, x, n)는 굵은 글씨 또는 일반 텍스트로만 표기하십시오.\n\n"
    )
    prompt_content = system_instr + get_recent_archives(7) + prompt_content

    # Phase 1: Grounded Research (Fail-Fast 및 Unpack 방어 적용)
    research_output = run_grounded_research(g_client, g_model, prompt_content, args.output)
    
    if not research_output:
        print("❌ API 서버 과부하 또는 오류로 리서치 실패. 워크플로우를 즉시 종료하여 비용을 보호합니다.")
        sys.exit(1)
        
    initial_result, gemini_chat = research_output

    # Phase 2: Claude Validation
    feedback = None
    claude_messages = []
    if c_client:
        print("\n--- Phase 2: Claude Validation ---")
        claude_messages = [
            {"role": "user", "content": f"다음 AI 뉴스 초안의 사실 관계와 KST 기준 시간 윈도우를 검증하십시오.\n\n[초안]\n{initial_result}"}
        ]
        feedback = run_claude_chat(c_client, c_model, claude_messages)
        if feedback:
            claude_messages.append({"role": "assistant", "content": feedback})
            with open("trial/feedback.txt", "w", encoding="utf-8") as f:
                f.write(feedback)

    # Phase 3: Gemini Refinement (Session Maintained)
    refined_result = initial_result
    if feedback:
        print("\n--- Phase 3: Gemini Refinement ---")
        refine_prompt = (
            "위 피드백을 반영하여 최종본을 작성하십시오.\n"
            "1. 대화 내 정보만 활용\n2. 오류 항목 삭제 및 신규 항목 보충\n"
            "3. 인라인 LaTeX 사용 절대 금지\n4. 기존 섹션 구조 유지"
        )
        
        # 🔥 [수정됨] Phase 3에도 서버 통신 끊김(Server disconnected) 방어용 재시도 로직 추가
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = gemini_chat.send_message(refine_prompt)
                refined_result = response.text
                with open("trial/2.txt", "w", encoding="utf-8") as f:
                    f.write(refined_result)
                print("✅ Refinement complete.")
                break
            except Exception as e:
                print(f"⚠️ Refinement attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(10)
                else:
                    print("❌ All refinement attempts failed. Using initial result.")

    # Phase 4: Claude Translation (Session Maintained)
    final_content = refined_result
    if c_client and feedback:
        print("\n--- Phase 4: Claude Translation ---")
        claude_messages.append({
            "role": "user", 
            "content": "수정된 최종본을 바탕으로 기존 구조를 완벽히 유지한 최상의 비즈니스 영어 보고서로 번역하십시오."
        })
        final_content = run_claude_chat(c_client, c_model, claude_messages)

    # Archive saving
    today_str = datetime.datetime.now().strftime("%Y-%m-%d")
    os.makedirs("data", exist_ok=True)
    with open(f"data/{today_str}.txt", "w", encoding="utf-8") as f:
        f.write(final_content)

    # Phase 5: HTML Generation
    print("\n--- Phase 5: HTML Generation ---")
    html_prompt_url = os.getenv("HTML_PROMPT_URL")
    html_prompt_content = ""
    
    if html_prompt_url:
        html_prompt_content = fetch_prompt_from_github(html_prompt_url)
    else:
        try:
            with open("prompt/html.txt", "r", encoding="utf-8") as f:
                html_prompt_content = f.read()
        except FileNotFoundError:
            try:
                with open("html.txt", "r", encoding="utf-8") as f:
                    html_prompt_content = f.read()
            except FileNotFoundError:
                print("⚠️ HTML prompt source not found. Skipping generation.")

    if html_prompt_content:
        print(f"Generating HTML using model: {g_model}...")
        try:
            response = g_client.models.generate_content(
                model=g_model,
                contents=[f"{html_prompt_content}\n\n{final_content}"]
            )
            html_code = response.text
            if "```html" in html_code:
                html_code = html_code.split("```html")[1].split("```")[0].strip()
            elif "```" in html_code:
                html_code = html_code.split("```")[1].split("```")[0].strip()
            with open("index.html", "w", encoding="utf-8") as f:
                f.write(html_code)
            print("✅ index.html saved successfully.")
        except Exception as e:
            print(f"❌ HTML Generation failed: {e}")
    else:
        print("❌ No HTML prompt available. index.html was not generated.")
    
    print("\nPipeline finished successfully.")

if __name__ == "__main__":
    main()


```
