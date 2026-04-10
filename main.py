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

# 실시간 로그 출력 설정
sys.stdout.reconfigure(line_buffering=True)

load_dotenv()

def return_none_on_error(retry_state):
    print(f"\nAction ultimately failed after {retry_state.attempt_number} attempts: {retry_state.outcome.exception()}")
    return None

def get_recent_archives(days=7):
    archive_data = ""
    archive_dir = "data"
    if not os.path.exists(archive_dir): return ""
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
                    archive_data += f"\n--- Report from {file_date_str} ---\n" + f.read() + "\n"
                count += 1
            if count >= days: break
        if archive_data:
            return f"\n[REPORTS FROM LAST {days} DAYS - DO NOT REPEAT UNLESS THERE IS NEW PROGRESS]\n{archive_data}\n"
        return ""
    except Exception as e:
        print(f"Warning: Could not read archives: {e}")
        return ""

def get_latest_gemini_model(client):
    """
    미래 확장형 모델 탐색기:
    모델명에서 숫자(버전)를 추출하여 가장 높은 숫자의 정식 Pro 버전을 자동으로 선택합니다.
    (preview 등 불안정한 버전은 제외)
    """
    try:
        models = client.models.list()
        all_names = [m.name for m in models]
        
        # 불안정한 preview, exp 및 경량화(flash/nano) 버전 무조건 배제
        bad_keywords = ["flash", "nano", "vision", "latest", "customtools", "deep-research", 
                        "live", "tts", "embedding", "imagen", "aqa", "preview", "exp"]
        
        pro_models = [
            n for n in all_names 
            if "gemini" in n.lower() and "pro" in n.lower() 
            and not any(bad in n.lower() for bad in bad_keywords)
        ]
        
        if pro_models:
            # 예: 'gemini-3.1-pro' -> (3, 1) 로 변환하여 내림차순 정렬 (가장 높은 숫자가 1등)
            pro_models.sort(key=lambda x: tuple(int(num) for num in re.findall(r'\d+', x)), reverse=True)
            latest = pro_models[0].replace("models/", "")
            print(f"Automatically selected Gemini Pro model (by semantic version): {latest}")
            return latest
        
        return "gemini-3.0-pro" # Fallback
    except Exception as e:
        print(f"Warning: Gemini model discovery failed, using fallback.")
        return "gemini-3.0-pro"

def get_latest_claude_model(client):
    """
    미래 확장형 모델 탐색기:
    Claude 모델명에서 숫자를 추출하여 가장 높은 버전을 자동으로 선택합니다.
    """
    try:
        models_page = client.models.list(limit=50)
        sonnet_models = [m.id for m in models_page.data if "sonnet" in m.id.lower()]
        
        if sonnet_models:
            # 예: 'claude-sonnet-4-6' -> (4, 6) 로 변환하여 가장 높은 숫자가 1등으로 오게 정렬
            sonnet_models.sort(key=lambda x: tuple(int(num) for num in re.findall(r'\d+', x)), reverse=True)
            target = sonnet_models[0]
            print(f"Automatically selected Claude model (by semantic version): {target}")
            return target
            
        return "claude-sonnet-4-6" # Fallback
    except Exception as e:
        print(f"Warning: Claude model discovery failed, using fallback.")
        return "claude-sonnet-4-6"

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10), retry_error_callback=return_none_on_error)
def run_claude_chat(client, model, messages):
    print(f"Starting Claude action (Model: {model})...")
    message = client.messages.create(model=model, max_tokens=8192, messages=messages)
    return message.content[0].text

@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=2, min=10, max=30), retry_error_callback=return_none_on_error)
def run_grounded_research(client, model_id, prompt, output_file="research_result.md"):
    print(f"Starting Grounded Research (Model: {model_id})...")
    try:
        chat = client.chats.create(model=model_id, config=types.GenerateContentConfig(tools=[types.Tool(google_search=types.GoogleSearch())], temperature=0.1))
        response = chat.send_message(prompt)
        result_text = response.text
        if not result_text: return None
        os.makedirs(os.path.dirname(output_file), exist_ok=True) if os.path.dirname(output_file) else None
        with open(output_file, "w", encoding="utf-8") as f: f.write(result_text)
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

    g_api_key = os.getenv("GEMINI_API_KEY")
    c_api_key = os.getenv("CLAUDE_API_KEY")
    if not g_api_key: sys.exit(1)

    g_client = genai.Client(api_key=g_api_key, http_options={'timeout': 150.0})
    g_model = get_latest_gemini_model(g_client)
    c_client = anthropic.Anthropic(api_key=c_api_key, timeout=120.0) if c_api_key else None
    c_model = get_latest_claude_model(c_client) if c_client else None

    # 프롬프트 로드
    url = args.url or os.getenv("PROMPT_URL")
    if url:
        prompt_content = fetch_prompt_from_github(url)
    else:
        try:
            with open("prompt/news.txt", "r", encoding="utf-8") as f: prompt_content = f.read()
        except FileNotFoundError:
            try:
                with open("news.txt", "r", encoding="utf-8") as f: prompt_content = f.read()
            except FileNotFoundError:
                sys.exit(1)

    # 파업 조항 정규식 청소
    prompt_content = re.sub(r"가장 먼저 다음 검색을.*?Briefing aborted\.", "", prompt_content, flags=re.DOTALL | re.IGNORECASE)
    prompt_content = prompt_content.replace("\"SYSTEM STATUS: Unable to confirm today's date. Briefing aborted.\"", "")

    archive_text = get_recent_archives(7)
    if archive_text:
        prompt_content = archive_text + "\n" + prompt_content

    # 명확한 오버라이드 지시 및 표기 규칙 주입
    current_kst = datetime.datetime.now().strftime("%Y-%m-%d %H:%M KST")
    override_instr = (
        f"\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"[🔥 FINAL SYSTEM OVERRIDE: KST {current_kst} 기준 🔥]\n"
        f"1. 날짜 확인을 위한 사전 검색(쿼리 0)을 생략하십시오. 위 명시된 KST 시간을 '오늘'로 확정합니다.\n"
        f"2. 'Briefing aborted'와 관련된 모든 기존 지시를 완전히 무시하고, 무조건 최신 AI 뉴스를 검색하여 브리핑을 시작하십시오.\n"
        f"3. 모든 답변에서 인라인 수식 기호($) 사용을 절대 금지합니다. 수식이나 중간 기호, 벡터(v, w, x 등)는 굵은 글씨로, 일반 변수(n, x, y 등)는 상황에 맞게 굵은 글씨 또는 일반 텍스트로만 표기하십시오.\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    )
    prompt_content = prompt_content + override_instr

    # ---------------------------------------------------------
    # [신규 추가 로직] ref.txt 확인 및 멀티 타겟 프롬프트 주입
    # ---------------------------------------------------------
    ref_content = ""
    ref_count = 0
    if os.path.exists("ref.txt"):
        try:
            with open("ref.txt", "r", encoding="utf-8") as f:
                # 빈 줄을 제외하고 각 줄을 읽어 리스트로 만듭니다.
                ref_lines = [line.strip() for line in f.readlines() if line.strip()]
                if ref_lines:
                    # AI가 읽기 쉽게 불릿 포인트 형태로 결합합니다.
                    ref_content = "\n".join(f"▶ {line}" for line in ref_lines)
                    ref_count = len(ref_lines)
        except Exception as e:
            print(f"Warning: Could not read ref.txt: {e}")

    if ref_content:
        # 5개 중에 몇 개를 자율적으로 찾을지 계산합니다. (최소 0개)
        auto_count = max(0, 5 - ref_count)
        
        ref_instr = (
            f"\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"[🎯 MANDATORY TARGET NEWS (최우선 반영 지시)]\n"
            f"사용자가 오늘 특별히 다음 **{ref_count}개**의 주제/사건에 대한 분석을 요청했습니다:\n"
            f"{ref_content}\n\n"
            f"지시사항: 당신이 선정하는 최종 5개의 주요 뉴스 중, **위 요청된 {ref_count}개의 주제를 빠짐없이 각각 다루는 최신 뉴스를 무조건 포함**시키십시오. "
            f"그리고 나머지 **{auto_count}개**의 뉴스는 산업 전체에 가장 큰 파급력을 미칠 핵심 뉴스로 당신이 직접 발굴하여 총 5개를 맞추십시오.\n"
            f"(단, 요청된 주제가 5개를 넘어가더라도 최종 리포트는 반드시 가장 중요한 5개의 뉴스만으로 압축해서 구성해야 합니다.)\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        )
        prompt_content += ref_instr
        print(f"🎯 ref.txt에서 {ref_count}개의 타겟 주제를 확인하여 프롬프트에 주입했습니다.")
    else:
        print("ℹ️ ref.txt 파일이 없거나 비어있어 기본 탐색 모드로 진행합니다.")
    # ---------------------------------------------------------

    # Phase 1: Grounded Research
    research_output = run_grounded_research(g_client, g_model, prompt_content, args.output)
    if not research_output:
        print("❌ API 서버 과부하로 리서치 실패. 워크플로우를 즉시 종료합니다.")
        sys.exit(1)
    initial_result, gemini_chat = research_output

    # Phase 2: Claude Validation
    feedback = None
    claude_messages = []
    if c_client:
        print("\n--- Phase 2: Claude Validation ---")
        claude_messages = [{"role": "user", "content": f"다음 AI 뉴스 초안의 사실 관계와 KST 기준 시간 윈도우를 검증하십시오.\n\n[초안]\n{initial_result}"}]
        feedback = run_claude_chat(c_client, c_model, claude_messages)
        if feedback:
            claude_messages.append({"role": "assistant", "content": feedback})
            with open("trial/feedback.txt", "w", encoding="utf-8") as f: f.write(feedback)

    # Phase 3: Gemini Refinement
    refined_result = initial_result
    if feedback:
        print("\n--- Phase 3: Gemini Refinement ---")
        refine_prompt = (
            "위 피드백을 반영하여 최종본을 작성하십시오.\n"
            "1. 대화 내 정보만 활용\n2. 오류 항목 삭제 및 신규 항목 보충\n"
            "3. 모든 답변에서 인라인 수식 기호 절대 금지 (벡터는 굵은 글씨, 일반 변수는 일반 텍스트 표기)\n4. 기존 섹션 구조 유지"
        )
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = gemini_chat.send_message(refine_prompt)
                refined_result = response.text
                with open("trial/2.txt", "w", encoding="utf-8") as f: f.write(refined_result)
                print("✅ Refinement complete.")
                break
            except Exception as e:
                print(f"⚠️ Phase 3 attempt {attempt + 1} failed (Server Disconnected): {e}")
                if attempt < max_retries - 1: time.sleep(15)
                else: print("❌ All refinement attempts failed. Using initial result.")

    # Phase 4: Claude Translation
    final_content = refined_result
    if c_client and feedback:
        print("\n--- Phase 4: Claude Translation ---")
        claude_messages.append({"role": "user", "content": "수정된 최종본을 바탕으로 기존 구조를 완벽히 유지한 최상의 비즈니스 영어 보고서로 번역하십시오."})
        final_content = run_claude_chat(c_client, c_model, claude_messages)

    today_str = datetime.datetime.now().strftime("%Y-%m-%d")
    os.makedirs("data", exist_ok=True)
    with open(f"data/{today_str}.txt", "w", encoding="utf-8") as f: f.write(final_content)

        # Phase 5: HTML Generation
    print("\n--- Phase 5: HTML Generation ---")
    html_prompt_url = os.getenv("HTML_PROMPT_URL")
    html_prompt_content = ""
    if html_prompt_url: 
        html_prompt_content = fetch_prompt_from_github(html_prompt_url)
    else:
        try:
            with open("prompt/html.txt", "r", encoding="utf-8") as f: html_prompt_content = f.read()
        except FileNotFoundError:
            try:
                with open("html.txt", "r", encoding="utf-8") as f: html_prompt_content = f.read()
            except FileNotFoundError: pass

    if html_prompt_content:
        print(f"Generating HTML using model: {g_model}...")
        max_html_retries = 3
        for attempt in range(max_html_retries):
            try:
                response = g_client.models.generate_content(model=g_model, contents=[f"{html_prompt_content}\n\n{final_content}"])
                html_code = response.text
                
                # Markdown 코드 블록(```html)이 섞여 나올 경우를 대비한 텍스트 정제 로직
                if "```html" in html_code:
                    html_code = html_code.split("```html")[1].split("```")[0].strip()
                elif "```" in html_code:
                    html_code = html_code.split("```")[1].strip()

                # 최종 index.html 파일 저장
                with open("index.html", "w", encoding="utf-8") as f: 
                    f.write(html_code)
                    
                print("✅ HTML generation complete. Saved to index.html")
                break
                
            except Exception as e:
                print(f"⚠️ Phase 5 attempt {attempt + 1} failed: {e}")
                if attempt < max_html_retries - 1: 
                    time.sleep(15)
                else: 
                    print("❌ All HTML generation attempts failed.")
