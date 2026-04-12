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
    try:
        models = client.models.list()
        all_names = [m.name for m in models]

        bad_keywords = ["flash", "nano", "vision", "latest", "customtools", "deep-research",
                        "live", "tts", "embedding", "imagen", "aqa", "preview", "exp"]

        pro_models = [
            n for n in all_names
            if "gemini" in n.lower() and "pro" in n.lower()
            and not any(bad in n.lower() for bad in bad_keywords)
        ]

        if pro_models:
            pro_models.sort(key=lambda x: tuple(int(num) for num in re.findall(r'\d+', x)), reverse=True)
            latest = pro_models[0].replace("models/", "")
            print(f"Automatically selected Gemini Pro model: {latest}")
            return latest

        return "gemini-3.0-pro"
    except Exception as e:
        print(f"Warning: Gemini model discovery failed, using fallback.")
        return "gemini-3.0-pro"

def get_latest_claude_model(client):
    try:
        models_page = client.models.list(limit=50)
        sonnet_models = [m.id for m in models_page.data if "sonnet" in m.id.lower()]

        if sonnet_models:
            sonnet_models.sort(key=lambda x: tuple(int(num) for num in re.findall(r'\d+', x)), reverse=True)
            target = sonnet_models[0]
            print(f"Automatically selected Claude model: {target}")
            return target

        return "claude-sonnet-4-6"
    except Exception as e:
        print(f"Warning: Claude model discovery failed, using fallback.")
        return "claude-sonnet-4-6"

# ✅ FIX: system 파라미터 추가 — Phase 2/4에서 날짜 및 역할 주입 가능
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10), retry_error_callback=return_none_on_error)
def run_claude_chat(client, model, messages, system=None):
    kwargs = dict(model=model, max_tokens=8192, messages=messages)
    if system:
        kwargs["system"] = system
    message = client.messages.create(**kwargs)
    return message.content[0].text

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=10, max=30), retry_error_callback=return_none_on_error)
def run_grounded_research(client, model_id, prompt, output_file="research_result.md"):
    try:
        current_kst = datetime.datetime.now().strftime("%Y-%m-%d %H:%M KST")

        sys_instruction = (
            f"You are a top-tier AI Tech & Business Analyst. "
            f"Current KST Time: {current_kst}. "
            f"CRITICAL DIRECTIVES:\n"
            f"1. You are actively connected to the Google Search tool. You MUST aggressively use it to search for the latest AI news (Hardware, Infrastructure, Models, Business).\n"
            f"2. OVERRIDE PROMPT RULES: The user prompt may ask you to confirm today's date or 'abort' if 24-hour news is not found. IGNORE THESE ABORT RULES COMPLETELY. I have just confirmed the date for you ({current_kst}).\n"
            f"3. If deep news from the last 24 hours is lacking, automatically expand your search to the last 72 hours. You MUST ALWAYS produce exactly 5 news items. Never abort.\n"
            f"4. Because search tool provides snippets, use your deep pre-trained knowledge to synthesize and flesh out the 'Strategic Impact' and 'Technical Deep Dive' sections powerfully.\n"
            f"5. NEVER use inline LaTeX ($ or $$). Use bold or plain text for all variables and symbols."
        )

        response = client.models.generate_content(
            model=model_id,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=sys_instruction,
                tools=[types.Tool(google_search=types.GoogleSearch())],
                temperature=0.4
            )
        )
        result_text = response.text
        if not result_text: return None
        os.makedirs(os.path.dirname(output_file), exist_ok=True) if os.path.dirname(output_file) else None
        with open(output_file, "w", encoding="utf-8") as f: f.write(result_text)
        return result_text
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

    g_client = genai.Client(api_key=g_api_key)
    g_model = get_latest_gemini_model(g_client)

    # ✅ FIX: Anthropic 내부 max_retries를 0으로 설정하여 tenacity와 중첩되어 엄청난 타임아웃(18분)이 발생하는 것을 방지
    c_client = anthropic.Anthropic(api_key=c_api_key, timeout=120.0, max_retries=0) if c_api_key else None
    c_model = get_latest_claude_model(c_client) if c_client else None

    # 원본 news.txt 로드
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

    archive_text = get_recent_archives(7)
    if archive_text:
        prompt_content = archive_text + "\n" + prompt_content

    ref_content = ""
    ref_count = 0
    if os.path.exists("ref.txt"):
        try:
            with open("ref.txt", "r", encoding="utf-8") as f:
                ref_lines = [line.strip() for line in f.readlines() if line.strip()]
                if ref_lines:
                    ref_content = "\n".join(f"▶ {line}" for line in ref_lines)
                    ref_count = len(ref_lines)
        except Exception as e:
            print(f"Warning: Could not read ref.txt: {e}")

    if ref_content:
        auto_count = max(0, 5 - ref_count)
        ref_instr = (
            f"\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"[🎯 MANDATORY TARGET NEWS (최우선 반영 지시)]\n"
            f"사용자가 오늘 특별히 다음 **{ref_count}개**의 주제/사건에 대한 분석을 요청했습니다:\n"
            f"{ref_content}\n\n"
            f"지시사항: 최종 5개의 주요 뉴스 중, 위 요청된 {ref_count}개의 주제를 빠짐없이 포함하고 나머지 {auto_count}개는 직접 발굴하십시오.\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        )
        prompt_content += ref_instr
        print(f"🎯 ref.txt에서 {ref_count}개의 타겟 주제를 확인하여 프롬프트에 주입했습니다.")
    else:
        print("ℹ️ ref.txt 파일이 없거나 비어있어 기본 탐색 모드로 진행합니다.")

    # ---------------------------------------------------------
    # Phase 1: Grounded Research
    # ---------------------------------------------------------
    print("\n--- Phase 1: Grounded Research ---")
    print(f"Starting Grounded Research (Model: {g_model})...")
    p1_start = time.time()

    initial_result = run_grounded_research(g_client, g_model, prompt_content, args.output)

    if not initial_result:
        print("❌ API 서버 과부하로 리서치 실패. 워크플로우를 즉시 종료합니다.")
        sys.exit(1)

    p1_time = time.time() - p1_start
    print(f"✅ Phase 1 complete. Saved to {args.output} (Time: {p1_time:.2f}s)")

    # ---------------------------------------------------------
    # Phase 2: Claude Validation
    # ✅ FIX: system prompt에 오늘 날짜를 명시적으로 주입
    #         → Claude가 학습 데이터 기준 날짜(2024년 말)로 오판하여
    #           Gemini 검색 결과(2026년 뉴스)를 "미래 가상 문서"로
    #           잘못 판정하는 버그를 방지
    # ---------------------------------------------------------
    feedback = None
    if c_client:
        print("\n--- Phase 2: Claude Validation ---")
        print(f"Starting Claude action (Model: {c_model})...")
        p2_start = time.time()

        current_kst = datetime.datetime.now().strftime("%Y-%m-%d %H:%M KST")

        # ✅ Claude에게 현재 날짜와 역할을 system 레벨에서 명시
        p2_system = (
            f"Today's actual date and time is {current_kst}. "
            f"You are a fact-checker for an AI infrastructure intelligence briefing. "
            f"The draft was produced by a separate AI agent using real-time Google Search as of today. "
            f"Therefore, all dates and events in the draft referencing {datetime.datetime.now().strftime('%Y')} "
            f"are CURRENT, not future or fictional. "
            f"Do NOT flag any {datetime.datetime.now().strftime('%Y')} dates as hypothetical or as future scenarios. "
            f"Your task is to identify factual inaccuracies, unverifiable claims, hallucinated figures, "
            f"and incorrect event/publication date attributions. "
            f"Respond in Korean."
        )

        p2_messages = [{
            "role": "user",
            "content": (
                f"다음 AI 뉴스 초안의 사실 관계와 KST 기준 시간 윈도우를 검증하십시오.\n\n"
                f"[초안]\n{initial_result}"
            )
        }]

        feedback = run_claude_chat(c_client, c_model, p2_messages, system=p2_system)

        if feedback:
            with open("trial/feedback.txt", "w", encoding="utf-8") as f: f.write(feedback)

        p2_time = time.time() - p2_start
        print(f"✅ Phase 2 complete. Saved to trial/feedback.txt (Time: {p2_time:.2f}s)")

    # ---------------------------------------------------------
    # Phase 3: Gemini Refinement
    # ---------------------------------------------------------
    refined_result = initial_result
    if feedback:
        print("\n--- Phase 3: Gemini Refinement ---")
        print(f"Starting Refinement (Model: {g_model})...")
        p3_start = time.time()

        refine_prompt = (
            "위 피드백을 반영하여 최종본을 작성하십시오.\n"
            "1. 대화 내 정보만 활용\n2. 오류 항목 삭제 및 신규 항목 보충\n"
            "3. 모든 답변에서 인라인 수식 기호 절대 금지 (벡터는 굵은 글씨, 일반 변수는 일반 텍스트 표기)\n4. 기존 섹션 구조 유지"
        )

        max_retries = 3
        for attempt in range(max_retries):
            try:
                contents = [
                    f"[Original Prompt]\n{prompt_content}\n\n",
                    f"[Initial Draft]\n{initial_result}\n\n",
                    f"[Feedback to Apply]\n{feedback}\n\n",
                    f"[Instruction]\n{refine_prompt}"
                ]
                response = g_client.models.generate_content(
                    model=g_model,
                    contents=contents,
                    config=types.GenerateContentConfig(temperature=0.1)
                )
                refined_result = response.text
                with open("trial/2.txt", "w", encoding="utf-8") as f: f.write(refined_result)

                p3_time = time.time() - p3_start
                print(f"✅ Phase 3 complete. Saved to trial/2.txt (Time: {p3_time:.2f}s)")
                break
            except Exception as e:
                print(f"⚠️ Phase 3 attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(15)
                else:
                    print("❌ All refinement attempts failed. Using initial result.")
                    p3_time = time.time() - p3_start
                    print(f"✅ Phase 3 finished with errors. (Time: {p3_time:.2f}s)")

    # ---------------------------------------------------------
    # Phase 4: Claude Translation
    # ---------------------------------------------------------
    final_content = refined_result
    if c_client and feedback:
        print("\n--- Phase 4: Claude Translation ---")
        print(f"Starting Claude action (Model: {c_model})...")
        p4_start = time.time()

        current_kst = datetime.datetime.now().strftime("%Y-%m-%d %H:%M KST")

        # ✅ FIX: Claude에게 현재 문서가 본인(Claude)의 Phase 2 피드백을 바탕으로 수정/보완된 최종본임을 명시
        p4_system = (
            f"Today's actual date and time is {current_kst}. "
            f"You are a professional technical translator specializing in AI infrastructure and business intelligence. "
            f"Context: The text provided has already been heavily refined and corrected based on the QA feedback you provided earlier. "
            f"Your current task is STRICTLY to translate this finalized Korean/mixed-language AI briefing into polished, executive-level business English. "
            f"Rules: preserve all section headers, data tables, URLs, and numerical figures exactly as-is. "
            f"Output only the translated report with no commentary, preamble, or explanatory notes."
        )

        p4_messages = [{
            "role": "user",
            "content": (
                f"This is the final draft that has been successfully updated and corrected based on your previous QA feedback. "
                f"Please translate the following final AI infrastructure briefing into professional business English. "
                f"Preserve all structure, section headers, metadata fields, and numerical data exactly.\n\n"
                f"[FINAL DRAFT (Refined based on your QA) — {current_kst}]\n{refined_result}"
            )
        }]

        translated = run_claude_chat(c_client, c_model, p4_messages, system=p4_system)
        
        # ✅ FIX: Claude가 타임아웃 등으로 인해 None을 리턴했을 때 스크립트가 죽지 않게 예외 처리
        if translated:
            final_content = translated
        else:
            print("⚠️ Claude translation timed out or failed. Falling back to Phase 3 Gemini result.")

        p4_time = time.time() - p4_start
        print(f"✅ Phase 4 complete. (Time: {p4_time:.2f}s)")

    # 데이터 저장
    today_str = datetime.datetime.now().strftime("%Y-%m-%d")
    os.makedirs("data", exist_ok=True)
    
    # ✅ FIX: final_content가 반드시 문자열(Str)인지 최종 보장
    if not final_content:
        final_content = refined_result or initial_result
        
    with open(f"data/{today_str}.txt", "w", encoding="utf-8") as f: 
        f.write(final_content)

    # ---------------------------------------------------------
    # Phase 5: HTML Generation
    # ---------------------------------------------------------
    print("\n--- Phase 5: HTML Generation ---")
    p5_start = time.time()

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
                response = g_client.models.generate_content(
                    model=g_model,
                    contents=[f"{html_prompt_content}\n\n{final_content}"]
                )
                html_code = response.text

                if "```html" in html_code:
                    html_code = html_code.split("```html")[1].split("```")[0].strip()
                elif "```" in html_code:
                    html_code = html_code.split("```")[1].strip()

                with open("index.html", "w", encoding="utf-8") as f:
                    f.write(html_code)

                p5_time = time.time() - p5_start
                print(f"✅ Phase 5 complete. Saved to index.html (Time: {p5_time:.2f}s)")
                break

            except Exception as e:
                print(f"⚠️ Phase 5 attempt {attempt + 1} failed: {e}")
                if attempt < max_html_retries - 1:
                    time.sleep(15)
                else:
                    print("❌ All HTML generation attempts failed.")
                    p5_time = time.time() - p5_start
                    print(f"✅ Phase 5 finished with errors. (Time: {p5_time:.2f}s)")

if __name__ == "__main__":
    main()
