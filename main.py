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

# ✅ [내 Fix] system 파라미터 + web_search 옵션 유지
# ✅ [내 Fix] web_search 사용 시 multi-block 응답에서 text만 추출하도록 처리
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10), retry_error_callback=return_none_on_error)
def run_claude_chat(client, model, messages, system=None, use_web_search=False):
    kwargs = dict(model=model, max_tokens=8192, messages=messages)
    if system:
        kwargs["system"] = system
    if use_web_search:
        kwargs["tools"] = [{"type": "web_search_20250305", "name": "web_search"}]
    message = client.messages.create(**kwargs)
    # web_search 사용 시 tool_use 블록이 섞이므로 text 블록만 추출
    text_blocks = [b.text for b in message.content if hasattr(b, "text")]
    return "\n".join(text_blocks) if text_blocks else None

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

    # ✅ [원본 Fix] max_retries=0: tenacity와 중첩 타임아웃(18분) 방지
    c_client = anthropic.Anthropic(api_key=c_api_key, timeout=120.0, max_retries=0) if c_api_key else None
    c_model = get_latest_claude_model(c_client) if c_client else None

    # ✅ [Gemini Fix] base_prompt와 phase1_prompt 분리
    # base_prompt_content: 순수 규칙/지시만 포함 → Phase 3/4에 전달
    # phase1_prompt_content: 아카이브 포함 → Phase 1에만 전달
    url = args.url or os.getenv("PROMPT_URL")
    base_prompt_content = ""
    if url:
        base_prompt_content = fetch_prompt_from_github(url)
    else:
        try:
            with open("prompt/news.txt", "r", encoding="utf-8") as f: base_prompt_content = f.read()
        except FileNotFoundError:
            try:
                with open("news.txt", "r", encoding="utf-8") as f: base_prompt_content = f.read()
            except FileNotFoundError:
                sys.exit(1)

    archive_text = get_recent_archives(7)
    # Phase 1 전용: 중복 방지 아카이브 포함
    phase1_prompt_content = (archive_text + "\n" + base_prompt_content) if archive_text else base_prompt_content

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
        phase1_prompt_content += ref_instr
        base_prompt_content += ref_instr  # ref는 Phase 3용 base에도 포함
        print(f"🎯 ref.txt에서 {ref_count}개의 타겟 주제를 확인하여 프롬프트에 주입했습니다.")
    else:
        print("ℹ️ ref.txt 파일이 없거나 비어있어 기본 탐색 모드로 진행합니다.")

    # ---------------------------------------------------------
    # Phase 1: Grounded Research
    # ---------------------------------------------------------
    print("\n--- Phase 1: Grounded Research ---")
    print(f"Starting Grounded Research (Model: {g_model})...")
    p1_start = time.time()

    # Phase 1은 아카이브가 포함된 phase1_prompt_content 사용
    initial_result = run_grounded_research(g_client, g_model, phase1_prompt_content, args.output)

    if not initial_result:
        print("❌ Phase 1 failed. Exiting.")
        sys.exit(1)

    p1_time = time.time() - p1_start
    print(f"✅ Phase 1 complete. Saved to {args.output} (Time: {p1_time:.2f}s)")

    # ---------------------------------------------------------
    # Phase 2: Claude Validation
    # ✅ [내 Fix] use_web_search=True: 실시간 검색으로 2026년 신규 사건 검증 가능
    # ✅ [Gemini Fix] guardrail 프롬프트: 설령 웹 검색 결과가 없더라도 단정적 삭제 지시 방지
    # ✅ [원본 Fix] system 파라미터로 날짜/역할 주입
    # ---------------------------------------------------------
    feedback = None
    claude_messages = []
    if c_client:
        print("\n--- Phase 2: Claude Validation ---")
        print(f"Starting Claude action (Model: {c_model})...")
        p2_start = time.time()

        current_kst = datetime.datetime.now().strftime("%Y-%m-%d %H:%M KST")

        p2_system = (
            f"Today's actual date and time is {current_kst}. "
            f"You are a fact-checker for an AI infrastructure intelligence briefing. "
            f"The draft was produced by a separate AI agent using real-time Google Search as of today. "
            f"Therefore, all dates and events in the draft referencing {datetime.datetime.now().strftime('%Y')} "
            f"are CURRENT, not future or fictional. "
            f"Do NOT flag any {datetime.datetime.now().strftime('%Y')} dates as hypothetical or future scenarios. "
            f"You have access to the web_search tool — use it aggressively to verify company names, "
            f"product specs, statistics, and URLs before forming any judgment. "
            f"Only flag an item as an error after attempting a web search and finding a clear contradiction. "
            f"Respond in Korean."
        )

        # ✅ [Gemini Fix] guardrail: 웹 검색으로 확인 안 되는 항목을 환각으로 단정하지 말도록 명시
        p2_user_prompt = (
            f"다음 AI 뉴스 초안의 사실 관계와 KST 기준 시간 윈도우를 검증하십시오.\n\n"
            f"[검증 지침]\n"
            f"- web_search 툴을 적극 사용하여 각 항목의 기업명, 제품명, 통계 수치, URL을 검색 후 판단하십시오.\n"
            f"- 검색 결과가 없다는 이유만으로 '환각(Hallucination)'이라고 단정하지 마십시오. "
            f"검색으로 명확한 반증을 찾았을 때만 오류로 지적하십시오.\n"
            f"- 오류가 없는 항목은 논리의 비약, 전략적 분석의 깊이 부족, 아키텍처 설명의 모호함, "
            f"양식 누락 등 품질 관점에서 건설적인 보완을 제안하십시오.\n\n"
            f"[초안]\n{initial_result}"
        )

        claude_messages = [{"role": "user", "content": p2_user_prompt}]

        # ✅ [내 Fix] use_web_search=True로 Claude가 직접 실시간 검색 수행
        feedback = run_claude_chat(c_client, c_model, claude_messages,
                                   system=p2_system, use_web_search=True)

        if feedback:
            claude_messages.append({"role": "assistant", "content": feedback})
            with open("trial/feedback.txt", "w", encoding="utf-8") as f: f.write(feedback)

        p2_time = time.time() - p2_start
        print(f"✅ Phase 2 complete. Saved to trial/feedback.txt (Time: {p2_time:.2f}s)")

    # ---------------------------------------------------------
    # Phase 3: Gemini Refinement
    # ✅ [Gemini Fix] base_prompt_content 사용: 아카이브 제거로 어제 기사 복사 방지
    # ✅ [두 Fix 합산] refine_prompt: 삭제 지시에도 항목 유지 + 지적된 부분만 수정
    # ---------------------------------------------------------
    refined_result = initial_result
    if feedback:
        print("\n--- Phase 3: Gemini Refinement ---")
        print(f"Starting Refinement (Model: {g_model})...")
        p3_start = time.time()

        refine_prompt = (
            "위 피드백을 반영하여 [Initial Draft]를 다듬어 최종본을 작성하십시오.\n"
            "1. [Initial Draft]의 5개 뉴스 항목 구조를 기본으로 유지할 것. "
            "피드백이 특정 항목을 삭제하라고 하더라도, 완전히 제거하지 말고 해당 오류 부분만 수정·보강하여 유지하십시오.\n"
            "2. 피드백에서 지적되지 않은 항목은 원문 그대로 유지하십시오. 불필요한 재작성을 금지합니다.\n"
            "3. 인라인 수식 기호($ 또는 $$) 절대 사용 금지. 벡터는 굵은 글씨, 변수는 일반 텍스트.\n"
            "4. 기존 섹션 구조(Overview, Strategic Impact, Technical Deep Dive 등)를 정확히 유지하십시오."
        )

        max_retries = 3
        for attempt in range(max_retries):
            try:
                # ✅ [Gemini Fix] base_prompt_content: 아카이브 없는 순수 규칙만 전달
                contents = [
                    f"[Original Rules]\n{base_prompt_content}\n\n",
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
    # ✅ [Gemini Fix] multi-turn 히스토리: Claude가 자신의 Phase 2 피드백 맥락을 유지한 채 번역
    # ✅ [원본 Fix] system 파라미터로 번역 역할 명시
    # ✅ [원본 Fix] translated가 None이면 refined_result로 fallback
    # ---------------------------------------------------------
    final_content = refined_result
    if c_client and feedback:
        print("\n--- Phase 4: Claude Translation ---")
        print(f"Starting Claude action (Model: {c_model})...")
        p4_start = time.time()

        current_kst = datetime.datetime.now().strftime("%Y-%m-%d %H:%M KST")

        p4_system = (
            f"Today's actual date and time is {current_kst}. "
            f"You are a professional technical translator specializing in AI infrastructure and business intelligence. "
            f"Context: The text provided has already been refined and corrected based on the QA feedback you provided earlier. "
            f"Your current task is STRICTLY to translate this finalized AI briefing into polished, executive-level business English. "
            f"Rules: preserve all section headers, data tables, URLs, and numerical figures exactly as-is. "
            f"Output only the translated report with no commentary, preamble, or explanatory notes."
        )

        # ✅ [Gemini Fix] Phase 2 대화 히스토리에 이어서 append → Claude가 자신의 이전 피드백을 인지한 채 번역
        claude_messages.append({
            "role": "user",
            "content": (
                f"This is the final draft that has been successfully updated and corrected based on your previous QA feedback. "
                f"Please translate the following final AI infrastructure briefing into professional business English. "
                f"Preserve all structure, section headers, metadata fields, and numerical data exactly.\n\n"
                f"[FINAL DRAFT (Refined based on your QA) — {current_kst}]\n{refined_result}"
            )
        })

        translated = run_claude_chat(c_client, c_model, claude_messages, system=p4_system)

        # ✅ [원본 Fix] None 리턴 시 refined_result로 fallback
        if translated:
            final_content = translated
        else:
            print("⚠️ Claude translation timed out or failed. Falling back to Phase 3 Gemini result.")

        p4_time = time.time() - p4_start
        print(f"✅ Phase 4 complete. (Time: {p4_time:.2f}s)")

    # 데이터 저장
    today_str = datetime.datetime.now().strftime("%Y-%m-%d")
    os.makedirs("data", exist_ok=True)

    # ✅ [원본 Fix] final_content가 None/빈 값일 경우 최종 안전망
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
