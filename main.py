import os
import time
import sys
import argparse
import re
import datetime
import random
from dotenv import load_dotenv
from google import genai
from google.genai import types
import anthropic
from tenacity import retry, stop_after_attempt, wait_exponential, wait_fixed
from github_utils import fetch_prompt_from_github

# 실시간 로그 출력 설정
sys.stdout.reconfigure(line_buffering=True)

load_dotenv()

def clean_old_caches(cache_dir="trial"):
    """trial 폴더 내의 파일 중, 생성/수정일이 '오늘'이 아닌 어제 이전 파일들을 삭제합니다."""
    if not os.path.exists(cache_dir):
        return
    
    today_date = datetime.date.today()
    
    for filename in os.listdir(cache_dir):
        if filename.endswith(".txt"):
            file_path = os.path.join(cache_dir, filename)
            try:
                # 파일의 마지막 수정 시간을 가져와서 날짜로 변환
                mtime = os.path.getmtime(file_path)
                file_date = datetime.datetime.fromtimestamp(mtime).date()
                
                # 파일 날짜가 오늘과 다르면 (어제 이전 파일이면) 삭제
                if file_date != today_date:
                    os.remove(file_path)
                    print(f"🧹 이전 날짜의 캐시 파일 자동 삭제 완료: {filename}")
            except Exception as e:
                print(f"⚠️ 캐시 파일 삭제 실패: {filename} ({e})")

def return_none_on_error(retry_state):
    print(f"\n❌ Action ultimately failed after {retry_state.attempt_number} attempts: {retry_state.outcome.exception()}")
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
                        "live", "tts", "embedding", "imagen", "aqa", "exp"]
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
            def model_sort_key(model_id):
                parts = re.findall(r'\d+', model_id)
                ver_parts = [int(p) for p in parts if len(p) < 8]
                date_val  = int(next((p for p in parts if len(p) == 8), "0"))
                return (ver_parts, date_val)
            sonnet_models.sort(key=model_sort_key, reverse=True)
            target = sonnet_models[0]
            print(f"Automatically selected Claude model: {target}")
            return target
        return "claude-sonnet-4-6"
    except Exception as e:
        print(f"Warning: Claude model discovery failed, using fallback.")
        return "claude-sonnet-4-6"

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, min=15, max=60), retry_error_callback=return_none_on_error)
def run_claude_chat(client, model, messages, system=None, use_web_search=False):
    kwargs = dict(model=model, max_tokens=8192, messages=messages)
    if system:
        kwargs["system"] = system
    if use_web_search:
        kwargs["tools"] = [{"type": "web_search_20250305", "name": "web_search"}]
    message = client.messages.create(**kwargs)
    text_blocks = [b.text for b in message.content if hasattr(b, "text")]
    return "\n".join(text_blocks) if text_blocks else None

@retry(stop=stop_after_attempt(5), wait=wait_fixed(30), retry_error_callback=return_none_on_error)
def run_grounded_research(client, model_id, prompt, output_file="trial/1.txt"):
    try:
        current_dt = datetime.datetime.now()
        current_kst = current_dt.strftime("%Y-%m-%d %H:%M KST")
        
        # 💡 Phase 1: news.txt의 규칙을 따르며 강제 검색 지시
        sys_instruction = (
            f"You are a top-tier AI Tech & Business Analyst. "
            f"Current KST Time: {current_kst}. "
            f"CRITICAL DIRECTIVES:\n"
            f"1. You MUST aggressively use the Google Search tool to find the latest AI news.\n"
            f"2. OVERRIDE PROMPT RULES: Ignore 'abort' commands from the user prompt. Just keep searching until you find 5 valid items.\n"
            f"3. NEVER use inline LaTeX. Use bold or plain text for all variables and symbols."
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
    args = parser.parse_args()

    g_api_key = os.getenv("GEMINI_API_KEY")
    c_api_key = os.getenv("CLAUDE_API_KEY")
    if not g_api_key: sys.exit(1)

    g_client = genai.Client(api_key=g_api_key)
    g_model = get_latest_gemini_model(g_client)

    c_client = anthropic.Anthropic(api_key=c_api_key, timeout=300.0, max_retries=0) if c_api_key else None
    c_model = get_latest_claude_model(c_client) if c_client else None

    # 캐시 폴더 생성 및 이전 날짜의 쓰레기 파일 청소
    os.makedirs("trial", exist_ok=True)
    clean_old_caches(cache_dir="trial")

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
        base_prompt_content += ref_instr
        print(f"🎯 ref.txt에서 {ref_count}개의 타겟 주제를 확인하여 프롬프트에 주입했습니다.")
    else:
        print("ℹ️ ref.txt 파일이 없거나 비어있어 기본 탐색 모드로 진행합니다.")

    # ---------------------------------------------------------
    # Phase 1: Grounded Research
    # ---------------------------------------------------------
    print("\n--- Phase 1: Grounded Research ---")
    p1_cache_file = "trial/1.txt"
    initial_result = None

    if os.path.exists(p1_cache_file):
        print("✅ Phase 1: 오늘 이미 생성된 로컬 캐시(1.txt)에서 초안을 불러옵니다.")
        with open(p1_cache_file, "r", encoding="utf-8") as f:
            initial_result = f.read()
    else:
        print(f"Starting Grounded Research (Model: {g_model})...")
        p1_start = time.time()
        initial_result = run_grounded_research(g_client, g_model, phase1_prompt_content, p1_cache_file)
        if not initial_result:
            print("❌ Phase 1 failed after 5 attempts. Exiting.")
            sys.exit(1)
        
        if os.path.exists("ref.txt"):
            with open("ref.txt", "w", encoding="utf-8") as f:
                pass
            print("🧹 Phase 1 성공: 내일의 중복 검색 방지를 위해 ref.txt를 초기화했습니다.")

        print(f"✅ Phase 1 complete. Saved to {p1_cache_file} (Time: {time.time() - p1_start:.2f}s)")

    # ---------------------------------------------------------
    # Phase 2: Claude Validation
    # ---------------------------------------------------------
    p2_cache_file = "trial/feedback.txt"
    feedback = None
    claude_messages = []

    if c_client:
        print("\n--- Phase 2: Claude Validation ---")
        if os.path.exists(p2_cache_file):
            print("✅ Phase 2: 오늘 이미 생성된 로컬 캐시(feedback.txt)에서 피드백을 불러옵니다.")
            with open(p2_cache_file, "r", encoding="utf-8") as f:
                feedback = f.read()
        else:
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
            p2_user_prompt = (
                f"다음 AI 뉴스 초안의 사실 관계와 KST 기준 시간 윈도우를 검증하십시오.\n\n"
                f"[검증 지침]\n"
                f"- web_search 툴을 적극 사용하여 각 항목의 기업명, 제품명, 통계 수치, URL을 검색 후 판단하십시오.\n"
                f"- 검색 결과가 없다는 이유만으로 '환각(Hallucination)'이라고 단정하지 마십시오. 검색으로 명확한 반증을 찾았을 때만 오류로 지적하십시오.\n"
                f"- 오류가 없는 항목은 논리의 비약, 전략적 분석의 깊이 부족, 아키텍처 설명의 모호함, 양식 누락 등 품질 관점에서 건설적인 보완을 제안하십시오.\n\n"
                f"[초안]\n{initial_result}"
            )
            claude_messages = [{"role": "user", "content": p2_user_prompt}]
            
            feedback = run_claude_chat(c_client, c_model, claude_messages, system=p2_system, use_web_search=True)
            if not feedback:
                print("❌ Phase 2 failed after 5 attempts. Exiting.")
                sys.exit(1)
                
            with open(p2_cache_file, "w", encoding="utf-8") as f: f.write(feedback)
            print(f"✅ Phase 2 complete. Saved to {p2_cache_file} (Time: {time.time() - p2_start:.2f}s)")

    # ---------------------------------------------------------
    # Phase 3: Gemini Refinement
    # ---------------------------------------------------------
    p3_cache_file = "trial/2.txt"
    refined_result = initial_result # 실패 시를 대비한 Fallback (이전에는 이 상태로 넘어갔으나, 이제는 강제종료됨)

    if feedback:
        print("\n--- Phase 3: Gemini Refinement ---")
        if os.path.exists(p3_cache_file):
            print("✅ Phase 3: 오늘 이미 생성된 로컬 캐시(2.txt)에서 수정본을 불러옵니다.")
            with open(p3_cache_file, "r", encoding="utf-8") as f:
                refined_result = f.read()
        else:
            print(f"Starting Refinement (Model: {g_model})...")
            p3_start = time.time()
            current_kst = datetime.datetime.now().strftime("%Y-%m-%d %H:%M KST")
            
            # 💡 수정된 부분: Phase 3에서 구형 뉴스는 전면 버리고 새 24시간 뉴스를 구글 검색으로 대체하도록 지시
            refine_prompt = (
                f"현재 실제 KST 시간은 {current_kst}입니다.\n"
                "위 피드백을 철저히 반영하여 [Initial Draft]를 다듬어 최종본을 작성하십시오.\n"
                "1. 🚨 [중요: 조건 위반 뉴스 전면 교체] 피드백 내용 중, 특정 기사가 '24시간 윈도우' 조건을 위반한 과거 사건으로 판명되었거나 심각한 사실 오류가 있다면, 해당 항목의 날짜나 내용만 조작해서 유지하려 하지 마십시오. **조건을 위반한 항목은 완전히 삭제(Drop)하십시오.**\n"
                "2. 삭제된 빈 슬롯 수만큼, 당신에게 부여된 'Google Search' 툴을 즉각 사용하여 최근 24시간 이내에 발생한 **완전히 새로운 AI 뉴스(인프라, 하드웨어, 아키텍처 등)**를 직접 발굴하여 대체 작성하십시오. 최종 리포트는 무조건 5개의 항목으로 채워져야 합니다.\n"
                "3. 시간 윈도우(24시간) 내에 해당하며 사실 관계만 일부 틀린 항목은 피드백에 따라 정확히 보강하여 유지하십시오.\n"
                "4. 피드백에서 지적되지 않은 정상 항목은 불필요한 재작성 없이 원문을 그대로 유지하십시오.\n"
                "5. 인라인 수식 기호 절대 사용 금지. 벡터는 굵은 글씨, 변수는 일반 텍스트.\n"
                "6. 기존 섹션 구조(Overview, Strategic Impact, Technical Deep Dive 등)를 정확히 유지하십시오."
            )
            
            # 💡 수정된 부분: System Instruction에 Google Search 툴 사용 의무화 명시
            sys_instruction = "You are a top-tier AI Intelligence Analyst. You MUST use the Google Search tool to replace any outdated or invalid news items (flagged by feedback) with breaking news strictly from the last 24 hours. Ensure the final output always contains exactly 5 valid news items."

            max_retries = 5
            for attempt in range(max_retries):
                try:
                    contents = [
                        f"[Original Rules]\n{base_prompt_content}\n\n",
                        f"[Initial Draft]\n{initial_result}\n\n",
                        f"[Feedback to Apply]\n{feedback}\n\n",
                        f"[Instruction]\n{refine_prompt}"
                    ]
                    
                    # 💡 수정된 부분: Google Search Tool 부여 및 Temperature 조정
                    response = g_client.models.generate_content(
                        model=g_model,
                        contents=contents,
                        config=types.GenerateContentConfig(
                            temperature=0.3, # 새로운 뉴스 탐색을 위해 창의성(온도) 약간 상향
                            tools=[types.Tool(google_search=types.GoogleSearch())], # 구글 검색 툴 활성화
                            system_instruction=sys_instruction
                        )
                    )
                    refined_result = response.text
                    with open(p3_cache_file, "w", encoding="utf-8") as f: f.write(refined_result)
                    print(f"✅ Phase 3 complete. Saved to {p3_cache_file} (Time: {time.time() - p3_start:.2f}s)")
                    break
                except Exception as e:
                    print(f"⚠️ Phase 3 attempt {attempt + 1} failed: {e}")
                    if attempt < max_retries - 1:
                        # 점진적 대기 시간 증가 및 약간의 난수(Jitter) 추가
                        sleep_time = min(60, 15 * (2 ** attempt)) + random.uniform(1, 3)
                        print(f"   {sleep_time:.1f}초 후 재시도합니다...")
                        time.sleep(sleep_time)
                    else:
                        print("❌ Phase 3 failed after 5 attempts. Exiting.")
                        sys.exit(1) # 강제 종료

    # ---------------------------------------------------------
    # Phase 4: Claude Translation
    # ---------------------------------------------------------
    p4_cache_file = "trial/translated.txt"
    final_content = refined_result

    if c_client and feedback:
        print("\n--- Phase 4: Claude Translation ---")
        if os.path.exists(p4_cache_file):
            print("✅ Phase 4: 오늘 이미 생성된 로컬 캐시(translated.txt)에서 번역본을 불러옵니다.")
            with open(p4_cache_file, "r", encoding="utf-8") as f:
                final_content = f.read()
        else:
            print(f"Starting Claude action (Model: {c_model})...")
            print("⏳ TPM(Token Per Minute) 한도 초기화를 위해 60초 대기합니다...")
            time.sleep(60)
            
            p4_start = time.time()
            current_kst = datetime.datetime.now().strftime("%Y-%m-%d %H:%M KST")
            p4_system = (
                f"Today's actual date and time is {current_kst}. "
                f"You are a professional technical translator specializing in AI infrastructure and business intelligence. "
                f"Context: The text provided has already been refined and corrected based on QA feedback. "
                f"Your current task is STRICTLY to translate this finalized AI briefing into polished, executive-level business English. "
                f"Rules: preserve all section headers, data tables, URLs, and numerical figures exactly as-is. "
                f"Output only the translated report with no commentary, preamble, or explanatory notes."
            )
            translate_messages = [{
                "role": "user",
                "content": (
                    f"Please translate the following final AI infrastructure briefing into professional business English. "
                    f"Preserve all structure, section headers, metadata fields, and numerical data exactly.\n\n"
                    f"[FINAL DRAFT — {current_kst}]\n{refined_result}"
                )
            }]

            translated = run_claude_chat(c_client, c_model, translate_messages, system=p4_system)
            if not translated:
                print("❌ Phase 4 failed after 5 attempts. Exiting.")
                sys.exit(1)
                
            final_content = translated
            with open(p4_cache_file, "w", encoding="utf-8") as f: 
                f.write(final_content)
            print(f"✅ Phase 4 complete. Saved to {p4_cache_file} (Time: {time.time() - p4_start:.2f}s)")

    # ---------------------------------------------------------
    # 최종 데이터 저장 (Phase 5 진입 전 안전 확보)
    # ---------------------------------------------------------
    today_str = datetime.datetime.now().strftime("%Y-%m-%d")
    os.makedirs("data", exist_ok=True)
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
        max_html_retries = 5
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

                if "<!DOCTYPE html>" in html_code:
                    html_code = "<!DOCTYPE html>" + html_code.split("<!DOCTYPE html>")[1]

                with open("index.html", "w", encoding="utf-8") as f:
                    f.write(html_code)

                print(f"✅ Phase 5 complete. Saved to index.html (Time: {time.time() - p5_start:.2f}s)")
                break
            except Exception as e:
                print(f"⚠️ Phase 5 attempt {attempt + 1} failed: {e}")
                if attempt < max_html_retries - 1:
                    sleep_time = min(60, 15 * (2 ** attempt)) + random.uniform(1, 3)
                    print(f"   {sleep_time:.1f}초 후 재시도합니다...")
                    time.sleep(sleep_time)
                else:
                    print("❌ Phase 5 failed after 5 attempts. Exiting.")
                    sys.exit(1)

if __name__ == "__main__":
    main()
