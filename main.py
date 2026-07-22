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
from openai import OpenAI
import anthropic  # 🛠️ [추가] Claude 사용을 위한 패키지 임포트
from tenacity import retry, stop_after_attempt, wait_exponential, wait_fixed
from github_utils import fetch_prompt_from_github

# 실시간 로그 출력 설정
sys.stdout.reconfigure(line_buffering=True)

load_dotenv()

def clean_old_caches(cache_dir="trial"):
    if not os.path.exists(cache_dir):
        return
    
    today_date = datetime.date.today()
    
    for filename in os.listdir(cache_dir):
        if filename.endswith(".txt"):
            file_path = os.path.join(cache_dir, filename)
            try:
                mtime = os.path.getmtime(file_path)
                file_date = datetime.datetime.fromtimestamp(mtime).date()
                
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
        bad_keywords = ["nano", "vision", "latest", "customtools", "deep-research",
                        "live", "tts", "embedding", "imagen", "aqa", "exp"]
        flash_models = [
            n for n in all_names
            if "gemini" in n.lower() and "flash" in n.lower()
            and not any(bad in n.lower() for bad in bad_keywords)
        ]
        if flash_models:
            def model_sort_key(model_name):
                parts = re.findall(r'\d+\.?\d*', model_name)
                version_parts = []
                for p in parts:
                    if '.' in p:
                        try:
                            version_parts.extend([int(x) for x in p.split('.')])
                        except ValueError:
                            pass
                    else:
                        try:
                            version_parts.append(int(p))
                        except ValueError:
                            pass
                return tuple(version_parts)

            flash_models.sort(key=model_sort_key, reverse=True)
            latest = flash_models[0].replace("models/", "")
            print(f"Automatically selected Gemini Flash model: {latest}")
            return latest
        return "gemini-3.5-flash"
    except Exception as e:
        print(f"Warning: Gemini model discovery failed, using fallback.")
        return "gemini-3.5-flash"

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, min=15, max=60), retry_error_callback=return_none_on_error)
def run_gpt_chat(client, model, messages, system=None):
    api_messages = []
    if system:
        api_messages.append({"role": "system", "content": system})
    api_messages.extend(messages)
    
    response = client.chat.completions.create(
        model=model,
        messages=api_messages
    )
    return response.choices[0].message.content

# 🛠️ [추가] Claude API 호출을 위한 전용 래퍼 함수 (HTML 생성을 위해 max_tokens 여유롭게 설정)
@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, min=15, max=60), retry_error_callback=return_none_on_error)
def run_claude_chat(client, model, messages, system=None, temperature=None, validate_fn=None):
    kwargs = {
        "model": model,
        # 🛡️ [수정] Sonnet 5는 max_tokens가 thinking+응답 텍스트를 합친 하드 리밋임(최대 128,000).
        # CSS/스켈레톤을 Python 템플릿으로 분리한 뒤에는 Claude가 기사 콘텐츠만 생성하므로 이 정도면 충분.
        "max_tokens": 16000,
        "messages": messages,
        # 🛡️ [수정] 이 작업은 정해진 템플릿/CSS에 데이터를 정확히 채워넣는 포맷팅 작업이라
        # 별도의 다단계 추론이 필요 없음. adaptive thinking을 꺼서 보이지 않는 thinking 토큰이
        # max_tokens 예산과 비용을 낭비하지 않도록 함 (결과물 품질에는 영향 없음).
        "thinking": {"type": "disabled"},
    }
    if system:
        kwargs["system"] = system
    # 🛡️ [수정] Claude Sonnet 5는 temperature/top_p/top_k를 기본값 외로 지정하면
    # 400 에러(`temperature` is deprecated for this model)를 반환함 (adaptive thinking 기본 적용).
    # 필요한 경우에만(구형 모델 등) 명시적으로 전달하도록 옵션화.
    if temperature is not None:
        kwargs["temperature"] = temperature

    response = client.messages.create(**kwargs)

    # 🛡️ [수정] Claude Sonnet 5는 adaptive thinking이 기본 활성화되어 있어,
    # content[0]이 텍스트가 아니라 ThinkingBlock(추론 과정)일 수 있음.
    # content 블록들을 순회하며 실제 text 타입 블록만 이어붙여야 함.
    text_parts = [block.text for block in response.content if getattr(block, "type", None) == "text"]
    if not text_parts:
        raise ValueError("Claude 응답에 text 블록이 없습니다 (ThinkingBlock만 반환됨).")
    result = "".join(text_parts)

    # 🛡️ [수정] 완성도 체크를 재시도 데코레이터 "안쪽"에서 수행.
    # 바깥에서 체크하면 API 호출 자체는 성공으로 끝나버려 재시도가 아예 발생하지 않음(1회 시도 후 즉시 종료되는 버그였음).
    # validate_fn이 실패 시 예외를 던지면 @retry가 지수 백오프로 실제 재시도를 수행함.
    if validate_fn:
        validate_fn(result)

    return result


def _strip_markdown_fence(text):
    """```html ... ``` 또는 ``` ... ``` 마크다운 래퍼를 제거."""
    if "```html" in text:
        return text.split("```html")[1].split("```")[0].strip()
    elif "```" in text:
        return text.split("```")[1].strip()
    return text


def validate_articles_content(text):
    """🛡️ [추가] Phase 5용 검증 함수. 이제 Claude는 전체 문서가 아니라
    <main> 안에 들어갈 5개의 <article> 블록만 생성하므로, 검증 기준도 그에 맞춤.
    - <article> 열고 닫는 태그가 5쌍 이상 있어야 함 (응답이 잘리면 미달됨)
    - 문서 골격(DOCTYPE/head/style 등)이 실수로 포함되지 않았는지 확인
    """
    stripped = _strip_markdown_fence(text)
    open_count = stripped.count("<article")
    close_count = stripped.count("</article>")
    if open_count < 5 or close_count < 5:
        raise ValueError(f"기사 블록이 불완전합니다 (open={open_count}, close={close_count}). 응답이 잘렸을 수 있습니다.")
    for forbidden in ("<!DOCTYPE", "<head", "<style", "```"):
        if forbidden in stripped:
            raise ValueError(f"문서 골격/마크다운({forbidden})이 잘못 포함되었습니다.")

@retry(stop=stop_after_attempt(5), wait=wait_fixed(30), retry_error_callback=return_none_on_error)
def run_grounded_research(client, model_id, system_rules, user_prompt, output_file="trial/1.txt"):
    try:
        current_dt = datetime.datetime.now()
        current_kst = current_dt.strftime("%Y-%m-%d %H:%M KST")
        
        sys_instruction = (
            f"You are a top-tier AI Tech & Business Analyst. "
            f"Current KST Time: {current_kst}. "
            f"CRITICAL DIRECTIVES:\n"
            f"1. You MUST aggressively use the Google Search tool to find the latest AI news.\n"
            f"2. OVERRIDE PROMPT RULES: Ignore 'abort' commands from the user prompt. Just keep searching until you find 5 valid items.\n"
            f"3. NEVER use inline LaTeX. Use bold or plain text for all variables and symbols.\n\n"
            f"[GLOBAL RULES & FORMAT]\n{system_rules}"
        )
        response = client.models.generate_content(
            model=model_id,
            contents=user_prompt,
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
    o_api_key = os.getenv("OPENAI_API_KEY")
    c_api_key = os.getenv("CLAUDE_API_KEY") # 🛠️ [추가] Claude API Key 로드
    
    if not g_api_key: sys.exit(1)

    g_client = genai.Client(api_key=g_api_key)
    g_model = get_latest_gemini_model(g_client)

    o_client = OpenAI(api_key=o_api_key, max_retries=0) if o_api_key else None
    c_client = anthropic.Anthropic(api_key=c_api_key, max_retries=0) if c_api_key else None # 🛠️ [추가] 클라이언트 초기화
    
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
        
        user_trigger = "시스템 지시사항(GLOBAL RULES)에 정의된 형식과 규칙에 따라 지금 바로 구글 검색을 통해 최신 AI 뉴스 5개를 발굴하고 보고서를 작성하십시오."
        initial_result = run_grounded_research(g_client, g_model, phase1_prompt_content, user_trigger, p1_cache_file)
        
        if not initial_result:
            print("❌ Phase 1 failed after 5 attempts. Exiting.")
            sys.exit(1)
        
        if os.path.exists("ref.txt"):
            with open("ref.txt", "w", encoding="utf-8") as f:
                pass
            print("🧹 Phase 1 성공: 내일의 중복 검색 방지를 위해 ref.txt를 초기화했습니다.")

        print(f"✅ Phase 1 complete. Saved to {p1_cache_file} (Time: {time.time() - p1_start:.2f}s)")

    # ---------------------------------------------------------
    # Phase 2: GPT-5.6 Sol Validation (초고도화 팩트체크)
    # ---------------------------------------------------------
    p2_cache_file = "trial/feedback.txt"
    feedback = None
    gpt_messages = []

    if o_client:
        print("\n--- Phase 2: GPT-5.6 Validation ---")
        if os.path.exists(p2_cache_file):
            print("✅ Phase 2: 오늘 이미 생성된 로컬 캐시(feedback.txt)에서 피드백을 불러옵니다.")
            with open(p2_cache_file, "r", encoding="utf-8") as f:
                feedback = f.read()
        else:
            print("Starting GPT-5.6 action (Model: gpt-5.6-sol)...")
            p2_start = time.time()
            current_kst = datetime.datetime.now().strftime("%Y-%m-%d %H:%M KST")
            p2_system = (
                f"Today's actual date and time is {current_kst}. "
                f"You are a fact-checker for an AI infrastructure intelligence briefing. "
                f"The draft was produced by a separate AI agent using real-time Google Search as of today. "
                f"Therefore, all dates and events in the draft referencing {datetime.datetime.now().strftime('%Y')} "
                f"are CURRENT, not future or fictional. "
                f"Do NOT flag any {datetime.datetime.now().strftime('%Y')} dates as hypothetical or future scenarios. "
                f"Note: most items must have an event date within the last 24 hours, EXCEPT items whose metadata "
                f"shows slot_category: 6 (HW-Circuit-Research / academic-research items), which are allowed an event "
                f"date up to 7 days old (event_within_7d_window: true). Do not flag those items as stale for that reason alone. "
                f"Use your advanced reasoning capabilities to verify company names, "
                f"product specs, statistics, and URLs before forming any judgment. "
                f"Respond in Korean."
            )
            p2_user_prompt = (
                f"다음 AI 뉴스 초안의 사실 관계와 KST 기준 시간 윈도우를 검증하십시오.\n\n"
                f"[검증 지침]\n"
                f"- 각 항목의 기업명, 제품명, 통계 수치, URL을 엄격하게 교차 검증하십시오.\n"
                f"- 명확한 논리적 모순이나 반증을 찾았을 때만 오류로 지적하십시오.\n"
                f"- 오류가 없는 항목은 논리의 비약, 전략적 분석의 깊이 부족, 아키텍처 설명의 모호함, 양식 누락 등 품질 관점에서 건설적인 보완을 제안하십시오.\n\n"
                f"[초안]\n{initial_result}"
            )
            gpt_messages = [{"role": "user", "content": p2_user_prompt}]
            
            feedback = run_gpt_chat(o_client, "gpt-5.6-sol", gpt_messages, system=p2_system)
            if not feedback:
                print("❌ Phase 2 failed. Exiting.")
                sys.exit(1)
                
            with open(p2_cache_file, "w", encoding="utf-8") as f: f.write(feedback)
            print(f"✅ Phase 2 complete. Saved to {p2_cache_file} (Time: {time.time() - p2_start:.2f}s)")

    # ---------------------------------------------------------
    # Phase 3: Gemini Refinement (품질 제안 흡수 피드백 루프)
    # ---------------------------------------------------------
    p3_cache_file = "trial/2.txt"
    refined_result = initial_result

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
            
            refine_prompt = (
                f"현재 실제 KST 시간은 {current_kst}입니다.\n"
                "위 피드백을 철저히 반영하여 [Initial Draft]를 다듬어 최종본을 작성하십시오.\n"
                "1. 🚨 [조건 위반 및 사실 오류 수정] 피드백 내용 중 기사가 조건을 위반했거나 심각한 사실 오류가 있다면, 해당 항목을 전면 삭제(Drop)하십시오.\n"
                "2. 💡 [품질 보완 제안 적극 반영] 피드백의 '품질 보완 제안(📌 보완 제안)' 파트에 명시된 개선 요청 사항들(누락된 하위 아키텍처 명시, 구체적인 파트너십 브랜드 보완, 교정 명칭 반영 등)을 본문에 완벽히 녹여내어 분석의 깊이를 극대화하십시오.\n"
                "3. 삭제된 빈 슬롯 수만큼, 당신에게 부여된 'Google Search' 툴을 즉각 사용하여 완전히 새로운 AI 인프라/하드웨어 뉴스를 직접 발굴하여 대체 작성하십시오. "
                "일반 슬롯(카테고리 1~5)은 최근 24시간 이내에 발생한 사건만 허용합니다. "
                "단, 삭제된 항목이 카테고리 6(HW-Circuit-Research, 학술연구/미래 로드맵)이었다면, 대체 항목은 GLOBAL RULES에 명시된 대로 최근 7일 이내 발표/게시된 논문·컨퍼런스·기술 블로그를 기준으로 발굴하십시오. 최종 리포트는 무조건 5개의 항목으로 채워져야 합니다.\n"
                "4. 피드백에서 지적되지 않은 정상 항목은 불필요한 재작성 없이 원문을 최대한 유지하십시오.\n"
                "5. 인라인 수식 기호 절대 사용 금지. 벡터는 굵은 글씨, 변수는 일반 텍스트.\n"
                "6. 기존 섹션 구조(Overview, Strategic Impact, Technical Deep Dive 등)를 정확히 유지하십시오."
            )
            
            sys_instruction = (
                "You are a top-tier AI Intelligence Analyst. You MUST use the Google Search tool to replace any outdated "
                "or invalid news items (flagged by feedback) with breaking news strictly from the last 24 hours — "
                "EXCEPT for items whose slot_category metadata is 6 (HW-Circuit-Research), which instead follow a "
                "7-day event-date window as defined in the GLOBAL RULES below. Do not apply the 24-hour cutoff to those items. "
                "Ensure the final output always contains exactly 5 valid news items.\n\n"
                f"[GLOBAL RULES & FORMAT]\n{base_prompt_content}"
            )

            max_retries = 5
            for attempt in range(max_retries):
                try:
                    contents = [
                        f"[Initial Draft]\n{initial_result}\n\n",
                        f"[Feedback to Apply]\n{feedback}\n\n",
                        f"[Instruction]\n{refine_prompt}"
                    ]
                    
                    response = g_client.models.generate_content(
                        model=g_model,
                        contents=contents,
                        config=types.GenerateContentConfig(
                            temperature=0.3,
                            tools=[types.Tool(google_search=types.GoogleSearch())],
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
                        sleep_time = min(60, 15 * (2 ** attempt)) + random.uniform(1, 3)
                        print(f"   {sleep_time:.1f}초 후 재시도합니다...")
                        time.sleep(sleep_time)
                    else:
                        print("❌ Phase 3 failed after 5 attempts. Exiting.")
                        sys.exit(1)

    # ---------------------------------------------------------
    # Phase 4: GPT-5.6 Terra Translation (번역)
    # ---------------------------------------------------------
    p4_cache_file = "trial/translated.txt"
    final_content = refined_result

    if o_client and feedback:
        print("\n--- Phase 4: GPT-5.6 Translation ---")
        if os.path.exists(p4_cache_file):
            print("✅ Phase 4: 오늘 이미 생성된 로컬 캐시(translated.txt)에서 번역본을 불러옵니다.")
            with open(p4_cache_file, "r", encoding="utf-8") as f:
                final_content = f.read()
        else:
            print("Starting GPT-5.6 action (Model: gpt-5.6-terra)...")
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

            translated = run_gpt_chat(o_client, "gpt-5.6-terra", translate_messages, system=p4_system)
            if not translated:
                print("❌ Phase 4 failed. Exiting.")
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
    # Phase 5: HTML Generation (Claude-5-Sonnet 적용)
    # ---------------------------------------------------------
    print("\n--- Phase 5: HTML Generation ---")
    p5_start = time.time()

    # 🛡️ [수정] CSS/문서 골격은 매일 바뀌지 않는 고정 부분이라 Python 템플릿으로 분리.
    # Claude는 이제 <main> 안에 들어갈 기사 콘텐츠만 생성하면 되므로 출력 토큰이 크게 줄어듦.
    template_content = ""
    for path in ("prompt/template.html", "template.html"):
        try:
            with open(path, "r", encoding="utf-8") as f: template_content = f.read()
            break
        except FileNotFoundError:
            continue

    if not template_content or "{{ARTICLES_CONTENT}}" not in template_content:
        print("❌ Phase 5 failed: template.html을 찾을 수 없거나 {{ARTICLES_CONTENT}} 자리표시자가 없습니다.")
        sys.exit(1)

    html_prompt_url = os.getenv("HTML_PROMPT_URL")
    html_prompt_content = ""
    if html_prompt_url:
        html_prompt_content = fetch_prompt_from_github(html_prompt_url)
    else:
        for path in ("prompt/html_content.txt", "html_content.txt"):
            try:
                with open(path, "r", encoding="utf-8") as f: html_prompt_content = f.read()
                break
            except FileNotFoundError:
                continue

    # 🛡️ [방어 로직] html 프롬프트가 없으면 Phase 5를 조용히 건너뛰지 말고 명확히 실패 처리
    # (이전에는 이 경우 아무 에러 없이 스크립트가 종료되어, index.html이 갱신되지 않았는데도
    #  워크플로우가 "성공"으로 표시되는 문제가 있었음)
    if not html_prompt_content:
        print("❌ Phase 5 failed: html_content.txt 프롬프트를 찾을 수 없습니다 (HTML_PROMPT_URL 또는 prompt/html_content.txt 확인 필요).")
        sys.exit(1)

    # 🛠️ [수정] OpenAI가 아닌 Anthropic 클라이언트 존재 여부 체크
    if not c_client:
        print("❌ Phase 5 failed: Anthropic client is required for Claude HTML generation. Please set CLAUDE_API_KEY.")
        sys.exit(1)

    claude_model_id = "claude-sonnet-5"
    print(f"Generating article content using model: {claude_model_id}...")

    user_prompt = f"다음 데이터를 바탕으로 시스템 지시사항에 맞춰 5개 기사(<article>) 블록만 생성하십시오:\n\n{final_content}"
    messages = [{"role": "user", "content": user_prompt}]

    # 🛠️ [수정] validate_fn=validate_articles_content: 응답이 잘리거나 골격이 잘못 섞이면
    # @retry가 실제로 재시도하도록 함
    articles_content = run_claude_chat(
        c_client, claude_model_id, messages,
        system=html_prompt_content, validate_fn=validate_articles_content
    )

    if not articles_content:
        print("❌ Phase 5 failed after retries. Exiting.")
        sys.exit(1)

    articles_content = _strip_markdown_fence(articles_content)

    # 🛡️ [방어 로직] 템플릿에 기사 콘텐츠 + 실제 생성 날짜를 끼워넣어 최종 문서를 Python 쪽에서 조립.
    # CSS/골격은 항상 template.html 그대로이므로 매일 CSS가 미세하게 달라질 위험이 원천 차단됨.
    report_date_str = datetime.datetime.now().strftime("%B %d, %Y")
    html_code = template_content.replace("{{ARTICLES_CONTENT}}", articles_content).replace("{{REPORT_DATE}}", report_date_str)

    # 🛡️ [방어 로직] 최종 안전망 - 조립 결과가 실제로 완전한 문서인지 마지막으로 확인
    if "{{ARTICLES_CONTENT}}" in html_code or "{{REPORT_DATE}}" in html_code or "<!DOCTYPE html>" not in html_code or "</html>" not in html_code:
        print("❌ Phase 5 failed: 최종 HTML 조립에 실패했습니다. index.html을 덮어쓰지 않습니다.")
        sys.exit(1)

    with open("index.html", "w", encoding="utf-8") as f:
        f.write(html_code)

    # 🛡️ [방어 로직] Phase 5가 "오늘 날짜"로 실제 완료됐다는 것을 증명하는 마커 파일.
    # 워크플로우의 "이미 완료됐는지" 체크가 data/*.txt(=Phase 5 이전에 저장됨)가 아니라
    # 이 마커를 기준으로 판단하도록 하여, Phase 5 실패 시 다음 스케줄에서 재시도가 막히지 않게 함.
    with open("data/.html_synced", "w", encoding="utf-8") as f:
        f.write(today_str)

    print(f"✅ Phase 5 complete. Saved to index.html (Time: {time.time() - p5_start:.2f}s)")

if __name__ == "__main__":

    main()
