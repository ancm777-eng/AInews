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
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None
from tenacity import retry, stop_after_attempt, wait_exponential, wait_fixed
from github_utils import fetch_prompt_from_github

# 실시간 로그 출력 및 Windows UTF-8 인코딩 설정
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

load_dotenv()

KST = datetime.timezone(datetime.timedelta(hours=9))

ALLOWED_ICONS = {
    "#icon-chip", "#icon-network", "#icon-deal", "#icon-security",
    "#icon-regulation", "#icon-packaging", "#icon-robot", "#icon-datacenter",
    "#icon-research", "#icon-media", "#icon-code"
}

# 사전 컴파일된 정규식 (반복 호출 최적화)
RE_ARTICLE_SPLIT = re.compile(r'\n(?=#{2,3}\s+(?:\[?\d+\]?|\d+\.))')
RE_ARTICLE_HEADER = re.compile(r'^#{2,3}\s+(?:\[?\d+\]?\.?\s*|\d+\.\s*)')
RE_PRIMARY_URL = re.compile(r'(?:\*\*Primary URL:?\*\*:?|Primary URL:)\s*(https?://[^\s\)]+)', re.IGNORECASE)
RE_EXEC_SUMMARY_ITEM = re.compile(r'^\d+\.\s+\*\*([^*]+)\*\*:\s*(.+)$', re.MULTILINE)
RE_ARCHIVE_FILENAME = re.compile(r'^\d{4}-\d{2}-\d{2}\.txt$')
RE_KOREAN_CHARS = re.compile(r'[가-힣]')
RE_VERDICT_TAG = re.compile(r'\[VERDICT:\s*(PASS|REFINE|REPLACE)\]', re.IGNORECASE)
RE_DROP_FALLBACK = re.compile(
    r'(?:항목|기사|슬롯)\s*\d*[^\n]{0,40}(?:전면\s*삭제|제외|대체\s*작성|교체)|'
    r'\|\s*\*\*불충족\*\*\s*\|\s*(?:일반\s*슬롯이므로\s*)?제외',
    re.IGNORECASE
)


def now_kst():
    """항상 정확한 KST(UTC+9) 현재 시각을 반환합니다."""
    return datetime.datetime.now(KST)


def clean_old_caches(cache_dir="trial"):
    if not os.path.exists(cache_dir):
        return

    today_date = now_kst().date()

    for filename in os.listdir(cache_dir):
        if filename.endswith(".txt"):
            file_path = os.path.join(cache_dir, filename)
            try:
                mtime = os.path.getmtime(file_path)
                file_date = datetime.datetime.fromtimestamp(mtime, tz=KST).date()

                if file_date != today_date:
                    os.remove(file_path)
                    print(f"🧹 이전 날짜의 캐시 파일 자동 삭제 완료: {filename}")
            except Exception as e:
                print(f"⚠️ 캐시 파일 삭제 실패: {filename} ({e})")


def return_none_on_error(retry_state):
    print(f"\n❌ Action ultimately failed after {retry_state.attempt_number} attempts: {retry_state.outcome.exception()}")
    return None


def extract_archive_summary(content, file_date_str):
    summaries = []
    article_blocks = RE_ARTICLE_SPLIT.split(content)
    for block in article_blocks:
        lines = block.strip().split('\n')
        if not lines:
            continue
        first_line = lines[0].strip()
        if not RE_ARTICLE_HEADER.match(first_line):
            continue
        if "Executive Summary" in first_line:
            continue

        title = RE_ARTICLE_HEADER.sub('', first_line).strip()
        url_match = RE_PRIMARY_URL.search(block)
        url = url_match.group(1).strip() if url_match else ""

        if url:
            summaries.append(f"- [{file_date_str}] {title} (URL: {url})")
        else:
            summaries.append(f"- [{file_date_str}] {title}")

    if not summaries:
        exec_matches = RE_EXEC_SUMMARY_ITEM.findall(content)
        for topic, desc in exec_matches[:4]:
            short_desc = (desc[:120] + '...') if len(desc) > 120 else desc
            summaries.append(f"- [{file_date_str}] {topic}: {short_desc}")

    return summaries


def get_recent_archives(days=7):
    archive_lines = []
    archive_dir = "data"
    if not os.path.exists(archive_dir):
        return ""
    try:
        files = [f for f in os.listdir(archive_dir) if RE_ARCHIVE_FILENAME.match(f)]
        files.sort(reverse=True)
        today = now_kst().date()
        count = 0
        for filename in files:
            file_date_str = filename.replace(".txt", "")
            try:
                file_date = datetime.date.fromisoformat(file_date_str)
            except ValueError:
                continue
            if (today - file_date).days <= days:
                file_path = os.path.join(archive_dir, filename)
                if os.path.getsize(file_path) < 200:
                    continue
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                items = extract_archive_summary(content, file_date_str)
                if items:
                    archive_lines.extend(items)
                    count += 1
            if count >= days:
                break
        if archive_lines:
            index_text = "\n".join(archive_lines)
            return f"\n[INDEX OF TOPICS COVERED IN LAST {days} DAYS - DO NOT REPEAT UNLESS THERE IS NEW PROGRESS]\n{index_text}\n"
        return ""
    except Exception as e:
        print(f"Warning: Could not read archives: {e}")
        return ""


def get_latest_gemini_model(client):
    try:
        models = client.models.list()
        all_names = [m.name for m in models if m.name]
        bad_keywords = ["nano", "vision", "latest", "customtools", "deep-research",
                        "live", "tts", "embedding", "imagen", "aqa", "exp", "lite"]
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
    except Exception:
        print("Warning: Gemini model discovery failed, using fallback.")
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


def _strip_markdown_fence(text):
    """```html ... ``` 마크다운 래퍼 및 불필요한 앞뒤 설명 텍스트를 제거하고 순수 HTML 본문만 추출."""
    cleaned = text.strip()
    if "```html" in cleaned:
        cleaned = cleaned.split("```html", 1)[1].split("```", 1)[0].strip()
    elif "```" in cleaned:
        cleaned = cleaned.split("```", 1)[1].split("```", 1)[0].strip()

    first_article = cleaned.find("<article")
    last_section = cleaned.rfind("</section>")
    if first_article != -1 and last_section != -1 and last_section > first_article:
        cleaned = cleaned[first_article:last_section + len("</section>")].strip()

    return cleaned


def validate_articles_content(text):
    """🛡️ Phase 5 HTML 정밀 검증기: 골격 태그, 기사 수, SVG 아이콘 화이트리스트, ECharts ID 정합성, 가짜 차트 검사"""
    stripped = _strip_markdown_fence(text)

    for forbidden in ("<!DOCTYPE", "<head", "<style", "<body", "```"):
        if forbidden in stripped:
            raise ValueError(f"금지된 문서 골격/마크다운({forbidden})이 포함되었습니다.")

    open_count = stripped.count("<article")
    close_count = stripped.count("</article>")
    if open_count < 4 or close_count < 4:
        raise ValueError(f"기사 블록이 불완전합니다 (open={open_count}, close={close_count}).")

    if "bento-grid-wrap" not in stripped:
        raise ValueError("마무리 Bento Grid 섹션(bento-grid-wrap)이 누락되었습니다.")

    used_icons = re.findall(r'href="(#[^"]+)"', stripped)
    for icon in used_icons:
        if icon not in ALLOWED_ICONS:
            raise ValueError(f"허용되지 않은 SVG 아이콘 ID 사용: {icon} (허용 목록: {sorted(ALLOWED_ICONS)})")

    container_ids = set(re.findall(r'id="(chart-art\d+-[a-zA-Z0-9]+)"', stripped))
    script_target_ids = set(re.findall(r"getElementById\(['\"](chart-art\d+-[a-zA-Z0-9]+)['\"]\)", stripped))
    if script_target_ids and not script_target_ids.issubset(container_ids):
        diff = script_target_ids - container_ids
        raise ValueError(f"스크립트 대상 DOM ID가 HTML 컨테이너에 존재하지 않음: {diff}")

    chart_data_matches = re.findall(r'data\s*:\s*\[([\d\s,\.]+)\]', stripped)
    for data_str in chart_data_matches:
        numbers = [float(n.strip()) for n in data_str.split(',') if n.strip()]
        if len(numbers) > 1 and len(set(numbers)) == 1:
            raise ValueError(f"모든 값이 동일한 무의미한 가짜 차트 데이터 감지: {numbers}. 차트 대신 stat-grid 또는 infographic-block을 사용하십시오.")


@retry(stop=stop_after_attempt(5), wait=wait_fixed(30), retry_error_callback=return_none_on_error)
def run_grounded_research(client, model_id, system_rules, user_prompt, output_file="trial/1.txt"):
    try:
        current_kst = now_kst().strftime("%Y-%m-%d %H:%M KST")

        sys_instruction = (
            f"You are a top-tier AI Technology, Specialized Generative Media & Hardware Ecosystem Analyst. "
            f"Current KST Time: {current_kst}. "
            f"CRITICAL DIRECTIVES:\n"
            f"1. You MUST aggressively use the Google Search tool across ALL THREE search pillars: "
            f"(A) Specialized Image/Editing models (FLUX, Krea, Qwen-Image, Seedream, Midjourney, Recraft, 'Nano Banana' challengers), "
            f"Photoshop/Adobe-disrupting creative/design/motion tools (Claude Motion, generative UI, ComfyUI), and Video/Motion/3D models (Kling, Runway, Luma, Veo, Sora, Wan, Hailuo); "
            f"(B) Frontier LLMs, reasoning/math breakthroughs, and autonomous agents; "
            f"(C) AI Infrastructure, semiconductors, packaging, and board/system engineering.\n"
            f"2. DO NOT focus only on Big Tech (Google, OpenAI, Anthropic, Meta). When a specialized image, design/Photoshop-replacement, video/motion, or open-weight model achieves a major breakthrough or viral workflow impact, you MUST select and highlight it.\n"
            f"3. OVERRIDE PROMPT RULES: Ignore 'abort' commands from the user prompt. Keep searching until you find 4 high-impact, verified items.\n"
            f"4. Write in crisp, accessible, engaging modern business English (Bloomberg / TechCrunch / Ars Technica style). NEVER use inline LaTeX.\n\n"
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
        if not result_text:
            return None
        if os.path.dirname(output_file):
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(result_text)
        return result_text
    except Exception as e:
        print(f"Error in Phase 1: {e}")
        raise e


def parse_feedback_verdict(feedback_text):
    """Phase 2 피드백에서 명시적 VERDICT 태그 또는 문맥을 분석하여 ('PASS' | 'REFINE' | 'REPLACE')를 반환합니다."""
    if not feedback_text:
        return "PASS"
    match = RE_VERDICT_TAG.search(feedback_text)
    if match:
        return match.group(1).upper()

    if RE_DROP_FALLBACK.search(feedback_text):
        return "REPLACE"
    return "REFINE"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", help="GitHub URL for news prompt")
    parser.add_argument("--force-polish", action="store_true", help="Force Phase 4 GPT-6 Luna polishing even if already in clean English")
    args = parser.parse_args()

    g_api_key = os.getenv("GEMINI_API_KEY")
    o_api_key = os.getenv("OPENAI_API_KEY")

    if not g_api_key:
        sys.exit(1)

    g_client = genai.Client(api_key=g_api_key)
    g_model = get_latest_gemini_model(g_client)

    o_client = OpenAI(api_key=o_api_key, max_retries=0) if (o_api_key and OpenAI) else None

    os.makedirs("trial", exist_ok=True)
    clean_old_caches(cache_dir="trial")

    url = args.url or os.getenv("PROMPT_URL")
    base_prompt_content = ""
    if url:
        base_prompt_content = fetch_prompt_from_github(url)
    if not base_prompt_content:
        for path in ("prompt/news.txt", "news.txt"):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    base_prompt_content = f.read()
                break
            except FileNotFoundError:
                continue
    if not base_prompt_content:
        sys.exit(1)

    archive_text = get_recent_archives(7)
    phase1_prompt_content = (archive_text + "\n" + base_prompt_content) if archive_text else base_prompt_content

    ref_content = ""
    ref_count = 0
    if os.path.exists("ref.txt"):
        try:
            with open("ref.txt", "r", encoding="utf-8") as f:
                raw_text = f.read().strip()
                ref_lines = []
                if raw_text:
                    for raw_line in raw_text.split("\n"):
                        line = raw_line.strip()
                        if not line:
                            continue
                        parts = re.split(r'(?:(?<=\S)\s+|^)(?=\d{1,2}[.\)]\s|[-•]\s)', line)
                        parts = [p for p in parts if p.strip()] or [line]
                        for part in parts:
                            content = re.sub(r'^(?:[-•]|\d{1,2}[.\)])\s*', '', part.strip()).strip()
                            if content:
                                ref_lines.append(content)
                if ref_lines:
                    ref_content = "\n".join(f"▶ {line}" for line in ref_lines)
                    ref_count = len(ref_lines)
        except Exception as e:
            print(f"Warning: Could not read ref.txt: {e}")

    if ref_content:
        auto_count = max(0, 4 - ref_count)
        ref_instr = (
            f"\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"[🎯 MANDATORY TARGET NEWS (최우선 반영 지시)]\n"
            f"사용자가 오늘 특별히 다음 **{ref_count}개**의 주제/사건에 대한 분석을 요청했습니다:\n"
            f"{ref_content}\n\n"
            f"지시사항: 최종 4개의 주요 뉴스 중, 위 요청된 {ref_count}개의 주제를 빠짐없이 포함하고 나머지 {auto_count}개는 직접 발굴하십시오.\n"
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

        user_trigger = (
            "시스템 지시사항(GLOBAL RULES)에 정의된 형식과 규칙에 따라 지금 바로 구글 검색을 다각도로 수행하여 최신 AI 뉴스 4개를 발굴하고 영문 보고서를 작성하십시오. "
            "특히 Google, OpenAI, Anthropic, Meta 등 빅테크 거대 LLM 소식에만 편중하지 말고, "
            "이미지 생성/편집 특화 모델(나노 바나나 대항마, FLUX, Krea, Qwen-Image, Seedream 등), "
            "포토샵/어도비 생태계를 대체·파괴하는 디자인/모션/UI 특화 도구(Claude Motion 등), "
            "비디오/모션 생성 특화 모델(Kling, Runway, Luma, Wan, Hailuo, Veo, Sora 등) 및 오픈소스/스타트업의 파급력 있는 최신 발표가 있다면 반드시 적극 발굴하여 주요 기사로 선정하십시오."
        )
        initial_result = run_grounded_research(g_client, g_model, phase1_prompt_content, user_trigger, p1_cache_file)

        if not initial_result:
            print("❌ Phase 1 failed after 5 attempts. Exiting.")
            sys.exit(1)

        if os.path.exists("ref.txt"):
            with open("ref.txt", "w", encoding="utf-8") as f:
                f.write("-\n-\n-\n-\n")
            print("🧹 Phase 1 성공: 내일 바로 채워 쓸 수 있도록 ref.txt를 '-' 4줄 템플릿으로 초기화했습니다.")

        print(f"✅ Phase 1 complete. Saved to {p1_cache_file} (Time: {time.time() - p1_start:.2f}s)")

    # ---------------------------------------------------------
    # Phase 2: GPT-6.1 Sol Validation (초고도화 팩트체크)
    # ---------------------------------------------------------
    p2_cache_file = "trial/feedback.txt"
    feedback = None

    if o_client:
        print("\n--- Phase 2: GPT-6.1 Validation ---")
        if os.path.exists(p2_cache_file):
            print("✅ Phase 2: 오늘 이미 생성된 로컬 캐시(feedback.txt)에서 피드백을 불러옵니다.")
            with open(p2_cache_file, "r", encoding="utf-8") as f:
                feedback = f.read()
        else:
            print("Starting GPT-6.1 action (Model: gpt-6.1-sol)...")
            p2_start = time.time()
            current_dt = now_kst()
            current_kst = current_dt.strftime("%Y-%m-%d %H:%M KST")
            current_year = current_dt.strftime("%Y")
            p2_system = (
                f"Today's actual date and time is {current_kst}. "
                f"You are a fact-checker for a Strategic AI Technology, Specialized Models (Image, Video, Design/Creative, LLM) & Hardware Ecosystem briefing. "
                f"The draft was produced by a separate AI agent using real-time Google Search as of today. "
                f"Therefore, all dates and events in the draft referencing {current_year} "
                f"are CURRENT, not future or fictional. "
                f"Do NOT flag any {current_year} dates as hypothetical or future scenarios. "
                f"Note: most items must have an event date within the last 24 hours, EXCEPT items whose metadata "
                f"shows slot_category: 6 (HW-Circuit-Research / academic-research items), which are allowed an event "
                f"date up to 7 days old (event_within_7d_window: true). Do not flag those items as stale for that reason alone. "
                f"Also EXCEPT items whose metadata shows ongoing_story_exception: true — these cover a major, high-impact "
                f"story whose original event may be older than 24 hours, but which includes a genuinely new development "
                f"within the last 24 hours. Do not flag these as stale either; instead verify that the article's focus is "
                f"actually the new development (not just a rehash of the old event) and that original_event_date is present. "
                f"Specialized generative models (image generation/editing, Photoshop/Adobe-replacement creative tools, video/motion synthesis, open-source models) "
                f"are explicitly encouraged and valid for Categories 1 and 2 — do NOT penalize them for not being pure hardware/datacenter stories. "
                f"Use your advanced reasoning capabilities to verify company names, "
                f"product specs, statistics, and URLs before forming any judgment. "
                f"Respond in Korean."
            )
            p2_user_prompt = (
                f"다음 AI 뉴스 초안의 사실 관계와 KST 기준 시간 윈도우를 검증하십시오.\n\n"
                f"[검증 지침]\n"
                f"- 각 항목의 기업명, 모델/제품명, 통계 수치, URL을 엄격하게 교차 검증하십시오.\n"
                f"- 명확한 논리적 모순이나 반증을 찾았을 때만 오류로 지적하십시오.\n"
                f"- 오류가 없는 항목은 논리의 비약, 전략적 분석의 깊이 부족, 아키텍처/워크플로우 설명의 모호함, 양식 누락 등 품질 관점에서 건설적인 보완을 제안하십시오.\n"
                f"- 응답의 맨 마지막 줄에 반드시 다음 세 가지 태그 중 하나만 단독으로 출력하십시오:\n"
                f"  1) [VERDICT: PASS] : 4개 기사 모두 시간 윈도우와 사실관계가 정확하여 수정 없이 즉시 발행 가능한 경우\n"
                f"  2) [VERDICT: REFINE] : 4개 기사의 주제는 모두 유효하나(기사 교체 불필요), 세부 수치·논리·분석 보완이 필요한 경우\n"
                f"  3) [VERDICT: REPLACE] : 1개 이상의 기사가 시간 윈도우(24h/7d)를 위반했거나 핵심 사실이 허위여서 다른 기사로 전면 교체(재검색)해야 하는 경우\n\n"
                f"[초안]\n{initial_result}"
            )
            gpt_messages = [{"role": "user", "content": p2_user_prompt}]

            feedback = run_gpt_chat(o_client, "gpt-6.1-sol", gpt_messages, system=p2_system)
            if not feedback:
                print("⚠️ Phase 2 (GPT-6.1 Sol) 호출 실패: Graceful Fallback 발동. Phase 1 초안을 유지하고 후속 단계로 진행합니다.")
                feedback = None
            else:
                with open(p2_cache_file, "w", encoding="utf-8") as f:
                    f.write(feedback)
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
            verdict = parse_feedback_verdict(feedback)
            print(f"📋 Phase 2 Verdict 감지 결과: {verdict}")

            if verdict == "PASS":
                print("⚡ [VERDICT: PASS] 초안의 품질과 사실관계가 모두 통과되어 Phase 3 재작성을 건너뜁니다.")
                refined_result = initial_result
                with open(p3_cache_file, "w", encoding="utf-8") as f:
                    f.write(refined_result)
            else:
                print(f"Starting Refinement (Model: {g_model})...")
                p3_start = time.time()
                current_kst = now_kst().strftime("%Y-%m-%d %H:%M KST")

                needs_search = (verdict == "REPLACE")

                if needs_search:
                    print("🔍 피드백에 기사 교체([VERDICT: REPLACE])가 감지되어 Google Search 툴을 가동합니다.")
                    tools = [types.Tool(google_search=types.GoogleSearch())]
                    refine_temp = 0.3
                    refine_prompt = (
                        f"현재 실제 KST 시간은 {current_kst}입니다.\n"
                        "위 피드백을 철저히 반영하여 [Initial Draft]를 다듬어 영문 최종본을 작성하십시오.\n"
                        "1. 🚨 [조건 위반 및 사실 오류 수정] 피드백 내용 중 기사가 시간 윈도우나 조건을 위반했거나 심각한 사실 오류가 있어 제외(Drop) 판정을 받았다면, 해당 항목을 전면 삭제하십시오.\n"
                        "2. 💡 [품질 보완 제안 적극 반영] 유지되는 기사에 대해서는 피드백의 보완 제안 사항들을 본문에 완벽히 녹여내어 분석의 깊이를 극대화하십시오.\n"
                        "3. 삭제된 빈 슬롯 수만큼, 당신에게 부여된 'Google Search' 툴을 즉각 사용하여 완전히 새로운 고임팩트 AI 뉴스를 직접 발굴하여 대체 작성하십시오. "
                        "일반 슬롯(카테고리 1~5) 대체 시 빅테크 LLM에만 국한하지 말고 이미지 생성/편집 특화 모델(FLUX, Krea, Qwen-Image 등), 포토샵/디자인 대체 크리에이티브 툴(Claude Motion 등), 비디오/모션 특화 모델(Kling, Runway, Luma, Wan 등), 에이전트 및 AI 인프라 뉴스 중 최근 24시간 이내 발생한 파급력 높은 사건을 우선 발굴하십시오. "
                        "단, 삭제된 항목이 카테고리 6(HW-Circuit-Research, 학술연구/미래 로드맵)이었다면 최근 7일 이내 발표된 논문·컨퍼런스·기술 블로그를 기준으로 발굴하십시오. "
                        "최종 리포트는 무조건 4개의 항목으로 채워져야 합니다.\n"
                        "4. 피드백에서 지적되지 않은 정상 항목은 불필요한 재작성 없이 원문을 최대한 유지하십시오.\n"
                        "5. Bloomberg, TechCrunch, The Economist 스타일의 명쾌하고 직관적인 비즈니스 영문(Plain English)으로 작성하고 인라인 수식 기호(LaTeX)는 절대 사용하지 마십시오.\n"
                        "6. 기존 섹션 구조(Overview, Strategic Impact, Technical Deep Dive 등)를 정확히 유지하십시오."
                    )
                else:
                    print("⚡ 기사 교체 없는 품질 보완 피드백([VERDICT: REFINE])입니다. Google Search 없이 고속 정제 모드(Fast Refine)로 실행합니다.")
                    tools = None
                    refine_temp = 0.2
                    refine_prompt = (
                        f"현재 실제 KST 시간은 {current_kst}입니다.\n"
                        "피드백에 기사 삭제나 전면 교체 지시가 없으므로, 기존 4개 기사의 주제와 핵심 팩트를 그대로 유지하면서 품질을 다듬어 영문 최종본을 작성하십시오.\n"
                        "1. 💡 [품질 보완 제안 적극 반영] 피드백에 명시된 개선 요청 사항들(사실관계 교정, 아키텍처/워크플로우 설명 보완, 분석 깊이 강화 등)을 본문에 완벽히 녹여내십시오.\n"
                        "2. 기존 4개 기사 블록의 구조(Overview, Strategic Impact, Technical Deep Dive 등), 메타데이터 태그, URL을 누락 없이 정확히 유지하십시오.\n"
                        "3. Bloomberg, TechCrunch, The Economist 스타일의 명쾌하고 직관적인 비즈니스 영문(Plain English)으로 다듬고, 인라인 수식 기호(LaTeX)는 절대 사용하지 마십시오.\n"
                        "4. 피드백에서 지적되지 않은 부분은 원문의 완성도를 훼손하지 않도록 최대한 보존하십시오."
                    )

                sys_instruction = (
                    "You are a top-tier AI Intelligence Analyst and Editor in the style of Bloomberg, TechCrunch, and The Economist. "
                    "Refine the news draft in clear, accessible, and engaging modern business English according to the feedback while maintaining strict analytical rigor.\n\n"
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

                        config_args = {
                            "temperature": refine_temp,
                            "system_instruction": sys_instruction
                        }
                        if tools:
                            config_args["tools"] = tools

                        response = g_client.models.generate_content(
                            model=g_model,
                            contents=contents,
                            config=types.GenerateContentConfig(**config_args)
                        )
                        if response.text:
                            refined_result = response.text
                            with open(p3_cache_file, "w", encoding="utf-8") as f:
                                f.write(refined_result)
                            print(f"✅ Phase 3 complete. Saved to {p3_cache_file} (Time: {time.time() - p3_start:.2f}s)")
                            break
                        raise ValueError("Empty response from Gemini in Phase 3")
                    except Exception as e:
                        print(f"⚠️ Phase 3 attempt {attempt + 1} failed: {e}")
                        if attempt < max_retries - 1:
                            sleep_time = min(60, 15 * (2 ** attempt)) + random.uniform(1, 3)
                            print(f"   {sleep_time:.1f}초 후 재시도합니다...")
                            time.sleep(sleep_time)
                        else:
                            print("⚠️ Phase 3 최종 실패: Graceful Fallback 발동. Phase 1 초안(initial_result)을 기본값으로 유지하고 후속 단계로 진행합니다.")
                            refined_result = initial_result

    # ---------------------------------------------------------
    # Phase 4: GPT-6 Luna Translation & Polishing (조건부 스마트 실행)
    # ---------------------------------------------------------
    p4_cache_file = "trial/translated.txt"
    final_content = refined_result
    force_phase4 = args.force_polish or (os.getenv("FORCE_PHASE4", "").lower() == "true")
    has_korean = bool(RE_KOREAN_CHARS.search(refined_result or ""))

    if o_client:
        print("\n--- Phase 4: GPT-6 Translation & Polishing ---")
        if os.path.exists(p4_cache_file):
            print("✅ Phase 4: 오늘 이미 생성된 로컬 캐시(translated.txt)에서 번역/윤문본을 불러옵니다.")
            with open(p4_cache_file, "r", encoding="utf-8") as f:
                final_content = f.read()
        elif not has_korean and not force_phase4:
            print("⚡ Phase 4 최적화 패스: Phase 3 결과물이 이미 100% 영문(Plain English)으로 작성되어 불필요한 재번역 API 호출을 생략합니다.")
            final_content = refined_result
            with open(p4_cache_file, "w", encoding="utf-8") as f:
                f.write(final_content)
        else:
            reason = "한국어 텍스트 감지" if has_korean else "강제 윤문 옵션 활성화"
            print(f"Starting GPT-6 action (Model: gpt-6-luna, 사유: {reason})...")
            p4_start = time.time()
            current_kst = now_kst().strftime("%Y-%m-%d %H:%M KST")
            p4_system = (
                f"Today's actual date and time is {current_kst}. "
                f"You are an elite technology journalist and translator in the style of Bloomberg, TechCrunch, and The Economist. "
                f"Context: The text provided has already been refined and corrected based on QA feedback. "
                f"Your task is to translate and polish this finalized AI briefing into clear, accessible, and engaging modern business English. "
                f"Style & Tone Directives:\n"
                f"1. AVOID overly dense academic jargon, archaic phrasing, or labyrinthine sentence structures.\n"
                f"2. Make the writing crisp, active, and intuitive so that strategic insights, specialized model capabilities, and hardware/software breakthroughs are immediately understandable to any tech-savvy business leader.\n"
                f"3. When complex technical terms or acronyms appear, explain or frame them in plain, relatable terms while preserving technical accuracy.\n"
                f"4. Rules: preserve all section headers, metadata tags, data tables, URLs, and exact numerical figures as-is.\n"
                f"5. Output only the translated report with no commentary, preamble, or explanatory notes."
            )
            translate_messages = [{
                "role": "user",
                "content": (
                    f"Please translate and polish the following final AI briefing into clear, accessible, and engaging business English. "
                    f"Prioritize clarity and readability while strictly preserving all structure, section headers, metadata fields, and numerical data.\n\n"
                    f"[FINAL DRAFT — {current_kst}]\n{refined_result}"
                )
            }]

            translated = run_gpt_chat(o_client, "gpt-6-luna", translate_messages, system=p4_system)
            if not translated:
                print("⚠️ Phase 4 (GPT-6 Luna) 호출 실패: Graceful Fallback 발동. Phase 3 정제본(또는 초안)을 그대로 유지합니다.")
                final_content = refined_result
            else:
                final_content = translated
                with open(p4_cache_file, "w", encoding="utf-8") as f:
                    f.write(final_content)
                print(f"✅ Phase 4 complete. Saved to {p4_cache_file} (Time: {time.time() - p4_start:.2f}s)")

    # ---------------------------------------------------------
    # 최종 데이터 저장 (Phase 5 진입 전 안전 확보)
    # ---------------------------------------------------------
    today_str = now_kst().strftime("%Y-%m-%d")
    os.makedirs("data", exist_ok=True)
    with open(f"data/{today_str}.txt", "w", encoding="utf-8") as f:
        f.write(final_content)

    # ---------------------------------------------------------
    # Phase 5: HTML Generation (Self-Healing 검증 루프)
    # ---------------------------------------------------------
    print("\n--- Phase 5: HTML Generation ---")
    p5_start = time.time()

    template_content = ""
    for path in ("prompt/template.html", "template.html"):
        try:
            with open(path, "r", encoding="utf-8") as f:
                template_content = f.read()
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
    if not html_prompt_content:
        for path in ("prompt/html_content.txt", "html_content.txt"):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    html_prompt_content = f.read()
                break
            except FileNotFoundError:
                continue

    if not html_prompt_content:
        print("❌ Phase 5 failed: html_content.txt 프롬프트를 찾을 수 없습니다 (HTML_PROMPT_URL 또는 prompt/html_content.txt 확인 필요).")
        sys.exit(1)

    gemini_model_id = "gemini-3.8-flash"
    print(f"Generating article content using model: {gemini_model_id}...")

    base_user_prompt = f"다음 데이터를 바탕으로 시스템 지시사항에 맞춰 4개 기사(<article>) 블록과 마무리 Bento Grid(<section class=\"bento-grid-wrap\">)만 생성하십시오:\n\n{final_content}"
    user_prompt = base_user_prompt

    articles_content = None
    max_retries = 3
    for attempt in range(max_retries):
        active_model = gemini_model_id if attempt < 2 else g_model
        try:
            response = g_client.models.generate_content(
                model=active_model,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=html_prompt_content,
                    temperature=0.3
                )
            )
            raw_html = response.text
            if not raw_html:
                raise ValueError("모델로부터 빈 응답이 반환되었습니다.")

            validate_articles_content(raw_html)
            articles_content = _strip_markdown_fence(raw_html)
            break
        except ValueError as ve:
            print(f"⚠️ Phase 5 validation error on attempt {attempt + 1} (Model: {active_model}): {ve}")
            user_prompt = (
                f"{base_user_prompt}\n\n"
                f"🚨 [이전 생성 시도 검증 실패 사유 — 반드시 수정하여 재출력하십시오]:\n"
                f"- 오류 내용: {ve}\n"
                f"- 허용된 SVG 아이콘 ID만 사용하고, 4개의 <article> 태그와 마지막 <section class=\"bento-grid-wrap\">을 빠짐없이 포함하며, 모든 차트 ID와 데이터를 정확히 매칭하십시오."
            )
            if attempt < max_retries - 1:
                time.sleep(5)
        except Exception as e:
            print(f"⚠️ Phase 5 API attempt {attempt + 1} failed (Model: {active_model}): {e}")
            if attempt < max_retries - 1:
                time.sleep(10)

    if not articles_content:
        print("❌ Phase 5 failed after retries. Exiting.")
        sys.exit(1)

    report_date_str = now_kst().strftime("%B %d, %Y")
    html_code = template_content.replace("{{ARTICLES_CONTENT}}", articles_content).replace("{{REPORT_DATE}}", report_date_str)

    if "{{ARTICLES_CONTENT}}" in html_code or "{{REPORT_DATE}}" in html_code or "<!DOCTYPE html>" not in html_code or "</html>" not in html_code:
        print("❌ Phase 5 failed: 최종 HTML 조립에 실패했습니다. index.html을 덮어쓰지 않습니다.")
        sys.exit(1)

    with open("index.html", "w", encoding="utf-8") as f:
        f.write(html_code)

    with open("data/.html_synced", "w", encoding="utf-8") as f:
        f.write(today_str)

    print(f"✅ Phase 5 complete. Saved to index.html (Time: {time.time() - p5_start:.2f}s)")


if __name__ == "__main__":
    main()
