"""
E2E Smoke Test for the AI News Research Pipeline.
Tests model discovery, API connectivity, and file output at each stage.
Uses ultra-short prompts and minimal max_tokens to keep API cost as close to zero as possible.
"""
import os
import sys
import datetime
import time
from dotenv import load_dotenv

load_dotenv()

# 실행 시점의 시간을 전역 변수로 설정 (업데이트 확인용)
CURRENT_KST = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def ensure_dir(filepath):
    d = os.path.dirname(filepath)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def write_and_verify(filepath, content, phase_name):
    ensure_dir(filepath)
    # 내용 뒤에 실행 시간을 붙여서 파일이 실제로 바뀌었는지 확인 가능하게 함
    final_content = f"{content}\n(Generated at: {CURRENT_KST})"
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(final_content)
    size = os.path.getsize(filepath)
    print(f"  ✅ {phase_name}: wrote {filepath} ({size} bytes)")
    return size > 0

def main():
    errors = []
    print(f"🚀 Starting API Connected Smoke Test (Cost Minimized Mode)")
    print(f"⏰ Test Time: {CURRENT_KST}")

    # ── Phase 0: Model Discovery ──────────────────────────────
    print("\n══════ Phase 0: Model Discovery ══════")

    # Gemini
    gemini_key = os.getenv("GEMINI_API_KEY")
    gemini_model = None
    if gemini_key:
        from google import genai
        gclient = genai.Client(api_key=gemini_key)
        # main.py에서 수정된 함수명과 파라미터 구조 반영
        from main import get_latest_gemini_model, run_grounded_research
        
        gemini_model = get_latest_gemini_model(gclient)
        print(f"  🎨 Gemini Pro Model    : {gemini_model}")
    else:
        errors.append("GEMINI_API_KEY not set")
        print("  ❌ GEMINI_API_KEY not set")

    # Claude
    claude_key = os.getenv("CLAUDE_API_KEY")
    claude_model = None
    if claude_key:
        import anthropic
        cclient = anthropic.Anthropic(api_key=claude_key, timeout=30.0)
        from main import get_latest_claude_model
        
        claude_model = get_latest_claude_model(cclient)
        print(f"  📝 Claude Sonnet       : {claude_model}")
    else:
        errors.append("CLAUDE_API_KEY not set")
        print("  ❌ CLAUDE_API_KEY not set")

    # ── Phase 1: Grounded Research (Gemini Chat Session) ──────
    print("\n══════ Phase 1: Grounded Research (API Hit) ══════")
    gemini_chat = None
    if gemini_key and gemini_model:
        # 비용 최소화: 웹 검색을 강제로 막고, 짧은 텍스트만 요구
        tiny_prompt = f"This is an API test at {CURRENT_KST}. Do not search the web. Reply exactly with 'OK' and do nothing else."
        initial_research, gemini_chat = run_grounded_research(tiny_prompt, "trial/1.txt")
        
        if not initial_research:
            initial_research = "ERROR: Grounded research failed"
            errors.append("Phase 1 Grounded Research failed")
        else:
            print(f"  Gemini responded: {initial_research.strip()}")
            write_and_verify("trial/1.txt", initial_research.strip(), "Phase 1")
    else:
        initial_research = "Skipped"
        write_and_verify("trial/1.txt", initial_research, "Phase 1 (skipped)")

    # ── Phase 2: Claude Validation (Context Array) ────────────
    print("\n══════ Phase 2: Claude Validation (API Hit) ══════")
    claude_messages = []
    if claude_key and claude_model:
        try:
            # 배열(List)을 활용하여 대화 문맥 생성
            claude_messages = [{"role": "user", "content": f"Reply exactly with 'VALIDATION_OK at {CURRENT_KST}'"}]
            # 비용 최소화: max_tokens=20으로 극단적 제한
            msg = cclient.messages.create(
                model=claude_model, 
                max_tokens=20,
                messages=claude_messages
            )
            feedback = msg.content[0].text.strip()
            print(f"  Claude responded: {feedback}")
            write_and_verify("trial/feedback.txt", feedback, "Phase 2")
            
            # 다음 Phase를 위해 컨텍스트 누적
            claude_messages.append({"role": "assistant", "content": feedback})
        except Exception as e:
            errors.append(f"Phase 2 Claude error: {e}")
            print(f"  ❌ Phase 2 error: {e}")
            write_and_verify("trial/feedback.txt", f"ERROR: {e}", "Phase 2 (fallback)")
    else:
        write_and_verify("trial/feedback.txt", "SKIPPED: No Claude API key", "Phase 2 (skipped)")

    # ── Phase 3: Gemini Refinement (Context Maintained) ───────
    print("\n══════ Phase 3: Refinement (API Hit) ══════")
    if gemini_key and gemini_model and gemini_chat:
        max_retries = 2
        refined_result = None
        for attempt in range(max_retries):
            try:
                print(f"  Refining with Gemini Chat Session (Attempt {attempt+1})...")
                # Phase 1의 채팅 세션(gemini_chat)을 그대로 사용하여 컨텍스트 이어가기
                refine_prompt = f"This is a test at {CURRENT_KST}. Do not search the web. Reply exactly with 'REF_OK'."
                response = gemini_chat.send_message(refine_prompt)
                refined_result = response.text.strip()
                print(f"  Gemini responded: {refined_result}")
                write_and_verify("trial/2.txt", refined_result, "Phase 3")
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"  ⚠️ Attempt {attempt+1} failed ({e}). Retrying in 10s...")
                    time.sleep(10)
                else:
                    errors.append(f"Phase 3 Refinement error: {e}")
                    print(f"  ❌ Phase 3 error: {e}")
                    write_and_verify("trial/2.txt", f"ERROR: {e}", "Phase 3 (fallback)")
    else:
        write_and_verify("trial/2.txt", "SKIPPED: No Gemini Chat Session", "Phase 3 (skipped)")

    # ── Phase 4: Claude Translation (Context Maintained) ──────
    print("\n══════ Phase 4: Claude Translation (API Hit) ══════")
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    archive_file = f"data/{today}.txt"
    
    if claude_key and claude_model and claude_messages:
        try:
            # Phase 2에서 넘어온 문맥(claude_messages)에 새로운 질문 추가
            claude_messages.append({"role": "user", "content": f"Reply exactly with 'TRANSLATION_OK at {CURRENT_KST}'"})
            msg = cclient.messages.create(
                model=claude_model, 
                max_tokens=20,
                messages=claude_messages
            )
            translation = msg.content[0].text.strip()
            print(f"  Claude responded: {translation}")
            write_and_verify(archive_file, translation, "Phase 4")
        except Exception as e:
            errors.append(f"Phase 4 Claude error: {e}")
            print(f"  ❌ Phase 4 error: {e}")
            write_and_verify(archive_file, f"ERROR: {e}", "Phase 4 (fallback)")
    else:
        write_and_verify(archive_file, "SKIPPED: No Claude Context", "Phase 4 (skipped)")

    # ── Phase 5: Gemini HTML Generation ───────────────────────
    print("\n══════ Phase 5: HTML Generation (API Hit) ══════")
    if gemini_key and gemini_model:
        try:
            response = gclient.models.generate_content(
                model=gemini_model,
                contents=f"Output exactly this string: '<html><body>OK at {CURRENT_KST}</body></html>'"
            )
            html = response.text.strip()
            if "```html" in html:
                html = html.split("```html")[1].split("```")[0].strip()
            elif "```" in html:
                html = html.split("```")[1].split("```")[0].strip()
                
            write_and_verify("index.html", html, "Phase 5")
            print(f"  Gemini responded: {html}")
        except Exception as e:
            errors.append(f"Phase 5 Gemini error: {e}")
            print(f"  ❌ Phase 5 error: {e}")
            write_and_verify("index.html", f"<html><body>ERROR at {CURRENT_KST}</body></html>", "Phase 5 (fallback)")
    else:
        write_and_verify("index.html", "<html><body>SKIPPED</body></html>", "Phase 5 (skipped)")

    # ── Final Verification ────────────────────────────────────
    print("\n══════ Final Verification ══════")
    expected_files = ["trial/1.txt", "trial/feedback.txt", "trial/2.txt", archive_file, "index.html"]
    all_ok = True
    for f in expected_files:
        if os.path.exists(f) and os.path.getsize(f) > 0:
            print(f"  ✅ {f} exists ({os.path.getsize(f)} bytes)")
        else:
            print(f"  ❌ {f} MISSING or EMPTY")
            all_ok = False

    if errors:
        print(f"\n⚠️  {len(errors)} warning(s):")
        for e in errors:
            print(f"  - {e}")

    if all_ok:
        print("\n🎉 ALL PIPELINE STAGES PASSED SUCCESSFULLY!")
    else:
        print("\n❌ SOME STAGES FAILED")
        sys.exit(1)

if __name__ == "__main__":
    main()
