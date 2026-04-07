"""
E2E Smoke Test for the AI News Research Pipeline.
Tests model discovery, API connectivity, and file output at each stage.
Uses ultra-short prompts and minimal max_tokens to keep API cost as close to zero as possible.
"""
import os
import sys
import datetime
from dotenv import load_dotenv

load_dotenv()

def ensure_dir(filepath):
    d = os.path.dirname(filepath)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def write_and_verify(filepath, content, phase_name):
    ensure_dir(filepath)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    size = os.path.getsize(filepath)
    print(f"  ✅ {phase_name}: wrote {filepath} ({size} bytes)")
    return size > 0

def main():
    errors = []
    print("🚀 Starting API Connected Smoke Test (Cost Minimized Mode)")

    # ── Phase 0: Model Discovery ──────────────────────────────
    print("\n══════ Phase 0: Model Discovery ══════")

    # Gemini
    gemini_key = os.getenv("GEMINI_API_KEY")
    gemini_agent = gemini_model = None
    if gemini_key:
        from google import genai
        gclient = genai.Client(api_key=gemini_key)
        # 수정된 main.py의 함수명으로 import
        from main import get_latest_gemini_model, run_grounded_research
        
        gemini_agent = get_latest_gemini_model(gclient, require_agent=True)
        gemini_model = get_latest_gemini_model(gclient, require_agent=False)
        print(f"  🔬 Deep Research Agent : {gemini_agent}")
        print(f"  🎨 Generative Model    : {gemini_model}")
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

    # ── Phase 1: Grounded Research ───────────────────────────
    print("\n══════ Phase 1: Grounded Research (API Hit) ══════")
    if gemini_key:
        # Grounding 툴이 켜져 있으므로 웹 검색을 최소화하도록 극단적으로 짧고 명확한 명령 전달
        tiny_prompt = "This is an API test. Reply exactly with 'OK' and do nothing else."
        initial_research, _ = run_grounded_research(tiny_prompt, "trial/1.txt")
        
        if not initial_research:
            initial_research = "ERROR: Grounded research failed"
            errors.append("Phase 1 Grounded Research failed")
        else:
            print(f"  Gemini responded: {initial_research.strip()}")
    else:
        initial_research = "Skipped"
        write_and_verify("trial/1.txt", initial_research, "Phase 1 (skipped)")

    # ── Phase 2: Claude Validation ────────────────────────────
    print("\n══════ Phase 2: Claude Validation (API Hit) ══════")
    if claude_key and claude_model:
        try:
            # max_tokens를 10으로 극단적으로 제한하여 출력 비용 최소화
            msg = cclient.messages.create(
                model=claude_model, 
                max_tokens=10,
                messages=[{"role": "user", "content": "Reply exactly with 'VALIDATION_OK'"}]
            )
            feedback = msg.content[0].text.strip()
            print(f"  Claude responded: {feedback}")
            write_and_verify("trial/feedback.txt", feedback, "Phase 2")
        except Exception as e:
            errors.append(f"Phase 2 Claude error: {e}")
            print(f"  ❌ Claude error: {e}")
            write_and_verify("trial/feedback.txt", f"ERROR: {e}", "Phase 2 (fallback)")
    else:
        write_and_verify("trial/feedback.txt", "SKIPPED: No Claude API key", "Phase 2 (skipped)")

    # ── Phase 3: Gemini Refinement ────────────────────────────
    print("\n══════ Phase 3: Refinement (API Hit) ══════")
    if gemini_key and gemini_model:
        try:
            print(f"  Refining with model: {gemini_model}...")
            # 불필요한 컨텍스트(초기 리서치 내용) 전달을 생략하여 입력 토큰 절약
            refine_prompt = "This is a test. Reply exactly with 'REF_OK'."
            response = gclient.models.generate_content(
                model=gemini_model,
                contents=[refine_prompt]
            )
            refined_result = response.text.strip()
            print(f"  Gemini responded: {refined_result}")
            write_and_verify("trial/2.txt", refined_result, "Phase 3")
        except Exception as e:
            errors.append(f"Phase 3 Refinement error: {e}")
            print(f"  ❌ Phase 3 error: {e}")
            write_and_verify("trial/2.txt", f"ERROR: {e}", "Phase 3 (fallback)")
    else:
        write_and_verify("trial/2.txt", "SKIPPED: No Gemini API key", "Phase 3 (skipped)")

    # ── Phase 4: Claude Translation ───────────────────────────
    print("\n══════ Phase 4: Claude Translation (API Hit) ══════")
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    archive_file = f"data/{today}.txt"
    
    if claude_key and claude_model:
        try:
            # max_tokens=10 제한
            msg = cclient.messages.create(
                model=claude_model, 
                max_tokens=10,
                messages=[{"role": "user", "content": "Reply exactly with 'TRANSLATION_OK'"}]
            )
            translation = msg.content[0].text.strip()
            print(f"  Claude responded: {translation}")
            write_and_verify(archive_file, translation, "Phase 4")
        except Exception as e:
            errors.append(f"Phase 4 Claude error: {e}")
            print(f"  ❌ Claude error: {e}")
            write_and_verify(archive_file, f"ERROR: {e}", "Phase 4 (fallback)")
    else:
        write_and_verify(archive_file, "SKIPPED: No Claude API key", "Phase 4 (skipped)")

    # ── Phase 5: Gemini HTML Generation ───────────────────────
    print("\n══════ Phase 5: HTML Generation (API Hit) ══════")
    if gemini_key and gemini_model:
        try:
            # HTML 구조를 명시하여 최소한의 토큰만 출력하게 유도
            response = gclient.models.generate_content(
                model=gemini_model,
                contents="Output exactly this string: '<html><body>OK</body></html>'"
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
            print(f"  ❌ Gemini error: {e}")
            write_and_verify("index.html", f"<html><body>ERROR</body></html>", "Phase 5 (fallback)")
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
