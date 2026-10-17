import os
import sys
from main import (
    ALLOWED_ICONS,
    extract_archive_summary,
    get_recent_archives,
    parse_feedback_verdict,
    validate_articles_content,
)


def run_smoke_tests():
    print("🚀 Running AI News Pipeline Smoke Tests...")

    # 1. 프롬프트 및 템플릿 파일 검증
    for required_file in ("prompt/news.txt", "prompt/html_content.txt", "prompt/template.html"):
        if not os.path.exists(required_file):
            raise FileNotFoundError(f"필수 파일 누락: {required_file}")

    with open("prompt/news.txt", "r", encoding="utf-8") as f:
        news_prompt = f.read()
    assert "Specialized Image, Video, Creative & Open-Weight AI Labs" in news_prompt, "특화 모델 생태계 지침 누락"

    with open("prompt/template.html", "r", encoding="utf-8") as f:
        template_html = f.read()
    assert "{{ARTICLES_CONTENT}}" in template_html, "템플릿 자리표시자 누락"
    for icon_id in ALLOWED_ICONS:
        symbol_id = f'id="{icon_id.lstrip("#")}"'
        assert symbol_id in template_html, f"template.html에 아이콘 심볼 누락: {symbol_id}"
    print("✅ 1. Prompt & Template integrity verified.")

    # 2. 아카이브 요약 추출 검증
    archive_index = get_recent_archives(30)
    print(f"✅ 2. Archive index extraction verified ({len(archive_index)} chars).")

    # 3. Phase 2 -> Phase 3 Verdict 파서 검증
    assert parse_feedback_verdict("모든 항목 정상입니다.\n[VERDICT: PASS]") == "PASS"
    assert parse_feedback_verdict("일부 수치 보완 필요.\n[VERDICT: REFINE]") == "REFINE"
    assert parse_feedback_verdict("시간 윈도우 불충족.\n[VERDICT: REPLACE]") == "REPLACE"
    assert parse_feedback_verdict("전압 강하(voltage drop) 설명 보완 필요.") == "REFINE"
    assert parse_feedback_verdict("항목 1은 시간 요건 불충족으로 전면 삭제 및 대체 작성 필요.") == "REPLACE"
    print("✅ 3. Deterministic feedback verdict parser verified.")

    # 4. Phase 5 HTML 검증기 테스트
    sample_valid_html = """
    <article class="paper-section" id="art1"><h2>1</h2><div id="chart-art1-a"></div><script>document.getElementById('chart-art1-a'); var d = {data: [10, 25, 40]};</script></article>
    <article class="paper-section" id="art2"><h2>2</h2><svg><use href="#icon-media"/></svg></article>
    <article class="paper-section" id="art3"><h2>3</h2><svg><use href="#icon-code"/></svg></article>
    <article class="paper-section" id="art4"><h2>4</h2><svg><use href="#icon-chip"/></svg></article>
    <section class="bento-grid-wrap"></section>
    """
    validate_articles_content(sample_valid_html)
    print("✅ 4. Phase 5 HTML validator verified.")
    print("🎉 All smoke tests passed successfully!")


if __name__ == "__main__":
    try:
        run_smoke_tests()
    except Exception as exc:
        print(f"❌ Smoke test failed: {exc}")
        sys.exit(1)
