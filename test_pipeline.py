import os
import sys
from main import (
    ALLOWED_ICONS,
    assemble_html_document,
    build_navigation_html,
    collect_archive_dates,
    get_recent_archives,
    inject_chart_fallbacks,
    parse_feedback_verdict,
    validate_articles_content,
)


def run_smoke_tests():
    print("🚀 Running AI News Pipeline Smoke Tests...")

    # 1. 프롬프트 및 템플릿 파일 검증
    for required_file in ("prompt/news.txt", "prompt/html_content.txt", "prompt/template.html", "archive.html"):
        if not os.path.exists(required_file):
            raise FileNotFoundError(f"필수 파일 누락: {required_file}")

    with open("prompt/news.txt", "r", encoding="utf-8") as f:
        news_prompt = f.read()
    assert "Specialized Image, Video, Creative & Open-Weight AI Labs" in news_prompt, "특화 모델 생태계 지침 누락"

    with open("prompt/template.html", "r", encoding="utf-8") as f:
        template_html = f.read()
    for placeholder in (
        "{{ARTICLES_CONTENT}}",
        "{{REPORT_DATE}}",
        "{{SIDE_NAV_ITEMS}}",
        "{{MOBILE_NAV_OPTIONS}}",
        "{{MOBILE_NAV_PILLS}}",
        "{{ARCHIVE_NAV_HTML}}",
        "{{ASSET_PREFIX}}",
    ):
        assert placeholder in template_html, f"템플릿 자리표시자 누락: {placeholder}"

    for css_selector in (".concept-box", ".code-term", ".outlook-list", ".arch-diagram-wrap", ".mobile-toc-bar", ".chart-fallback"):
        assert css_selector in template_html, f"필수 CSS 클래스 누락: {css_selector}"

    for icon_id in ALLOWED_ICONS:
        symbol_id = f'id="{icon_id.lstrip("#")}"'
        assert symbol_id in template_html, f"template.html에 아이콘 심볼 누락: {symbol_id}"
    print("✅ 1. Prompt & Template integrity verified.")

    # 2. 아카이브 요약 추출 검증
    archive_index = get_recent_archives(30)
    archive_entries = collect_archive_dates(data_dir="data", archive_dir="archive")
    assert len(archive_entries) > 0, "아카이브 날짜 목록 추출 실패"
    print(f"✅ 2. Archive index extraction verified ({len(archive_index)} chars, {len(archive_entries)} dates).")

    # 3. Phase 2 -> Phase 3 Verdict 파서 검증
    assert parse_feedback_verdict("모든 항목 정상입니다.\n[VERDICT: PASS]") == "PASS"
    assert parse_feedback_verdict("일부 수치 보완 필요.\n[VERDICT: REFINE]") == "REFINE"
    assert parse_feedback_verdict("시간 윈도우 불충족.\n[VERDICT: REPLACE]") == "REPLACE"
    assert parse_feedback_verdict("전압 강하(voltage drop) 설명 보완 필요.") == "REFINE"
    assert parse_feedback_verdict("항목 1은 시간 요건 불충족으로 전면 삭제 및 대체 작성 필요.") == "REPLACE"
    print("✅ 3. Deterministic feedback verdict parser verified.")

    # 4. Phase 5 HTML 검증기 & 정적 선렌더링(TOC + Chart Fallback) 테스트
    sample_valid_html = """
    <article class="paper-section" id="art1">
      <h2>Article One Architecture</h2>
      <div class="arch-diagram-wrap"><div class="arch-diagram-stage"><svg viewBox="0 0 720 180"><rect x="10" y="10" width="100" height="40"/></svg></div></div>
      <div class="chart-container"><div id="chart-art1-a"></div></div>
      <script>
      (function(){
        var el = document.getElementById('chart-art1-a');
        if (el && window.echarts) {
          var c = echarts.init(el, null, {renderer: 'svg'});
          c.setOption({
            xAxis: { type: 'category', data: ['Alpha', 'Beta', 'Gamma'] },
            series: [{ name: 'Score', type: 'bar', data: [10, 25, 40] }]
          });
        }
      })();
      </script>
    </article>
    <article class="paper-section" id="art2"><h2>Article Two Media</h2><svg><use href="#icon-media"/></svg></article>
    <article class="paper-section" id="art3"><h2>Article Three Code</h2><svg><use href="#icon-code"/></svg></article>
    <article class="paper-section" id="art4"><h2>Article Four Chiplet</h2><svg><use href="#icon-chip"/></svg></article>
    <section class="bento-grid-wrap"><h2 class="bento-grid-title">Key Strategic Takeaways</h2></section>
    """
    validate_articles_content(sample_valid_html)

    with_fallbacks = inject_chart_fallbacks(sample_valid_html)
    assert '<div class="chart-fallback">' in with_fallbacks, "차트 정적 폴백 생성 실패"
    assert "chart-fallback-table" in with_fallbacks, "차트 폴백 데이터 테이블 생성 실패"

    side_items, mob_opts, mob_pills = build_navigation_html(sample_valid_html)
    assert 'href="#art1"' in side_items and 'href="#art4"' in side_items, "정적 사이드바 목차 생성 실패"
    assert '<option value="#art1">' in mob_opts, "모바일 점프 옵션 생성 실패"
    assert 'href="#art1"' in mob_pills, "모바일 퀵 필 생성 실패"

    assembled = assemble_html_document(
        template_html,
        sample_valid_html,
        "November 13, 2026",
        current_iso_date="2026-11-13",
        archive_entries=archive_entries,
        latest_iso_date="2026-11-13",
    )
    assert "{{" not in assembled, "치환되지 않은 템플릿 자리표시자 잔존"
    assert 'id="takeaways"' in assembled, "takeaways 앵커 ID 자동 보정 실패"
    print("✅ 4. Phase 5 HTML validator, static TOC, chart fallback, and assembly verified.")
    print("🎉 All smoke tests passed successfully!")


if __name__ == "__main__":
    try:
        run_smoke_tests()
    except Exception as exc:
        print(f"❌ Smoke test failed: {exc}")
        sys.exit(1)
