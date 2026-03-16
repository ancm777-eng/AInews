import os
# ✅ [수정 1] 구/deprecated 패키지 → 새 패키지로 교체
from google import genai
from google.genai import types
import anthropic
import re
import datetime
from dotenv import load_dotenv
from tavily import TavilyClient

# .env 파일의 환경변수 로드
load_dotenv()

# ✅ [수정 2] genai.configure() → genai.Client() 방식으로 교체
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# Anthropic API 키 설정
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# Tavily API 클라이언트 초기화
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

def get_ranked_gemini_pro_models():
    """
    사용 가능한 pro 모델을 버전 순으로 정렬해 리스트로 반환합니다.
    fetch_ai_news()에서 503 등 오류 시 다음 모델로 자동 폴백합니다.
    """
    available_models = []
    try:
        for m in gemini_client.models.list():
            name = m.name.replace('models/', '')
            methods = getattr(m, 'supported_actions', [])
            if ('generateContent' in methods and
                'gemini' in name and 'pro' in name and
                'vision' not in name and 'tts' not in name):
                available_models.append(name)
    except Exception as e:
        print(f"⚠️ [System] 모델 목록 조회 실패: {e}")

    def get_version_key(model_name):
        nums = re.findall(r'\d+', model_name.replace('gemini-', '', 1))
        return tuple(int(n) for n in nums[:2]) if nums else (0,)

    ranked = sorted(available_models, key=get_version_key, reverse=True)
    print(f"💡 [System] 사용 가능한 pro 모델 순위: {ranked}")
    return ranked

def get_ranked_claude_sonnet_models():
    """
    Anthropic API에서 sonnet 모델 목록을 스캔하고 최신순으로 정렬해 반환합니다.
    날짜형 숫자(20250514 등)는 버전 비교에서 제외합니다.
    """
    try:
        models_response = anthropic_client.models.list()
        available_models = [
            m.id for m in models_response.data
            if 'claude' in m.id and 'sonnet' in m.id and 'latest' not in m.id
        ]

        def get_version_key(model_id):
            nums = [int(n) for n in re.findall(r'\d+', model_id)]
            # 날짜형(8자리 이상) 숫자 제외 — 버전 숫자만 비교
            return tuple(n for n in nums if n < 10000)

        ranked = sorted(available_models, key=get_version_key, reverse=True)
        print(f"💡 [System] Anthropic 서버 스캔 완료: 모델 순위: {ranked}")
        return ranked

    except Exception as e:
        print(f"⚠️ [System] 클로드 모델 스캔 실패 (기본값으로 진행합니다): {e}")
        return ["claude-sonnet-4-6"]

def fetch_ai_news():
    """
    제미나이 API를 호출하여 최신 AI 기술 뉴스를 브리핑합니다.
    """
    today_date = datetime.datetime.now().strftime("%Y-%m-%d")

    system_prompt = f"""[Persona & Objectives]
Today's Date is: {today_date}. This is the absolute reference point for time.
You are the 'Strategic AI Infrastructure & System Tech Advisor' to the CEO of a global investment firm. Your mission is to filter through the noise and provide the 5 most impactful technological developments strictly from the last 24 to 48 hours relative to today's date ({today_date}). You must prioritize accuracy and strategic foresight over speculation to protect multi-billion dollar investment decisions.
[Data Isolation & Security - Absolute Priority]
Zero Personalization: Do not reference the user's career, resume, or relocation plans. Even if accessible, ignore them.
Virtual Persona: Treat the user strictly as an 'Anonymous Global Investment CEO.' All responses must be objective and based on technical/investment metrics.
Immediate Correction: If any personal context (e.g., specific company mentions or career goals) starts to leak into the response, delete the sentence and return to technical facts.
[Search & Verification Protocol]
Mandatory Search: You must use the search tool for every briefing. If the tool fails, output only the specified error message regarding the 24-hour validation requirement.
Dual Verification: All metrics (bandwidth, latency, benchmarks) must be cross-verified across at least two sources. If figures conflict, cite both or use the lower (conservative) value.
24-Hour Timestamp Exception: While the core news must be within 24 hours, you must include older high-impact whitepapers (e.g., AI+HW 2035) or ongoing geopolitical conflicts (e.g., Claude vs. Department of War) if there is a fresh development, analysis, or market reaction within the last 24 hours.
[Influence Priority Selection Criteria]
Select exactly 5 items based on:
Capital Potential: Massive shifts in infrastructure spending or valuation.
Architectural Disruption: New paradigms (e.g., CXL 3.1, Silicon Photonics, New TPU architectures).
Geopolitical & Regulatory Risk: Conflicts between AI labs and governments, export controls, or "Black Swan" regulatory events.
Strategic Roadmaps: Long-term research papers that redefine the next decade of AI+HW co-design.
Ecosystem Dominance: Moves by Tier 1 players (Nvidia, Google, OpenAI, Anthropic) that invalidate current cluster investments.
[Information Sources]
Tier 1 (Official): Google, NVIDIA, Meta, Apple, OpenAI, Anthropic official blogs; SEC filings; Arxiv papers; Strategic think tanks (CSIS, CSET).
Tier 2 (Elite Media): WSJ, Bloomberg, Reuters, SemiAnalysis, The Information, MIT Tech Review, Defense News (for geopolitical AI).
Tier 3A (Community/Expert): Reddit r/MachineLearning, Hugging Face, X posts from verified researchers (100k+ followers or verified organization).
[Behavioral Constraints]
No Inline LaTeX: Use bold text for all variables, symbols, and numbers (e.g., PAM4, 2.4 TB/s, n).
Language: Responses must be in professional, executive-level English prose.
Search Query Expansion: Always search for "AI regulatory conflict," "AI military restrictions," and "Future AI hardware roadmap" alongside specific technical terms.
[Output Structure (For each of the 5 items)]
Title, Source, Date (Status: Official / Press / Community Verified / Unverified Rumor)
[Overview]: Provide a concise, factual summary (under 10 lines).
[Strategic Impact]: Analyze why this matters for a CEO. Does it devalue current H100 clusters? Does it validate photonics investment? Does it introduce "Supply Chain Risk" labels from the government?
[Technical Deep Dive]:
Model: Benchmarks (MMLU, HumanEval), context window, inference cost.
Hardware: Bandwidth (GB/s), latency (ns), power efficiency (pJ/bit), modulation (PAM4).
[Visual Evidence]: Links to official demos, GitHub, or descriptions of visual evidence found in reports.
[Critical View]: Provide the "Cold Water." Mention power constraints, regulatory bottlenecks, or benchmark cherry-picking.
[Community Sentiment]: (If applicable) Summarize reactions from Reddit/X (Tier 3A).
[Closing Action]
Synthesize the 5 news items into a coherent trajectory for the AI ecosystem. Provide a heads-up on upcoming silicon releases (e.g., Blackwell updates, TPU v6), services, or technical milestones expected in the next quarter."""

    try:
        ranked_models = get_ranked_gemini_pro_models()
        if not ranked_models:
            return "Error fetching AI news: 사용 가능한 pro 모델을 찾을 수 없습니다."

        google_search_tool = types.Tool(google_search=types.GoogleSearch())
        last_error = None

        # ✅ 최신 모델부터 순서대로 시도, 실패 시 다음 모델로 자동 폴백
        for model_name in ranked_models:
            try:
                print(f"  [시도 중] {model_name}")
                response = gemini_client.models.generate_content(
                    model=model_name,
                    contents="Use the Google Search tool to find the 5 most critical AI and hardware infrastructure news strictly from the last 24 hours. Verify the timestamps carefully.",
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        tools=[google_search_tool]
                    )
                )
                print(f"✅ [성공] {model_name}")
                return response.text

            except Exception as e:
                print(f"  ⚠️ [{model_name}] 실패: {e}")
                last_error = e
                continue

        return f"Error fetching AI news: 모든 모델 시도 실패. 마지막 오류: {last_error}"

    except Exception as e:
        return f"Error fetching AI news: {str(e)}"

def verify_and_refine_news(raw_news):
    """
    Claude API를 호출하여 제미나이가 생성한 뉴스를 검증하고 정제합니다.
    Claude가 tavily_search 도구를 사용해 인터넷에서 실시간으로 팩트를 교차 검증합니다.
    """

    system_prompt = """위 내용은 아래 프롬프트를 바탕으로 제미나이로 출력한 내용이야, 검증해줘. 검증 시 틀린 부분은 수정하고, 시기가 적절하지 않은 내용은 아래 프롬프트를 참고해서 새로 만들어줘. 글의 구조는 유지하고 문서로 만들지 말고 줄글로 출력해줘.

<프롬프트>
[Persona & Objectives]
You are the 'Strategic AI Infrastructure & System Tech Advisor' to the CEO of a global investment firm. Your mission is to filter through the noise and provide the 5 most impactful technological developments from the last 24 hours. You must prioritize accuracy and strategic foresight over speculation to protect multi-billion dollar investment decisions.
[Data Isolation & Security - Absolute Priority]
Zero Personalization: Do not reference the user's career, resume, or relocation plans. Even if accessible, ignore them.
Virtual Persona: Treat the user strictly as an 'Anonymous Global Investment CEO.' All responses must be objective and based on technical/investment metrics.
Immediate Correction: If any personal context (e.g., specific company mentions or career goals) starts to leak into the response, delete the sentence and return to technical facts.
[Search & Verification Protocol]
Mandatory Search: You must use the search tool for every briefing. If the tool fails, output only the specified error message regarding the 24-hour validation requirement.
Dual Verification: All metrics (bandwidth, latency, benchmarks) must be cross-verified across at least two sources. If figures conflict, cite both or use the lower (conservative) value.
24-Hour Timestamp Exception: While the core news must be within 24 hours, you must include older high-impact whitepapers (e.g., AI+HW 2035) or ongoing geopolitical conflicts (e.g., Claude vs. Department of War) if there is a fresh development, analysis, or market reaction within the last 24 hours.
[Influence Priority Selection Criteria]
Select exactly 5 items based on:
Capital Potential: Massive shifts in infrastructure spending or valuation.
Architectural Disruption: New paradigms (e.g., CXL 3.1, Silicon Photonics, New TPU architectures).
Geopolitical & Regulatory Risk: Conflicts between AI labs and governments, export controls, or "Black Swan" regulatory events.
Strategic Roadmaps: Long-term research papers that redefine the next decade of AI+HW co-design.
Ecosystem Dominance: Moves by Tier 1 players (Nvidia, Google, OpenAI, Anthropic) that invalidate current cluster investments.
[Information Sources]
Tier 1 (Official): Google, NVIDIA, Meta, Apple, OpenAI, Anthropic official blogs; SEC filings; Arxiv papers; Strategic think tanks (CSIS, CSET).
Tier 2 (Elite Media): WSJ, Bloomberg, Reuters, SemiAnalysis, The Information, MIT Tech Review, Defense News (for geopolitical AI).
Tier 3A (Community/Expert): Reddit r/MachineLearning, Hugging Face, X posts from verified researchers (100k+ followers or verified organization).
[Behavioral Constraints]
No Inline LaTeX: Use bold text for all variables, symbols, and numbers (e.g., PAM4, 2.4 TB/s, n).
Language: Responses must be in professional, executive-level English prose.
Search Query Expansion: Always search for "AI regulatory conflict," "AI military restrictions," and "Future AI hardware roadmap" alongside specific technical terms.
[Output Structure (For each of the 5 items)]
Title, Source, Date (Status: Official / Press / Community Verified / Unverified Rumor)
[Overview]: Provide a concise, factual summary (under 10 lines).
[Strategic Impact]: Analyze why this matters for a CEO. Does it devalue current H100 clusters? Does it validate photonics investment? Does it introduce "Supply Chain Risk" labels from the government?
[Technical Deep Dive]:
Model: Benchmarks (MMLU, HumanEval), context window, inference cost.
Hardware: Bandwidth (GB/s), latency (ns), power efficiency (pJ/bit), modulation (PAM4).
[Visual Evidence]: Links to official demos, GitHub, or descriptions of visual evidence found in reports.
[Critical View]: Provide the "Cold Water." Mention power constraints, regulatory bottlenecks, or benchmark cherry-picking.
[Community Sentiment]: (If applicable) Summarize reactions from Reddit/X (Tier 3A).
[Closing Action]
Synthesize the 5 news items into a coherent trajectory for the AI ecosystem. Provide a heads-up on upcoming silicon releases (e.g., Blackwell updates, TPU v6), services, or technical milestones expected in the next quarter.
</프롬프트>"""

    tools = [
        {
            "name": "tavily_search",
            "description": "인터넷에서 최신 AI/하드웨어 뉴스를 검색하고 팩트를 교차 검증합니다.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "검색할 키워드 또는 문장"
                    }
                },
                "required": ["query"]
            }
        }
    ]

    messages = [
        {"role": "user", "content": raw_news}
    ]

    try:
        ranked_models = get_ranked_claude_sonnet_models()
        if not ranked_models:
            return "Error refining news with Claude: 사용 가능한 sonnet 모델을 찾을 수 없습니다."

        last_error = None
        for model_name in ranked_models:
            try:
                print(f"  [Claude 검증 시도 중] {model_name}")
                # messages를 매 모델 시도마다 초기화
                messages = [{"role": "user", "content": raw_news}]

                while True:
                    response = anthropic_client.messages.create(
                        model=model_name,
                        max_tokens=4000,
                        system=system_prompt,
                        tools=tools,
                        messages=messages
                    )

                    if response.stop_reason == "tool_use":
                        messages.append({"role": "assistant", "content": response.content})
                        tool_results = []
                        for block in response.content:
                            if block.type == "tool_use" and block.name == "tavily_search":
                                query = block.input.get("query", "")
                                print(f"  [Tavily 검색 실행] Query: {query}")
                                try:
                                    search_response = tavily_client.search(query)
                                    result_text = ""
                                    for r in search_response.get("results", []):
                                        result_text += f"- [{r.get('title', '')}]({r.get('url', '')})\n  {r.get('content', '')}\n\n"
                                    if not result_text:
                                        result_text = "검색 결과가 없습니다."
                                except Exception as search_err:
                                    result_text = f"검색 중 오류 발생: {str(search_err)}"
                                tool_results.append({
                                    "type": "tool_result",
                                    "tool_use_id": block.id,
                                    "content": result_text
                                })
                        messages.append({"role": "user", "content": tool_results})

                    else:
                        for block in response.content:
                            if hasattr(block, "text"):
                                print(f"✅ [성공] {model_name}")
                                return block.text
                        return ""

            except Exception as e:
                print(f"  ⚠️ [{model_name}] 실패: {e}")
                last_error = e
                continue

        return f"Error refining news with Claude: 모든 모델 시도 실패. 마지막 오류: {last_error}"

    except Exception as e:
        return f"Error refining news with Claude: {str(e)}"

def generate_html(refined_news):
    """
    Claude API를 호출하여 정제된 뉴스를 프리미엄 웹 리포트(HTML)로 변환합니다.
    """

    system_prompt = """ROLE: Senior Technical Web Designer (McKinsey Style)
CONTEXT: You are provided with 5 verified news items in the user message.
GOAL: Render these 5 items into a single, high-end HTML report.

CRITICAL DIRECTIVES:
1. OUTPUT ONLY THE HTML CODE. No greetings, no explanations, no protocol warnings.
2. RENDER ALL 5 NEWS ITEMS. Do not skip any. Each item must have its own <article> section.
3. PRESERVE ALL ORIGINAL TEXT. Copy the overview, strategic impact, and deep dive sections exactly as provided.
4. ADD VISUALIZATIONS (DYNAMIC SELECTION): DO NOT generate raw SVG code. For each news item, analyze the content and choose the most optimal visualization library between Chart.js and Mermaid.js:
   - OPTION A (Chart.js): Use this for quantitative data, comparisons, and metrics (e.g., bandwidth TB/s, power consumption MW, benchmark scores, cost reductions). Insert a <canvas> element with a unique ID and provide the inline <script> to render a sleek, professional chart (bar, line, or doughnut). Use modern, corporate colors.
   - OPTION B (Mermaid.js): Use this for structural, topological, or sequential data (e.g., hardware architecture, network topologies, timelines, ecosystem relationships). Place perfectly valid Mermaid syntax inside a <pre class="mermaid"></pre> tag.
   - Choose exactly ONE visualization type per article that best represents the core takeaway.
5. DARK MODE: Include a functional JS toggle for dark mode. Use Tailwind's 'dark' class strategy. Ensure your Chart.js text colors adapt nicely to dark mode if possible.
6. REQUIRED SCRIPTS: You MUST include the following libraries right before the closing </body> tag:
   <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
   <script type="module">
     import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
     mermaid.initialize({ startOnLoad: true, theme: 'base' });
   </script>

DESIGN SYSTEM:
- Use Tailwind CSS.
- Premium, professional typography (Inter).
- Clean whitespace, card-based layout, subtle shadows.
"""

    try:
        ranked_models = get_ranked_claude_sonnet_models()
        if not ranked_models:
            return "Error generating HTML with Claude: 사용 가능한 sonnet 모델을 찾을 수 없습니다."

        last_error = None
        for model_name in ranked_models:
            try:
                print(f"  [Claude HTML 시도 중] {model_name}")
                response = anthropic_client.messages.create(
                    model=model_name,
                    max_tokens=20000,
                    system=system_prompt + "\nIMPORTANT: Ensure the </html> tag is included. Be concise with SVG code to stay within limits.",
                    messages=[
                        {"role": "user", "content": f"Here is the verified data for 5 news items. Please render them into the HTML report as instructed:\n\n{refined_news}"}
                    ]
                )

                html_content = response.content[0].text
                match = re.search(r'(<!DOCTYPE html>.*</html>)', html_content, re.DOTALL | re.IGNORECASE)
                if match:
                    print(f"✅ [성공] {model_name}")
                    return match.group(1)

                if "<!DOCTYPE html>" in html_content:
                    html_content = re.sub(r'```(html)?', '', html_content).strip()
                    if not html_content.strip().endswith("</html>"):
                        html_content += "\n    </main>\n    <script>function toggleDarkMode(){document.documentElement.classList.toggle('dark');}</script>\n</body>\n</html>"
                    print(f"✅ [성공] {model_name}")
                    return html_content

                print(f"✅ [성공] {model_name}")
                return html_content

            except Exception as e:
                print(f"  ⚠️ [{model_name}] 실패: {e}")
                last_error = e
                continue

        return f"Error generating HTML with Claude: 모든 모델 시도 실패. 마지막 오류: {last_error}"

    except Exception as e:
        return f"Error generating HTML with Claude: {str(e)}"

def generate_html_gemini(refined_news):
    """
    Gemini Pro API를 호출하여 정제된 뉴스를 프리미엄 웹 리포트로 변환합니다.
    """
    # ✅ get_latest_gemini_pro_model → get_ranked_gemini_pro_models 로 교체
    ranked_models = get_ranked_gemini_pro_models()
    if not ranked_models:
        return "Error generating HTML with Gemini: 사용 가능한 pro 모델을 찾을 수 없습니다."

    system_prompt = """ROLE: Senior Technical Web Designer (McKinsey Style)
CONTEXT: You are provided with 5 verified news items in the user message.
GOAL: Render these 5 items into a single, high-end HTML report.

CRITICAL DIRECTIVES:
1. OUTPUT ONLY THE HTML CODE. No greetings, no explanations, no protocol warnings.
2. RENDER ALL 5 NEWS ITEMS. Do not skip any. Each item must have its own <article> section.
3. PRESERVE ALL ORIGINAL TEXT. Copy the overview, strategic impact, and deep dive sections exactly as provided.
4. ADD VISUALIZATIONS (DYNAMIC SELECTION): DO NOT generate raw SVG code. For each news item, analyze the content and choose the most optimal visualization library between Chart.js and Mermaid.js:
   - OPTION A (Chart.js): Use this for quantitative data, comparisons, and metrics (e.g., bandwidth TB/s, power consumption MW, benchmark scores, cost reductions). Insert a <canvas> element with a unique ID and provide the inline <script> to render a sleek, professional chart (bar, line, or doughnut). Use modern, corporate colors.
   - OPTION B (Mermaid.js): Use this for structural, topological, or sequential data (e.g., hardware architecture, network topologies, timelines, ecosystem relationships). Place perfectly valid Mermaid syntax inside a <pre class="mermaid"></pre> tag.
   - Choose exactly ONE visualization type per article that best represents the core takeaway.
5. DARK MODE: Include a functional JS toggle for dark mode. Use Tailwind's 'dark' class strategy. Ensure your Chart.js text colors adapt nicely to dark mode if possible.
6. REQUIRED SCRIPTS: You MUST include the following libraries right before the closing </body> tag:
   <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
   <script type="module">
     import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
     mermaid.initialize({ startOnLoad: true, theme: 'base' });
   </script>

DESIGN SYSTEM:
- Use Tailwind CSS.
- Premium, professional typography (Inter).
- Clean whitespace, card-based layout, subtle shadows.
"""

    last_error = None
    for model_name in ranked_models:
        try:
            print(f"  [Gemini HTML 시도 중] {model_name}")
            response = gemini_client.models.generate_content(
                model=model_name,
                contents=f"Render this data into the HTML report:\n\n{refined_news}",
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt
                )
            )
            html_content = response.text
            html_content = re.sub(r'```(html)?', '', html_content).strip()
            if "<!DOCTYPE html>" not in html_content:
                print("⚠️ [System] Gemini의 응답 형식이 불완전하여 보정을 시도합니다.")
            print(f"✅ [성공] {model_name}")
            return html_content

        except Exception as e:
            print(f"  ⚠️ [{model_name}] 실패: {e}")
            last_error = e
            continue

    return f"Error generating HTML with Gemini: 모든 모델 시도 실패. 마지막 오류: {last_error}"

if __name__ == "__main__":
    print("=======================================")
    print("🚀 전략 AI 뉴스 자동화 파이프라인 가동")
    print("=======================================")

    # 1단계: 제미나이로 뉴스 수집
    print("\n[1단계] Gemini Pro: 최신 뉴스 추출 중...")
    raw_news = fetch_ai_news()

    if "Error fetching AI news" in raw_news:
        print(f"\n❌ 1단계 실패! 제미나이 API 에러:\n{raw_news}")
        exit()

    with open("step1_gemini.txt", "w", encoding="utf-8") as f:
        f.write(raw_news)
    print("✅ 1단계 완료 (step1_gemini.txt 저장됨)")

    # 2단계: 클로드로 검증 및 정제
    print("\n[2단계] Claude: 팩트 체크 및 내용 보완 중...")
    refined_news = verify_and_refine_news(raw_news)
    with open("step2_claude.txt", "w", encoding="utf-8") as f:
        f.write(refined_news)
    print("✅ 2단계 완료 (step2_claude.txt 저장됨)")

    # 3단계: Claude HTML 시각화 보고서 생성
    print("\n[3단계] Claude: HTML 및 차트 렌더링 중...")
    final_html = generate_html(refined_news)
    with open("index.html", "w", encoding="utf-8") as f:
        f.write(final_html)
    print("✅ 3단계 완료 (index.html 저장됨)")

    # 4단계: Gemini HTML 시각화 보고서 생성 (비교 테스트)
    print("\n[4단계] Gemini Pro: HTML 및 차트 렌더링 중...")
    final_html_gemini = generate_html_gemini(refined_news)
    with open("index_gemini.html", "w", encoding="utf-8") as f:
        f.write(final_html_gemini)
    print("✅ 4단계 완료 (index_gemini.html 저장됨)")

    print("\n🎉 모든 프로세스 완료! 생성된 파일들을 확인해주세요.")