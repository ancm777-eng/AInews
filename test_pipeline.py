import os
import unittest
from unittest.mock import patch, MagicMock

# Import the original main script
import main

def test_pipeline_dry_run():
    """
    Mocks all API calls and runs the full 4-stage pipeline logic.
    """
    print("=== Pipeline Dry Run (Mock Mode) Start ===\n")
    
    # Define dummy responses
    mock_prompt = "이것은 테스트용 프롬프트입니다."
    mock_research_v1 = "# 초기 리서치 결과\n데이터 내용..."
    mock_feedback = "이 부분은 좋습니다. 하지만 수치가 부족하니 보완해주세요."
    mock_research_v2 = "# 최종 리서치 결과 (보완됨)\n수정된 데이터 내용..."
    mock_html = "<html><body><h1>완성된 뉴스</h1></body></html>"
    
    # Mocking environment variables
    with patch.dict(os.environ, {
        "PROMPT_URL": "https://github.com/test/prompt.txt",
        "HTML_PROMPT_URL": "https://github.com/test/html.txt",
        "GEMINI_API_KEY": "mock_key",
        "CLAUDE_API_KEY": "mock_key",
        "OUTPUT_FILE": "trial/test_1.txt",
        "FEEDBACK_FILE": "trial/test_feedback.txt",
        "REFINED_OUTPUT_FILE": "trial/test_2.txt"
    }):
        # Mock fetch_prompt_from_github
        with patch('main.fetch_prompt_from_github', return_value=mock_prompt) as mock_fetch:
            # Mock run_deep_research
            # We need it to return (text, interaction_id)
            with patch('main.run_deep_research') as mock_research:
                mock_research.side_effect = [
                    (mock_research_v1, "int_123"), # Stage 1
                    (mock_research_v2, "int_456")  # Stage 3
                ]
                
                # Mock validate_with_claude
                with patch('main.validate_with_claude', return_value=mock_feedback) as mock_val:
                    
                    # Mock generate_html
                    with patch('main.generate_html') as mock_gen_html:
                        
                        # Execute main orchestration
                        print("단계 1: 프롬프트 로딩 및 초기 리서치 시물레이션")
                        main.main()
                        
                        # Verify the flow
                        print("\n=== 검증 결과 ===")
                        print(f"프롬프트 호출 횟수: {mock_fetch.call_count}")
                        print(f"제미나이 리서치 호출 횟수: {mock_research.call_count} (초기 + 보완)")
                        print(f"클로드 검증 호출 여부: {'성공' if mock_val.called else '실패'}")
                        print(f"HTML 생성 호출 여부: {'성공' if mock_gen_html.called else '실패'}")

    print("\n=== Pipeline Dry Run Complete ===")

if __name__ == "__main__":
    # Ensure trial directory exists for mock output (if any)
    if not os.path.exists("trial"):
        os.makedirs("trial")
        
    test_pipeline_dry_run()
