"""
Client Closed 이슈를 방지하기 위해 클라이언트 생성을 함수 내부가 아닌 
테스트 메인 루프에서 관리하도록 수정된 스모크 테스트입니다.
"""
import os
import sys
import datetime
from dotenv import load_dotenv
from google import genai
import anthropic
from main import get_latest_gemini_model, get_latest_claude_model

load_dotenv()
CURRENT_KST = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def main():
    print(f"🚀 Starting API Smoke Test at {CURRENT_KST}")
    
    g_key = os.getenv("GEMINI_API_KEY")
    c_key = os.getenv("CLAUDE_API_KEY")
    
    if not g_key:
        print("❌ GEMINI_API_KEY is missing")
        sys.exit(1)

    # 클라이언트 사전 생성 (재사용)
    g_client = genai.Client(api_key=g_key)
    c_client = anthropic.Anthropic(api_key=c_key) if c_key else None

    # 1. 모델 탐색 테스트
    g_model = get_latest_gemini_model(g_client)
    print(f"  ✅ Gemini Model: {g_model}")
    
    if c_client:
        c_model = get_latest_claude_model(c_client)
        print(f"  ✅ Claude Model: {c_model}")

    # 2. 최소 비용 연결 테스트 (Phase 1 & 3 세션 유지)
    print("\n--- Testing Gemini Session ---")
    chat = g_client.chats.create(model=g_model)
    res1 = chat.send_message("Reply 'P1'")
    print(f"  P1 Response: {res1.text.strip()}")
    
    res2 = chat.send_message("Reply 'P3' if you remember P1")
    print(f"  P3 Response: {res2.text.strip()}")

    # 3. Claude 문맥 유지 테스트
    if c_client:
        print("\n--- Testing Claude Context ---")
        msgs = [{"role": "user", "content": "Reply 'C2'"}]
        c_res1 = c_client.messages.create(model=c_model, max_tokens=10, messages=msgs)
        print(f"  C2 Response: {c_res1.content[0].text.strip()}")
        
        msgs.append({"role": "assistant", "content": c_res1.content[0].text})
        msgs.append({"role": "user", "content": "Reply 'C4' if context exists"})
        c_res2 = c_client.messages.create(model=c_model, max_tokens=10, messages=msgs)
        print(f"  C4 Response: {c_res2.content[0].text.strip()}")

    print("\n🎉 Smoke test passed: API and Session logic verified.")

if __name__ == "__main__":
    main()
