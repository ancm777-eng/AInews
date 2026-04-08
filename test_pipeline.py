name: Smoke Test Pipeline

on:
  workflow_dispatch:

permissions:
  contents: write

jobs:
  run-test:
    runs-on: ubuntu-latest
    timeout-minutes: 10
    env:
      TZ: Asia/Seoul
      FORCE_JAVASCRIPT_ACTIONS_TO_NODE24: "true" # Node.js 20 Deprecated 경고 해결을 위한 강제 설정

    steps:
      - name: Checkout code
        uses: actions/checkout@v4 # 최신 버전 유지
        with:
          fetch-depth: 0

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt

      - name: Run Smoke Test
        env:
          GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY }}
          CLAUDE_API_KEY: ${{ secrets.CLAUDE_API_KEY }}
        run: python test_pipeline.py

      - name: Upload Test Results
        if: always()
        uses: actions/upload-artifact@v4 # v4로 업데이트하여 Node.js 경고 해결
        with:
          name: smoke-test-artifacts
          path: |
            trial/
            data/
            index.html
            test_status.txt

      # 결과물 리포지토리 자동 업데이트(Commit & Push)
      - name: Commit and Push results
        if: always() # 스크립트 실행 중 503 에러가 발생해도 결과를 무조건 푸시
        run: |
          git config --global user.name "github-actions[bot]"
          git config --global user.email "github-actions[bot]@users.noreply.github.com"
          
          # ⭐️ 업데이트 확인을 무조건 보장하기 위한 더미 시간 파일 생성
          echo "Last Smoke Test Run: $(date +'%Y-%m-%d %H:%M:%S KST')" > test_status.txt
          
          # 변경사항 추적 (test_status.txt 포함)
          git add trial/ data/ index.html test_status.txt
          
          # 커밋 메시지에 실행 시간을 포함하여 리포지토리에서 바로 보이게 함
          if ! git diff --staged --quiet; then
            git commit -m "Test update: $(date +'%Y-%m-%d %H:%M:%S KST')"
            git pull --rebase origin main
            git push
          else
            echo "변경 사항이 없습니다."
          fi
