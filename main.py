import os
import time
import argparse
import re
from dotenv import load_dotenv
from google import genai
import anthropic
from tenacity import retry, stop_after_attempt, wait_exponential
import datetime
from github_utils import fetch_prompt_from_github

# Load environment variables
load_dotenv()

def return_none_on_error(retry_state):
    """Callback for tenacity when all retry attempts fail."""
    print(f"\nAction ultimately failed after {retry_state.attempt_number} attempts: {retry_state.outcome.exception()}")
    return None

def get_recent_archives(days=7):
    """
    Reads news content from the data/ directory for the last N days.
    """
    archive_data = ""
    archive_dir = "data"
    if not os.path.exists(archive_dir):
        return ""
    
    try:
        # Get list of files and filter by date-like names (YYYY-MM-DD.txt)
        files = [f for f in os.listdir(archive_dir) if re.match(r'\d{4}-\d{2}-\d{2}\.txt', f)]
        files.sort(reverse=True) # Most recent first
        
        today = datetime.datetime.now()
        count: int = 0
        for filename in files:
            file_date_str = filename.replace(".txt", "")
            file_date = datetime.datetime.strptime(file_date_str, "%Y-%m-%d")
            
            # Check if within window
            if (today - file_date).days <= days:
                file_path = os.path.join(archive_dir, filename)
                with open(file_path, "r", encoding="utf-8") as f:
                    archive_data += f"\n--- Report from {file_date_str} ---\n"
                    archive_data += f.read() + "\n"
                count = count + 1
            if count >= days:
                break
                
        if archive_data:
            return f"\n[REPORTS FROM LAST {days} DAYS - DO NOT REPEAT UNLESS THERE IS NEW PROGRESS]\n{archive_data}\n"
        return ""
    except Exception as e:
        print(f"Warning: Could not read archives: {e}")
        return ""

def get_latest_pro_model(client, require_agent=False):
    """
    Dynamically finds the latest available Gemini Pro model, 
    filtering out internal/preview codenames.
    If require_agent=True, selects deep-research agents.
    If require_agent=False, specifically excludes agents and selects generative models.
    """
    try:
        models = client.models.list()
        
        # Priority 1: Pick Deep Research Agent (For Phase 1)
        if require_agent:
            dr_models = [m.name for m in models if "deep-research-pro" in m.name.lower()]
            if dr_models:
                dr_models.sort(reverse=True)
                latest_dr = dr_models[0].replace("models/", "")
                print(f"Found latest specialized Deep Research agent: {latest_dr}")
                return latest_dr
            return "deep-research-pro-preview-12-2025"

        # Priority 2: Pick Standard Generative Model (For Phase 5 HTML)
        pro_models = []
        for m in models:
            name = m.name.lower()
            if "gemini" in name and "pro" in name and "vision" not in name:
                bad_keywords = ["flash", "nano", "banana", "vision", "latest", "preview", "customtools", "experimental", "deep-research"]
                if not any(bad in name for bad in bad_keywords):
                    pro_models.append(m.name)
        
        if pro_models:
            pro_models.sort(reverse=True)
            latest = pro_models[0].replace("models/", "")
            print(f"Automatically selected latest standard Pro model: {latest}")
            return latest

        return "gemini-2.0-pro-exp-02-05" # Generative fallback
    except Exception as e:
        print(f"Warning: Could not list models automatically: {e}")
        return "deep-research-pro-preview-12-2025" if require_agent else "gemini-2.0-pro-exp-02-05"

def get_latest_claude_model(client, flavor="sonnet"):
    """
    Dynamically finds the latest available Claude Sonnet model.
    """
    try:
        env_model = os.getenv("CLAUDE_MODEL")
        if env_model and env_model != "latest-sonnet":
            return env_model
        return "claude-3-5-sonnet-latest" 
    except Exception as e:
        print(f"Warning: Claude model selection failed: {e}")
        return "claude-3-5-sonnet-latest"

@retry(
    stop=stop_after_attempt(3), 
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry_error_callback=return_none_on_error
)
def validate_with_claude(content, custom_prompt="검증해주세요"):
    """
    Validates the research content using Claude.
    """
    api_key = os.getenv("CLAUDE_API_KEY")
    if not api_key:
        print("Warning: CLAUDE_API_KEY not found. Skipping validation.")
        return None
    
    # Initialize client with a generous timeout for large content
    client = anthropic.Anthropic(api_key=api_key, timeout=120.0) 
    model = get_latest_claude_model(client)
    
    print(f"Starting Claude action (Model: {model}, Prompt: {custom_prompt})...")
    # Removed internal try-except to allow @retry to catch errors
    message = client.messages.create(
        model=model,
        max_tokens=8192,  # Increased for translation
        messages=[
            {"role": "user", "content": f"{content}\n\n{custom_prompt}"}
        ]
    )
    return message.content[0].text

def run_deep_research(prompt, output_file="research_result.md", agent_id=None, previous_interaction_id=None):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "YOUR_API_KEY_HERE":
        print("Error: GEMINI_API_KEY not found in .env file.")
        return None, None

    research_agent = agent_id or os.getenv("RESEARCH_AGENT")
    client = genai.Client(api_key=api_key)

    if not research_agent or research_agent.lower() == "latest-pro":
        # Crucial: Request an 'agent' explicitly for Deep Research, not a generative model
        research_agent = get_latest_pro_model(client, require_agent=True)
    
    print(f"Starting Gemini Deep Research (Agent: {research_agent}, Continued: {previous_interaction_id is not None})...")
    
    try:
        # Crucial: Deep Research specific IDs are 'agents', while 'gemini-*' IDs are 'models'.
        # The API errors if the wrong parameter name is used.
        if "deep-research" in research_agent.lower():
            interaction = client.interactions.create(
                agent=research_agent, 
                input=prompt, 
                background=True,
                previous_interaction_id=previous_interaction_id
            )
        else:
            interaction = client.interactions.create(
                model=research_agent, 
                input=prompt, 
                background=True,
                previous_interaction_id=previous_interaction_id
            )
        print(f"Interaction ID: {interaction.id}")
        
        start_time = time.time()
        while True:
            interaction = client.interactions.get(interaction.id)
            if interaction.status == "completed":
                print("\nResearch Turn Completed!")
                # Collect all non-empty text from all outputs
                result_text = "\n".join([o.text for o in interaction.outputs if o.text])
                
                if not result_text:
                    print("Warning: API returned completed status but no text outputs were found.")
                    result_text = "No content returned from research agent."

                output_dir = os.path.dirname(output_file)
                if output_dir and not os.path.exists(output_dir):
                    os.makedirs(output_dir, exist_ok=True)
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(result_text)
                print(f"Results saved to {output_file} (Total length: {len(result_text)})")
                return result_text, interaction.id
            elif interaction.status == "failed":
                print(f"\nResearch failed: {interaction.error}")
                return None, interaction.id
            else:
                elapsed = int(time.time() - start_time)
                print(f"\rStatus: {interaction.status} (Elapsed: {elapsed}s)...", end="", flush=True)
                time.sleep(10)
    except Exception as e:
        print(f"An error occurred during API interaction: {e}")
        return None, None

def main():
    parser = argparse.ArgumentParser(description="Gemini Deep Research with GitHub Prompt")
    parser.add_argument("--url", help="GitHub URL of the prompt file")
    parser.add_argument("--agent", help="Gemini Research Agent ID (e.g., gemini-3.1-pro)")
    default_output = os.getenv("OUTPUT_FILE") or "trial/1.txt"
    parser.add_argument("--output", default=default_output, help="Output file name")
    
    args = parser.parse_args()
    url = args.url or os.getenv("PROMPT_URL")
    
    if not url:
        print("Required: --url <GitHub_Prompt_URL> or set PROMPT_URL in .env")
        return

    print(f"Using prompt from: {url}")
    prompt_content = fetch_prompt_from_github(url)
    
    if prompt_content:
        # Inject recent archives to avoid duplicates
        archives = get_recent_archives(days=7)
        if archives:
            print("Injecting recent archives for duplicate filtering...")
            prompt_content = archives + "\n" + prompt_content

        # Step 1: Initial Research
        initial_result, first_interaction_id = run_deep_research(prompt_content, args.output, args.agent)
        
        if initial_result and first_interaction_id:
            # Step 2: Claude Validation
            print("\n--- Phase 2: Claude Validation ---")
            feedback = validate_with_claude(initial_result)
            
            if feedback:
                feedback_file = os.getenv("FEEDBACK_FILE") or "trial/feedback.txt"
                feedback_dir = os.path.dirname(feedback_file)
                if feedback_dir and not os.path.exists(feedback_dir):
                    os.makedirs(feedback_dir, exist_ok=True)
                with open(feedback_file, "w", encoding="utf-8") as f:
                    f.write(feedback)
                print(f"Feedback saved to {feedback_file}")
                
                # Step 3: Refinement using feedback
                print("\n--- Phase 3: Gemini Refinement ---")
                refine_prompt = f"다음은 작성된 내용에 대한 검증 피드백입니다. 이 내용을 바탕으로 최종적으로 수정 및 보완해주세요:\n\n{feedback}"
                refined_output = os.getenv("REFINED_OUTPUT_FILE") or "trial/2.txt"
                
                refined_result, _ = run_deep_research(
                    refine_prompt, 
                    refined_output, 
                    args.agent, 
                    previous_interaction_id=first_interaction_id
                )
            else:
                print("Claude validation failed or skipped. Using initial research as refined result.")
                refined_result = initial_result
                refined_output = args.output
            
            if refined_result:
                # --- Phase 4: Claude Audit 2 & English Translation ---
                print("\n--- Phase 4: Claude Audit 2 & English Translation ---")
                translation_prompt = "구조를 유지한 채 최상의 영어 문장으로 최종 수정 및 번역해주세요."
                translated_result = validate_with_claude(refined_result, custom_prompt=translation_prompt)
                
                # If translation also fails, use the refined result (Korean) as a fallback for archival
                final_content_for_html = translated_result if translated_result else refined_result
                
                if final_content_for_html:
                    # Save to dated file: data/YYYY-MM-DD.txt
                    today = datetime.datetime.now().strftime("%Y-%m-%d")
                    archive_file = f"data/{today}.txt"
                    archive_dir = os.path.dirname(archive_file)
                    if archive_dir and not os.path.exists(archive_dir):
                        os.makedirs(archive_dir, exist_ok=True)
                    
                    with open(archive_file, "w", encoding="utf-8") as f:
                        f.write(final_content_for_html)
                    print(f"Archived content to {archive_file}")

                    # --- Phase 5: HTML Generation ---
                    print("\n--- Phase 5: HTML Generation ---")
                    html_prompt_url = os.getenv("HTML_PROMPT_URL")
                    if html_prompt_url:
                        html_prompt = fetch_prompt_from_github(html_prompt_url)
                        if html_prompt:
                            generate_html(final_content_for_html, html_prompt, "index.html")
                        else:
                            print("Warning: Failed to fetch HTML prompt.")
                    else:
                        print("Warning: HTML_PROMPT_URL not set in .env.")
                else:
                    print("Error: No content available for HTML generation.")
    else:
        print("Failed to fetch prompt.")

@retry(
    stop=stop_after_attempt(3), 
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry_error_callback=return_none_on_error
)
def generate_html(content, prompt, output_file="index.html"):
    """
    Generates index.html using Gemini Pro.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)
    model_id = get_latest_pro_model(client)
    
    print(f"Generating HTML using model: {model_id}...")
    # Removed internal try-except to allow @retry to catch errors
    response = client.models.generate_content(
        model=model_id,
        contents=[f"{prompt}\n{content}"]
    )
    html_content = response.text
    if "```html" in html_content:
        html_content = html_content.split("```html")[1].split("```")[0].strip()
    elif "```" in html_content:
        html_content = html_content.split("```")[1].split("```")[0].strip()
        
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"HTML generation complete. Saved to {output_file}")

if __name__ == "__main__":
    main()
