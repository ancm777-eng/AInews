import requests
import re
from tenacity import retry, stop_after_attempt, wait_exponential

def get_raw_github_url(url):
    """
    Converts a standard GitHub URL to a raw content URL.
    Example: https://github.com/user/repo/blob/main/prompt.txt 
    -> https://raw.githubusercontent.com/user/repo/main/prompt.txt
    """
    if "raw.githubusercontent.com" in url:
        return url
    
    # Replace github.com with raw.githubusercontent.com and remove /blob/
    raw_url = url.replace("github.com", "raw.githubusercontent.com")
    raw_url = raw_url.replace("/blob/", "/")
    return raw_url

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def fetch_prompt_from_github(url):
    """
    Fetches the content of a file from GitHub.
    """
    raw_url = get_raw_github_url(url)
    try:
        response = requests.get(raw_url, timeout=10)
        response.raise_for_status()
        return response.text
    except Exception as e:
        print(f"Error fetching prompt from GitHub: {e}")
        return None
