import requests
from tenacity import retry, stop_after_attempt, wait_exponential


def return_none_on_error(retry_state):
    """Callback for tenacity when all retry attempts fail."""
    print(f"\nAction ultimately failed after {retry_state.attempt_number} attempts: {retry_state.outcome.exception()}")
    return None


def get_raw_github_url(url):
    """
    Converts a standard GitHub URL to a raw content URL.
    Example: https://github.com/user/repo/blob/main/prompt.txt
    -> https://raw.githubusercontent.com/user/repo/main/prompt.txt
    """
    url = url.strip()
    if "raw.githubusercontent.com" in url:
        return url

    raw_url = url.replace("github.com", "raw.githubusercontent.com")
    raw_url = raw_url.replace("/blob/", "/").replace("/raw/", "/")
    return raw_url


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry_error_callback=return_none_on_error
)
def fetch_prompt_from_github(url):
    """
    Fetches the content of a file from GitHub with explicit UTF-8 encoding.
    """
    raw_url = get_raw_github_url(url)
    response = requests.get(raw_url, timeout=10)
    response.raise_for_status()
    response.encoding = "utf-8"
    return response.text
