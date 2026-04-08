import requests
import time
from config import OPENROUTER_API_KEY

def call_openrouter(model, text_prompt, temperature=0.2):
    """Handles API calls to OpenRouter with retries and backoff."""
    if not OPENROUTER_API_KEY:
        return "ERROR: API Key not found. Check your .env file."

    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
    data = {
        "model": model,
        "messages": [{"role": "user", "content": text_prompt}],
        "temperature": temperature
    }
    
    max_retries = 5
    base_delay = 2 
    
    for attempt in range(max_retries):
        try:
            # 120s timeout to allow reasoning models (like o1 or r1) time to think
            r = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data, timeout=120)
            response_json = r.json()
            
            if 'error' in response_json:
                error_info = response_json['error']
                print(f"\n[!] API Error for {model} (Attempt {attempt + 1}/{max_retries}): {error_info}")
                if attempt == max_retries - 1:
                    return "API_ERROR" 
                time.sleep(base_delay * (2 ** attempt))
                continue 
                
            return response_json['choices'][0]['message']['content']
        
        except requests.exceptions.Timeout:
            print(f"\n[!] Request timed out for {model}. Retrying...")
            time.sleep(base_delay * (2 ** attempt))
        except Exception as e:
            print(f"\n[!] Unexpected error: {e}")
            time.sleep(base_delay * (2 ** attempt))
            
    return "API_ERROR"