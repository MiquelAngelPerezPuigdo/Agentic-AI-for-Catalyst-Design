import requests
import time
from config import OPENROUTER_API_KEY

def call_openrouter(model, text_prompt, system_prompt=None, temperature=0.2):
    """Handles API calls to OpenRouter with retries and backoff."""
    if not OPENROUTER_API_KEY:
        return "ERROR: API Key not found. Check your .env file."

    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
    
    # 1. Dynamically build the message list based on whether a system prompt exists
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": text_prompt})

    data = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": 8000 # CRITICAL: Added high token limit for long Benchmark 2 outputs
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
                
            # 3. Added safe extraction from your dirty notebook logic
            content = response_json['choices'][0]['message'].get('content')
            if content is None: return "API_RETURNED_NULL"
            return content
        
        except requests.exceptions.Timeout:
            print(f"\n[!] Request timed out for {model}. Retrying...")
            time.sleep(base_delay * (2 ** attempt))
        except Exception as e:
            print(f"\n[!] Unexpected Error for {model}: {e}")
            time.sleep(base_delay * (2 ** attempt))
            
    return "API_ERROR"