import requests
import time
from config import OPENROUTER_API_KEY, GOOGLE_API_KEY

try:
    from config import USE_PROMPT_CACHING
except ImportError:
    USE_PROMPT_CACHING = False


def _supports_explicit_cache_control(model):
    """Anthropic models (via OpenRouter) require explicit `cache_control` breakpoints.
    Other providers (OpenAI, Grok, DeepSeek, etc.) cache automatically and need no markers."""
    return "anthropic" in model.lower() or "claude" in model.lower()


def _build_cached_messages(model, text_prompt, system_prompt):
    """Build the OpenRouter `messages` list, adding Anthropic `cache_control` breakpoints
    on the large static blocks (system prompt and user prefix) when caching is enabled.

    The system and user prompts are byte-identical across the repeated iterations of a
    benchmark combination, so marking them as ephemeral cache breakpoints lets the 4
    follow-up calls read the prefill from cache instead of re-paying for it."""
    use_cache = USE_PROMPT_CACHING and _supports_explicit_cache_control(model)

    messages = []
    if system_prompt:
        if use_cache:
            messages.append({
                "role": "system",
                "content": [{
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }],
            })
        else:
            messages.append({"role": "system", "content": system_prompt})

    if use_cache:
        messages.append({
            "role": "user",
            "content": [{
                "type": "text",
                "text": text_prompt,
                "cache_control": {"type": "ephemeral"},
            }],
        })
    else:
        messages.append({"role": "user", "content": text_prompt})

    return messages

def call_gemini_direct(model, text_prompt, system_prompt=None, temperature=0.2):
    """Directly calls Google Gemini API with retries and fallback."""
    if not GOOGLE_API_KEY or GOOGLE_API_KEY.strip() == "":
        raise ValueError("CRITICAL ERROR: GOOGLE_API_KEY is not set or empty, but Gemini model was requested. Stopping campaign!")
    
    model_name = model.split("/")[-1] # strip google/ or other provider prefix
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={GOOGLE_API_KEY}"
    
    # Build payload
    payload = {
        "contents": [{"parts": [{"text": text_prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": 8000
        }
    }
    if system_prompt:
        payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}
        
    headers = {"Content-Type": "application/json"}
    max_retries = 5
    base_delay = 2
    
    for attempt in range(max_retries):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=120)
            
            # If 404 or 400 (unrecognized model), attempt a fallback
            if r.status_code in [400, 404]:
                if model_name == "gemini-3.5-flash":
                    print(f"[!] {model_name} failed with status {r.status_code}. Falling back to gemini-2.5-flash...")
                    model_name = "gemini-2.5-flash"
                    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={GOOGLE_API_KEY}"
                    continue
                elif model_name == "gemini-2.5-flash":
                    print(f"[!] {model_name} failed with status {r.status_code}. Falling back to gemini-1.5-flash...")
                    model_name = "gemini-1.5-flash"
                    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={GOOGLE_API_KEY}"
                    continue
            
            r_json = r.json()
            if 'error' in r_json:
                error_info = r_json['error']
                print(f"\n[!] Gemini API Error for {model_name} (Attempt {attempt + 1}/{max_retries}): {error_info}")
                time.sleep(base_delay * (2 ** attempt))
                continue
            
            candidates = r_json.get('candidates', [])
            if not candidates:
                return "API_RETURNED_NULL"
            
            parts = candidates[0].get('content', {}).get('parts', [])
            if not parts:
                return "API_RETURNED_NULL"
                
            text = parts[0].get('text')
            if text is None:
                return "API_RETURNED_NULL"
            return text
            
        except Exception as e:
            print(f"\n[!] Unexpected Error calling Gemini API: {e}")
            time.sleep(base_delay * (2 ** attempt))
            
    return "API_ERROR"

def call_openrouter(model, text_prompt, system_prompt=None, temperature=0.2):
    """Handles API calls to OpenRouter, or routes to Gemini directly if GOOGLE_API_KEY is available."""
    if "gemini" in model.lower():
        if not GOOGLE_API_KEY or GOOGLE_API_KEY.strip() == "":
            raise ValueError(f"CRITICAL ERROR: GOOGLE_API_KEY is not set or empty, but Gemini model {model} was requested. Stopping campaign!")
        return call_gemini_direct(model, text_prompt, system_prompt, temperature)
        
    if not OPENROUTER_API_KEY:
        return "ERROR: API Key not found. Check your .env file."

    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
    
    # 1. Dynamically build the message list (adds Anthropic cache_control breakpoints
    #    on the large static blocks when prompt caching is enabled).
    messages = _build_cached_messages(model, text_prompt, system_prompt)

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