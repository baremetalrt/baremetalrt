import requests
import json

API_URL = "http://localhost:8000/v1/completions"

payload = {
    "prompt": "What is the capital of France?",
    "max_tokens": 32
}
headers = {"Content-Type": "application/json"}

try:
    response = requests.post(API_URL, headers=headers, data=json.dumps(payload))
    response.raise_for_status()
    data = response.json()
    print("API response:")
    print(json.dumps(data, indent=2))
    # Check OpenAI schema
    assert "choices" in data and isinstance(data["choices"], list), "Missing or invalid 'choices' field"
    # Check model field
    assert data.get("model") == "meta-llama/Llama-2-7b-chat-hf", "Model name mismatch"
    print("Test passed: API returned a valid OpenAI-style response from Llama 2.")
except Exception as e:
    print(f"Test failed: {e}")
