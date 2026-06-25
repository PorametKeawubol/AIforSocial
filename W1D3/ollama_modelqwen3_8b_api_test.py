import requests


URL = "http://127.0.0.1:11434/api/generate"


payload = {
    "model": "qwen3:8b", #qwen3:8b , gemma3:12b , llama3.1:8b
    "prompt": "ประเทศไทยมีกี่จังหวัด ",
    "stream": False,
    
}


response = requests.post(URL, json=payload, timeout=300)
response.raise_for_status()

print(response.json()["response"])

