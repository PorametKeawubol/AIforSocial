import os

from openai import OpenAI

MODEL = os.getenv("LLM_MODEL", "Qwen/Qwen3-0.6B")
BASE_URL = os.getenv("LLM_BASE_URL", "http://127.0.0.1:8000/v1")

client = OpenAI(api_key="EMPTY", base_url=BASE_URL)

response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {
            "role": "user",
            "content": "อธิบายความแตกต่างระหว่าง Machine Learning และ Deep Learning เป็นภาษาไทย /no_think",
        }
    ],
    stream=True,
    temperature=0.2,
    max_tokens=256,
)

for chunk in response:
    content = chunk.choices[0].delta.content
    if content is not None:
        print(content, end="", flush=True)

print()
