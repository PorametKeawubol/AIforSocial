import time
from pathlib import Path

import requests


URL = "http://127.0.0.1:11434/api/generate"
MODELS = ["qwen3:8b", "gemma3:12b", "llama3.1:8b"]
QUESTIONS = [
    "ประเทศไทยมีกี่จังหวัด",
    "อธิบายการทำงานของ VLAN และ OSPF",
    "เขียน Python Bubble Sort",
    "สรุปเนื้อหา Animal Farm",
    "วิเคราะห์ผลลัพธ์ Nmap Scan",
]
OUTPUT_PATH = Path(__file__).with_name("ollama_model_compare_results.md")


def markdown_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", "<br>")


def generate(model: str, prompt: str) -> tuple[str, float]:
    start = time.perf_counter()
    response = requests.post(
        URL,
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "think": False,
            "options": {"num_predict": 384},
        },
        timeout=600,
    )
    response.raise_for_status()
    elapsed = time.perf_counter() - start
    return response.json().get("response", ""), elapsed


def main() -> None:
    lines = [
        "# ผลการทดลองเปรียบเทียบ Ollama Models",
        "",
        "| คำถาม | โมเดล | เวลา (วินาที) | คำตอบ |",
        "| --- | --- | ---: | --- |",
    ]

    for question in QUESTIONS:
        for model in MODELS:
            print(f"Running {model}: {question}")
            try:
                answer, elapsed = generate(model, question)
            except requests.RequestException as exc:
                answer = f"ERROR: {exc}"
                elapsed = 0.0

            lines.append(
                f"| {markdown_cell(question)} | {model} | {elapsed:.2f} | {markdown_cell(answer)} |"
            )

    lines.extend(
        [
            "",
            "## ตารางบันทึกผล",
            "",
            "| หัวข้อ | Qwen3:8B | Gemma3:12B | Llama3.1:8B |",
            "| --- | --- | --- | --- |",
            "| ความถูกต้อง |  |  |  |",
            "| ภาษาไทย |  |  |  |",
            "| Coding |  |  |  |",
            "| ความเร็ว |  |  |  |",
            "| ความครบถ้วน |  |  |  |",
        ]
    )

    OUTPUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
