"""Turn saved RAG runs into a readable lab report; does not call the LLM."""
import argparse
import json
from pathlib import Path

from common import ROOT, read_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'results' / 'comparison.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'results' / 'lab5_report.md')
    args = parser.parse_args()
    saved = read_json(args.input)
    lines = ['# Lab 5 — ผลการทดลอง Graph RAG', '',
             f"เวลาบันทึก: {saved['created_at']}", '',
             'รายงานนี้รวบรวมผลที่รันจริง คำตอบ LLM เป็นผลการทดลอง ต้องตรวจว่าแต่ละข้ออ้างตรงกับหลักฐานที่อ้างถึง', '',
             'ทั้งสองระบบใช้ Top-K Chunks ชุดเดียวกัน โมเดลและการตั้งค่าเดียวกัน '
             'Graph RAG เพิ่มกราฟ โดยใช้ชื่อ/รหัสที่ตรงกับคำถามเป็นจุดเริ่มต้นเมื่อพบ '
             'มิฉะนั้นใช้ Entity ที่ได้จาก Vector Search ดังนั้นผลเปรียบเทียบรวมประโยชน์ของการจับชื่อแบบตรงตัวด้วย', '',
             '## Dataset', '', '```json', json.dumps(saved['dataset'], indent=2), '```', '',
             '## ภาพรวมการทดลอง', '',
             '| คำถาม | Top-K | Graph paths ใน Prompt | จำกัดผล | Vector RAG (วินาที) | Graph RAG (วินาที) |',
             '|---|---:|---:|---|---:|---:|']
    for run in saved['results']:
        lines.append(f'| {run["question"]} | {len(run["chunks"])} | '
                     f'{run["graph_paths_in_prompt"]} | {run["context_truncated"]} | '
                     f'{run.get("vector_rag", {}).get("duration_seconds", "—")} | '
                     f'{run.get("graph_rag", {}).get("duration_seconds", "—")} |')
    lines += ['', 'เวลาคือเวลาที่ Ollama รายงานสำหรับการสร้างคำตอบ '
              'อาจรวมการโหลดโมเดล จึงไม่ใช่ benchmark ความเร็วที่ควบคุมครบทุกปัจจัย', '']
    for i, run in enumerate(saved['results'], 1):
        lines += [f'## คำถาม {i}: {run["question"]}', '',
                  '### Vector Search', '', '| Evidence | Entity | Score |', '|---|---|---:|']
        for rank, chunk in enumerate(run['chunks'], 1):
            lines.append(f'| V{rank} | [{chunk["entity_name"]}]({chunk["url"]}) | {chunk["score"]:.4f} |')
        lines += ['', '### Graph Retrieval', '',
                  'จุดเริ่มต้น: ' + ', '.join(f'{s["name"]} ({s["source"]})' for s in run['graph']['seeds']), '',
                  '```cypher', run['graph']['cypher'], '```', '',
                  'พารามิเตอร์:', '', '```json',
                  json.dumps(run['graph']['parameters'], indent=2), '```', '',
                  '### Vector Context', '', '```text', run['vector_context'], '```', '',
                  '### Graph Context', '', '```text', run['graph_context'], '```', '']
        for key, title in [('vector_rag', 'Vector RAG'), ('graph_rag', 'Graph RAG')]:
            if key not in run:
                continue
            output = run[key]
            lines += [f'### คำตอบ {title}', '', output['answer'], '',
                      f'โมเดล: `{output["model"]}`; สาเหตุจบ: `{output.get("done_reason")}`', '',
                      f'รหัสอ้างอิงที่ไม่มีใน Context: {output["invalid_citation_ids"] or "ไม่พบ"}', '',
                      'การตรวจรหัสอ้างอิงไม่ได้ยืนยันว่าข้อความที่อ้างถูกต้อง ต้องอ่านเทียบหลักฐานข้างต้น', '']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'Saved {args.output}')


if __name__ == '__main__':
    main()
