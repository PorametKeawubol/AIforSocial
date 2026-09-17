"""Small, agent-authored section-linking evaluation, not a KG accuracy benchmark."""
import argparse
import importlib.metadata
import json
import platform
import resource
import time
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from pipeline import ROOT, dump, uid

MODELS = {'bge-m3': 'BAAI/bge-m3', 'e5-large': 'intfloat/multilingual-e5-large',
          'qwen3-4b': 'Qwen/Qwen3-Embedding-4B', 'qwen3-8b': 'Qwen/Qwen3-Embedding-8B'}
# Agent-written paraphrases of the listed manual headings. No training/tuning split.
CASES = [
 ('ญาติหายตัวไป อยากให้เจ้าหน้าที่ช่วยติดตาม', '1.4'),
 ('รถจักรยานยนต์ถูกขโมย จะรายงานเรื่องนี้อย่างไร', '1.5'),
 ('ปืนที่มีทะเบียนของฉันหายไป', '1.6'),
 ('มีคนทำเอกสารปลอมขึ้นมา ต้องไปแจ้งเรื่องอะไร', '1.8'),
 ('ถูกหลอกเอาเงินไป อยากร้องทุกข์เรื่องฉ้อโกง', '1.9'),
 ('มีคนทุบทำลายข้าวของจนเสียหาย', '1.12'),
 ('ต้องการยื่นหลักทรัพย์ขอปล่อยผู้ต้องหาชั่วคราว', '1.17'),
 ('ถูกจับกุมแล้วมีสิทธิอะไรบ้าง', '1.18'),
 ('อยากได้เอกสารรับรองว่ามีความประพฤติดี', '1.19'),
 ('จะจัดงานจุดพลุ ต้องขออนุญาตอย่างไร', '2.2'),
 ('ต้องการใช้ลำโพงกระจายเสียงในงาน', '2.5'),
 ('อยากทำใบอนุญาตขับขี่มอเตอร์ไซค์', '2.9'),
 ('ดูแลบ้านอย่างไรไม่ให้โจรเข้ามาขโมยของ', '3.1'),
 ('ระวังอย่างไรไม่ให้ลูกถูกคนร้ายพาตัวไป', '3.5'),
 ('ป้องกันรถถูกขโมยได้อย่างไร', '3.6'),
 ('มีพายุเข้า ต้องปฏิบัติตัวอย่างไรให้ปลอดภัย', '4.2'),
 ('ได้รับประกาศเตือนสึนามิ ควรทำอย่างไร', '4.4'),
 ('ต้องการยื่นฟ้องหน่วยงานรัฐต่อศาลปกครอง', '6.1'),
 ('กฎเกี่ยวกับบุหรี่และการสูบบุหรี่', '6.4'),
 ('เจ้าหนี้ตามทวงเงิน มีกฎหมายควบคุมอย่างไร', '6.7'),
]

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', choices=MODELS, required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--threads', type=int, default=6)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    graph = json.loads((ROOT / 'artifacts/graph.json').read_text())
    nodes = {n['id']: n for n in graph['nodes']}
    sections = [n for n in graph['nodes'] if n['label'] == 'Section']
    section_numbers = [n['props']['section_number'] for n in sections]
    assert all(target in section_numbers for _, target in CASES)
    texts = []
    for section in sections:
        passages = [nodes[e['target']]['props']['text'] for e in graph['edges']
                    if e['source'] == section['id'] and e['type'] == 'HAS_PASSAGE']
        texts.append(section['props']['name'] + '\n' + '\n'.join(passages)[:800])
    queries = [case[0] for case in CASES]
    out = ROOT / 'artifacts/benchmark'
    out.mkdir(exist_ok=True)
    dataset = {'sections': list(zip(section_numbers, texts)), 'cases': CASES,
               'authorship': 'Agent-authored exploratory set; requires independent human review'}
    dump(out / 'dataset.json', dataset)
    model_id = MODELS[args.model]
    kwargs = {}
    if args.model.startswith('qwen'):
        kwargs['model_kwargs'] = {'torch_dtype': torch.bfloat16, 'attn_implementation': 'sdpa'}
        kwargs['tokenizer_kwargs'] = {'padding_side': 'left'}
    start = time.perf_counter()
    model = SentenceTransformer(model_id, device=args.device, **kwargs)
    load_seconds = time.perf_counter() - start
    model.max_seq_length = 512
    query_prompt, doc_prompt = '', ''
    if args.model == 'e5-large':
        query_prompt, doc_prompt = 'query: ', 'passage: '
    elif args.model.startswith('qwen'):
        query_prompt = 'Instruct: Given a Thai citizen request, retrieve the relevant section of the police citizen manual.\nQuery: '
    if args.device.startswith('cuda'):
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    dv = model.encode(texts, prompt=doc_prompt, batch_size=args.batch_size,
                      normalize_embeddings=True, show_progress_bar=True)
    qv = model.encode(queries, prompt=query_prompt, batch_size=args.batch_size,
                      normalize_embeddings=True, show_progress_bar=True)
    if args.device.startswith('cuda'):
        torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    scores = qv @ dv.T
    rows = []
    for (query, target), row in zip(CASES, scores):
        order = np.argsort(-row)
        rank = next(i + 1 for i, index in enumerate(order) if section_numbers[index] == target)
        rows.append({'query': query, 'target': target, 'rank': rank,
                     'top3': [{'section': section_numbers[i], 'cosine': float(row[i])} for i in order[:3]]})
    result = {'model': model_id, 'dataset_sha256': uid(json.dumps(dataset, ensure_ascii=False)),
        'accuracy_at_1': sum(r['rank'] == 1 for r in rows) / len(rows),
        'hit_at_3': sum(r['rank'] <= 3 for r in rows) / len(rows),
        'mrr': sum(1 / r['rank'] for r in rows) / len(rows),
        'load_seconds_including_download': load_seconds, 'encode_seconds': seconds,
        'peak_process_rss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        'peak_gpu_allocated_mib': torch.cuda.max_memory_allocated() / 2**20 if args.device.startswith('cuda') else None,
        'device': args.device, 'cpu': platform.processor(), 'threads': args.threads,
        'dtype': str(next(model.parameters()).dtype), 'dimension': int(dv.shape[1]),
        'truncated_sections': [section_numbers[i] for i, text in enumerate(texts)
                  if len(model.tokenizer.encode(doc_prompt + text, truncation=False)) > 512],
        'versions': {name: importlib.metadata.version(name) for name in ['torch','transformers','sentence-transformers']},
        'rows': rows, 'limitation': '20 agent-written cases, single run; not relation extraction accuracy or independent test set'}
    dump(out / f'{args.model}.json', result)
    np.save(out / f'{args.model}.npy', dv)
    print(json.dumps({k:v for k,v in result.items() if k != 'rows'}, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
