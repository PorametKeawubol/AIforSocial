"""Rank ontology suggestions; similarity is never treated as a factual relation."""
import argparse
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from pipeline import ROOT, TERMS, dump

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='BAAI/bge-m3')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--max-length', type=int, default=512)
    args = parser.parse_args()
    graph = json.loads((ROOT / 'artifacts/graph.json').read_text())
    passages = [n for n in graph['nodes'] if n['label'] == 'Passage']
    anchors = [(typ, term) for typ, terms in TERMS.items() for term in terms]
    model = SentenceTransformer(args.model, device=args.device)
    model.max_seq_length = args.max_length
    texts = [p['props']['text'] for p in passages]
    lengths = [len(model.tokenizer.encode(t, truncation=False)) for t in texts]
    vectors = model.encode(texts, batch_size=args.batch_size, normalize_embeddings=True,
                           show_progress_bar=True)
    kwargs = {'prompt': 'Instruct: Match a police manual passage to its relevant topic or entity category.\nQuery:'} if 'Qwen3' in args.model else {}
    term_vectors = model.encode([x[1] for x in anchors], normalize_embeddings=True,
                                batch_size=args.batch_size, **kwargs)
    scores = vectors @ term_vectors.T
    suggestions = []
    for p, row in zip(passages, scores):
        suggestions.append({'passage_id': p['id'], 'pdf_page': p['props']['pdf_page'],
          'text': p['props']['text'], 'status': 'candidate_only', 'candidates': [
              {'type': anchors[i][0], 'term': anchors[i][1], 'cosine': float(row[i])}
              for i in np.argsort(row)[-3:][::-1]]})
    np.save(ROOT / 'artifacts/passage_embeddings.npy', vectors)
    dump(ROOT / 'artifacts/embedding_candidates.json', suggestions)
    dump(ROOT / 'artifacts/embedding_manifest.json', {
        'model': args.model, 'dimension': int(vectors.shape[1]), 'passage_ids': [p['id'] for p in passages],
        'max_length': args.max_length, 'device': args.device,
        'truncated_passage_ids': [p['id'] for p, length in zip(passages, lengths) if length > args.max_length],
        'notes': 'No calibrated threshold; no automatic entity merge or factual relation creation'})
    print('Saved embeddings and review candidates for', len(passages), 'passages')

if __name__ == '__main__':
    main()
