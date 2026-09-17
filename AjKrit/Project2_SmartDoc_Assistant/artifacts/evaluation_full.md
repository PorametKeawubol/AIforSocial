# SmartDoc Evaluation

Dataset: `/home/porametk/Desktop/AIforSocial/AIforSocial/AjKrit/Project2_SmartDoc_Assistant/eval/questions.jsonl`

| Metric | Value | Target | Interpretation |
|---|---:|---:|---|
| Retrieval Hit@k | 1.000 | 1.000 | target section appears in retrieved context |
| MRR | 1.000 | 1.000 | target section ranking quality |
| SBERT cosine / reference | 0.833 | ≥ 0.700 | higher is better; inspect examples with the aggregate |
| Answer relevance / SBERT | 0.756 | ≥ 0.700 | higher is better; inspect examples with the aggregate |
| Faithfulness / semantic | 1.000 | ≥ 0.700 | higher is better; inspect examples with the aggregate |
| Faithfulness / lexical proxy | 0.812 | ≥ 0.700 | higher is better; inspect examples with the aggregate |
| BERTScore Precision | 0.766 | ≥ 0.700 | higher is better; inspect examples with the aggregate |
| BERTScore Recall | 0.850 | ≥ 0.700 | higher is better; inspect examples with the aggregate |
| BERTScore F1 | 0.805 | ≥ 0.700 | higher is better; inspect examples with the aggregate |
| Faithfulness / LLM judge | 0.981 | ≥ 0.700 | higher is better; inspect examples with the aggregate |
| Answer relevance / LLM judge | 1.000 | ≥ 0.700 | higher is better; inspect examples with the aggregate |
