"""Worksheet demo: manual embeddings, fixed retrieval, and prompt construction.

Run: .venv/bin/python code.py
The graph is illustrative lab data, not a medical knowledge source.
"""

import numpy as np
import networkx as nx


class QueryRouter:
    def __init__(self, input_dim=4):
        if input_dim != 4:
            raise ValueError("The worksheet weight matrix requires input_dim=4")
        self.W = np.array([
            [0.8, -0.2, 0.1, 0.1],  # Entity Expert
            [0.6, 0.4, 0.2, 0.0],   # Relation Expert
            [0.1, 0.8, 0.9, 0.7],   # Subgraph Expert
        ])

    def route(self, h_q: np.ndarray) -> tuple[np.ndarray, str]:
        h_q = np.asarray(h_q, dtype=float)
        if h_q.shape != (4,) or not np.isfinite(h_q).all():
            raise ValueError("h_q must contain exactly four finite numbers")
        logits = self.W @ h_q
        exp_logits = np.exp(logits - np.max(logits))
        weights = exp_logits / exp_logits.sum()
        query_type = "Complex (Multi-hop)" if weights[2] > 0.4 else "Simple (1-hop)"
        return weights, query_type


class KnowledgeGraphExperts:
    def __init__(self):
        # Directed edges preserve subject -> object when rendering triples.
        self.graph = nx.DiGraph()
        self.graph.add_edge("Aspirin", "Pain", relation="treats")
        self.graph.add_edge("Aspirin", "COX-2", relation="inhibits")
        self.graph.add_edge("COX-2", "Inflammation", relation="causes")

    def entity_expert(self, query_emb: np.ndarray) -> list:
        return ["Aspirin", "COX-2"]

    def relation_expert(self, query_emb: np.ndarray) -> list:
        return [("Aspirin", "treats", "Pain")]

    def subgraph_expert(self, query_emb: np.ndarray) -> nx.DiGraph:
        return self.graph.subgraph(["Aspirin", "COX-2", "Inflammation"])



class MoERAGPipeline:
    def __init__(self):
        self.router = QueryRouter()
        self.experts = KnowledgeGraphExperts()
        

    def process_query(self, query_text: str, h_q: np.ndarray, sparse=True) -> dict:
        weights, q_type = self.router.route(h_q)
        context_parts = []
        called_experts = []
        names = ["Entity", "Relation", "Subgraph"]
        retrievers = [self.experts.entity_expert, self.experts.relation_expert,
                      self.experts.subgraph_expert]
        for name, weight, retrieve in zip(names, weights, retrievers):
            # Task 2: gate BEFORE retrieval to avoid unnecessary computation.
            if sparse and weight <= 0.3:
                continue
            output = retrieve(h_q)
            called_experts.append(name)
            # Dense baseline follows the worksheet: call all, retain alpha > 0.2.
            if not sparse and weight <= 0.2:
                continue
            if name == "Entity":
                context_parts.append(f"Entities: {', '.join(output)}")
            elif name == "Relation":
                triples = ", ".join(f"{s}-{p}->{o}" for s, p, o in output)
                context_parts.append(f"Relations: {triples}")
            else:
                edges = ", ".join(f"{u}-{d['relation']}->{v}"
                                  for u, v, d in output.edges(data=True))
                context_parts.append(f"Subgraph Structure: {edges}")
        p_text = " | ".join(context_parts)
        prompt = f"[System: Answer using context]\n[Context: {p_text}]\n[Query: {query_text}]"
        return {
            "query_type": q_type,
            "weights": dict(zip(names, map(float, weights))),
            "called_experts": called_experts,
            "constructed_prompt": prompt,
        }


LAB_CASES = [
    ("1.1 Simple", "Aspirin รักษาอะไร?", [0.9, 0.1, 0.1, 0.2]),
    ("1.2 Complex", "กลไกการลดอักเสบของ Aspirin ผ่าน COX-2 เป็นอย่างไร?", [0.1, 0.8, 0.9, 0.8]),
    ("1.3 Custom", "Aspirin เกี่ยวข้องกับเอนทิตีใด?", [1.0, 0.0, 0.0, 0.0]),
]


def main():
    pipeline = MoERAGPipeline()
    print("Task 1: Gating weights")
    for label, query, vector in LAB_CASES:
        result = pipeline.process_query(query, np.array(vector))
        print(f"{label}: h_q={vector}, weights={result['weights']}, {result['query_type']}")
    print("\nTasks 2–3: Dense baseline vs sparse gating (alpha > 0.3)")
    for label, query, vector in LAB_CASES:
        for sparse in (False, True):
            result = pipeline.process_query(query, np.array(vector), sparse=sparse)
            print(f"\n{label} / {'Sparse' if sparse else 'Dense'}")
            print(f"Called experts ({len(result['called_experts'])}/3): {result['called_experts']}")
            print(result['constructed_prompt'])


if __name__ == "__main__":
    main()
