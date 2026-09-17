"""Step 2 lab using Step 1 routing and illustrative, manually embedded data.

This is a deterministic simulation, not a live LLM or medical fact checker.
"""

import json
from pathlib import Path

import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from code import MoERAGPipeline


QUERY = np.array([0.1, 0.8, 0.9, 0.8])
CONTEXTS = ["Aspirin inhibits COX-2", "COX-2 causes Inflammation",
            "Paracetamol is over-the-counter medicine"]
EMBEDDINGS = [np.array(v) for v in ([0.1, 0.7, 0.8, 0.7],
                                  [0.2, 0.8, 0.9, 0.8], [0.9, 0.1, 0.1, 0.1])]
FAITHFUL = np.array([0.15, 0.75, 0.85, 0.80])
HALLUCINATED = np.array([0.9, 0.1, 0.1, 0.1])
REFUSAL = "Cannot verify the answer from the available context."


def vector(value):
    value = np.asarray(value, dtype=float)
    if value.shape != (4,) or not np.isfinite(value).all() or not np.any(value):
        raise ValueError("Expected a finite, nonzero four-dimensional vector")
    return value


def similarity(left, right):
    return float(cosine_similarity(vector(left)[None, :], vector(right)[None, :])[0, 0])


class ContextRefiner:
    def __init__(self, threshold=0.35):
        if not np.isfinite(threshold) or not -1 <= threshold <= 1:
            raise ValueError("threshold must be between -1 and 1")
        self.threshold = threshold

    def select(self, h_q, context_items, item_embeddings):
        vector(h_q)
        if len(context_items) != len(item_embeddings):
            raise ValueError("Each context item must have one embedding")
        scores = [similarity(h_q, emb) for emb in item_embeddings]
        indices = [i for i, score in enumerate(scores) if score >= self.threshold]
        return indices, scores

    def refine(self, h_q, context_items, item_embeddings):
        indices, _ = self.select(h_q, context_items, item_embeddings)
        return [context_items[i] for i in indices]


class FactVerifier:
    def __init__(self):
        self.W_v = np.array([0.4, 0.4, 0.2])

    def components(self, h_q, h_c, h_a):
        return dict(sim_qc=similarity(h_q, h_c), sim_ca=similarity(h_c, h_a),
                    sim_qa=similarity(h_q, h_a))

    def verify(self, h_q, h_c, h_a):
        # Match the executable worksheet formula (weighted cosine, no sigmoid).
        return float(self.W_v @ list(self.components(h_q, h_c, h_a).values()))


class AdvancedMoERAGPipeline(MoERAGPipeline):
    def __init__(self, threshold=0.35, generator=None):
        super().__init__()
        self.refiner = ContextRefiner(threshold)
        self.verifier = FactVerifier()
        self.generator = generator or self.generate_llm_response

    def generate_llm_response(self, prompt):
        # Extractive mock: do not add claims absent from the supplied context.
        return prompt.split("[Context: ", 1)[1].split("]", 1)[0], FAITHFUL.copy()

    def context_complete(self, items, required_path):
        if required_path is None:
            return bool(items)
        graph = nx.DiGraph()
        for source, target, data in self.experts.graph.edges(data=True):
            if f"{source} {data['relation']} {target}" in items:
                graph.add_edge(source, target)
        start, end = required_path
        return start in graph and end in graph and nx.has_path(graph, start, end)

    def run_step_2(self, h_q, raw_contexts, raw_embeddings,
                   query_text="Explain the path in the illustrative graph.",
                   required_path=("Aspirin", "Inflammation")):
        indices, scores = self.refiner.select(h_q, raw_contexts, raw_embeddings)
        contexts = [raw_contexts[i] for i in indices]
        result = dict(status="Refused", refined_context=contexts, item_similarities=scores,
                      context_embedding=None, verification_score=None, similarities=None,
                      final_response=REFUSAL, reason="No relevant context", constructed_prompt=None)
        if not contexts:
            return result
        if not self.context_complete(contexts, required_path):
            result.update(status="Insufficient Context", reason="Required directed path is missing")
            return result
        h_c = np.mean([raw_embeddings[i] for i in indices], axis=0)
        prompt = f"[System: Use only the supplied context]\n[Context: {' | '.join(contexts)}]\n[Query: {query_text}]"
        answer, h_a = self.generator(prompt)
        score = self.verifier.verify(h_q, h_c, h_a)
        passed = score >= 0.60
        result.update(status="Success" if passed else "Verification Failed",
                      reason=None if passed else "Verification score below 0.60",
                      context_embedding=h_c.tolist(), verification_score=score,
                      similarities=self.verifier.components(h_q, h_c, h_a),
                      final_response=answer if passed else REFUSAL, constructed_prompt=prompt)
        return result


class IterativeMoERAGPipeline(AdvancedMoERAGPipeline):
    def retrieve(self, h_q, weights):
        embeddings = dict(zip(CONTEXTS, EMBEDDINGS))
        embeddings.update({"Aspirin": np.array([0.9, 0.1, 0.1, 0.2]),
                           "COX-2": np.array([0.1, 0.7, 0.8, 0.7]),
                           "Aspirin treats Pain": np.array([0.9, 0.1, 0.1, 0.2])})
        contexts, called = [], []
        retrievers = [self.experts.entity_expert, self.experts.relation_expert,
                      self.experts.subgraph_expert]
        for name, weight, retrieve in zip(("Entity", "Relation", "Subgraph"), weights, retrievers):
            if weight <= 0.3:
                continue
            output = retrieve(h_q)
            called.append(name)
            if name == "Entity":
                contexts.extend(output)
            elif name == "Relation":
                contexts.extend(f"{s} {p} {o}" for s, p, o in output)
            else:
                contexts.extend(f"{s} {d['relation']} {o}" for s, o, d in output.edges(data=True))
        contexts = list(dict.fromkeys(contexts))
        return contexts, [embeddings[item] for item in contexts], called

    def run_step_2_iterative(self, h_q, raw_contexts=None, raw_embeddings=None,
                             max_retries=2, **kwargs):
        if isinstance(max_retries, bool) or not isinstance(max_retries, int) or max_retries < 0:
            raise ValueError("max_retries must be a nonnegative integer")
        if (raw_contexts is None) != (raw_embeddings is None):
            raise ValueError("Provide both raw_contexts and raw_embeddings")
        weights, _ = self.router.route(vector(h_q))
        history = []
        for attempt in range(max_retries + 1):
            if attempt == 0 and raw_contexts is not None:
                contexts, embeddings, called = raw_contexts, raw_embeddings, []
                source = "supplied initial context"
            else:
                contexts, embeddings, called = self.retrieve(h_q, weights)
                source = "Step 1 experts"
            result = self.run_step_2(h_q, contexts, embeddings, **kwargs)
            history.append(dict(attempt=attempt, weights=weights.tolist(), source=source,
                                called_experts=called, raw_contexts=list(contexts), **result))
            if result["status"] == "Success":
                break
            if attempt < max_retries:
                # Add 0.20 before normalization so weights still sum to one.
                weights = weights.copy()
                weights[2] += 0.20
                weights /= weights.sum()
        return dict(result, attempts=len(history), retries=len(history) - 1,
                    exhausted=result["status"] != "Success", history=history)


def experiments():
    thresholds = {str(t): AdvancedMoERAGPipeline(t).run_step_2(QUERY, CONTEXTS, EMBEDDINGS)
                  for t in (0.35, 0.70)}
    hallucination = AdvancedMoERAGPipeline(generator=lambda prompt: ("Unsupported mock answer", HALLUCINATED))
    failed = hallucination.run_step_2(QUERY, CONTEXTS, EMBEDDINGS)
    recovery = IterativeMoERAGPipeline().run_step_2_iterative(QUERY, CONTEXTS[:1], EMBEDDINGS[:1])
    calls = 0

    def transient_failure(prompt):
        nonlocal calls
        calls += 1
        return AdvancedMoERAGPipeline().generate_llm_response(prompt) if calls > 1 else ("Unsupported mock answer", HALLUCINATED)

    retry = IterativeMoERAGPipeline(generator=transient_failure).run_step_2_iterative(QUERY)
    exhausted = IterativeMoERAGPipeline(generator=hallucination.generator).run_step_2_iterative(QUERY)
    return dict(thresholds=thresholds, hallucination=failed, incomplete_recovery=recovery,
                verification_recovery=retry, persistent_failure=exhausted)


if __name__ == "__main__":
    results = experiments()
    target = Path(__file__).with_name("step2_results.json")
    target.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for name, result in results.items():
        if name == "thresholds":
            for threshold, row in result.items():
                print(f"threshold={threshold}: {row['status']}, score={row['verification_score']:.6f}")
        else:
            print(f"{name}: {result['status']}, score={result['verification_score']}, attempts={result.get('attempts', 1)}")
    print(f"Saved: {target}")
