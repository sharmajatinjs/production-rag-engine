import os
import sys
import json
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from retrieval.dense import search_dense, get_qdrant_client
from retrieval.hybrid import search_hybrid
from retrieval.rerank import search_reranked
from generation.prompts import answer_query, clean_llm_output
from generation.llm_client import generate_answer

JUDGE_PROMPT = """You are grading an AI's answer against an Expected Answer.
Question: {question}
Expected Answer: {expected}
AI Answer: {actual}

Does the AI Answer capture the main points or key facts of the Expected Answer?
Reply with ONLY the digit 1 if yes, or 0 if no."""

def run_stage(dataset: list[dict], mode: str, top_k: int = 4) -> tuple[float, float, float]:
    retrieval_hits, retrieval_total = 0, 0
    refusal_correct, refusal_total = 0, 0
    answer_correct, answer_total = 0, 0

    print(f"\n--- Evaluating Mode: {mode.upper()} (top_k={top_k}) ---")
    for item in dataset:
        qid, qtype, question = item["id"], item["type"], item["question"]
        expected_ans, expected_src, expected_page = item["expected_answer"], item["expected_source"], item["expected_page"]

        if mode == "dense":
            hits = search_dense(question, top_k=top_k)
        elif mode == "hybrid":
            hits = search_hybrid(question, top_k=top_k)
        else:
            hits = search_reranked(question, initial_k=15, final_k=top_k)

        if qtype != "unanswerable":
            retrieval_total += 1
            found = any(
                (h["source"] == expected_src and abs(h["page"] - expected_page) <= 1)
                or (expected_src.endswith(".excalidraw") and h["source"] == expected_src)
                for h in hits
            )
            if found:
                retrieval_hits += 1

        actual_ans = answer_query(question, mode=mode, top_k=top_k)
        time.sleep(1.2)

        if qtype == "unanswerable":
            refusal_total += 1
            if "i don't know" in actual_ans.lower() or "not found" in actual_ans.lower():
                refusal_correct += 1
        else:
            answer_total += 1
            verdict = clean_llm_output(generate_answer(
                JUDGE_PROMPT.format(question=question, expected=expected_ans, actual=actual_ans)
            ))
            if "1" in verdict:
                answer_correct += 1

    hit_rate = (retrieval_hits / retrieval_total) * 100
    ans_acc = (answer_correct / answer_total) * 100
    ref_acc = (refusal_correct / refusal_total) * 100
    print(f"Result [{mode.upper()}] -> Hit Rate: {hit_rate:.1f}% | Correctness: {ans_acc:.1f}% | Refusal: {ref_acc:.1f}%")
    return hit_rate, ans_acc, ref_acc

if __name__ == "__main__":
    with open("src/eval/dataset.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    # Baseline numbers already logged in Week 2 (46.2% Hit Rate, 23.1% Correctness, 100.0% Refusal)
    h_hit, h_ans, h_ref = run_stage(data, mode="hybrid", top_k=4)
    r_hit, r_ans, r_ref = run_stage(data, mode="rerank", top_k=4)

    table = f"""# RAG Experiment Results (Before vs. After)

| Pipeline Stage | Retrieval Hit Rate | Answer Correctness | Refusal Accuracy |
|---|---|---|---|
| 1. Baseline (Dense Only, top-3) | 46.2% | 23.1% | 100.0% |
| 2. Hybrid Search (BM25 + Dense RRF, top-4) | {h_hit:.1f}% | {h_ans:.1f}% | {h_ref:.1f}% |
| 3. Hybrid + Cross-Encoder Re-ranker (top-15 -> top-4) | {r_hit:.1f}% | {r_ans:.1f}% | {r_ref:.1f}% |
"""
    print("\n" + table)
    with open("experiments/results.md", "w", encoding="utf-8") as f:
        f.write(table)
    
    client = get_qdrant_client()
    client.close()