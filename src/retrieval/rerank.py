import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sentence_transformers import CrossEncoder
from retrieval.hybrid import search_hybrid

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_reranker = None

def get_reranker():
    global _reranker
    if _reranker is None:
        print(f"Loading Cross-Encoder Re-ranker ({RERANK_MODEL})...")
        _reranker = CrossEncoder(RERANK_MODEL)
    return _reranker

def search_reranked(query: str, initial_k: int = 15, final_k: int = 4) -> list[dict]:
    candidates = search_hybrid(query, top_k=initial_k)
    if not candidates:
        return []
        
    reranker = get_reranker()
    pairs = [[query, c["text"]] for c in candidates]
    scores = reranker.predict(pairs)
    
    for idx, c in enumerate(candidates):
        c["rerank_score"] = float(scores[idx])
        
    ranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)[:final_k]
    return ranked