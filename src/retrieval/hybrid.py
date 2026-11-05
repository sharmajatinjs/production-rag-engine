import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from retrieval.dense import search_dense
from retrieval.sparse import search_sparse

def search_hybrid(query: str, top_k: int = 10, rrf_k: int = 60) -> list[dict]:
    dense_hits = search_dense(query, top_k=top_k)
    sparse_hits = search_sparse(query, top_k=top_k)
    
    rrf_scores = {}
    chunk_map = {}
    
    for rank, hit in enumerate(dense_hits):
        cid = hit["chunk_id"]
        chunk_map[cid] = hit
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (rrf_k + rank + 1)
        
    for rank, hit in enumerate(sparse_hits):
        cid = hit["chunk_id"]
        chunk_map[cid] = hit
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (rrf_k + rank + 1)
        
    sorted_cids = sorted(rrf_scores.keys(), key=lambda c: rrf_scores[c], reverse=True)[:top_k]
    
    results = []
    for cid in sorted_cids:
        item = chunk_map[cid].copy()
        item["score"] = rrf_scores[cid]
        results.append(item)
    return results