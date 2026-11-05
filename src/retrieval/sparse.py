import sys
import os
import re
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rank_bm25 import BM25Okapi
from ingest.parsers import parse_all_pdfs
from ingest.chunkers import chunk_documents

_bm25 = None
_chunks = None

def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())

def init_bm25():
    global _bm25, _chunks
    if _bm25 is None:
        raw_docs = parse_all_pdfs()
        _chunks = chunk_documents(raw_docs)
        tokenized_corpus = [tokenize(c["text"]) for c in _chunks]
        _bm25 = BM25Okapi(tokenized_corpus)
    return _bm25, _chunks

def search_sparse(query: str, top_k: int = 10) -> list[dict]:
    bm25, chunks = init_bm25()
    tokenized_query = tokenize(query)
    scores = bm25.get_scores(tokenized_query)
    
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    
    results = []
    for idx in ranked_indices:
        c = chunks[idx]
        results.append({
            "score": float(scores[idx]),
            "text": c["text"],
            "source": c["metadata"]["source"],
            "page": c["metadata"]["page"],
            "chunk_id": c["chunk_id"]
        })
    return results

if __name__ == "__main__":
    hits = search_sparse("Positional encoding 10,000 constant KV Cache", top_k=2)
    for h in hits:
        print(f"BM25 Match: {h['chunk_id']} (Score: {h['score']:.2f})")