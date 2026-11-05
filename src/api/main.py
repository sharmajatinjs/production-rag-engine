import sys
import os
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from generation.prompts import answer_query_with_context

app = FastAPI(title="Production RAG API", version="1.0.0")

class QueryRequest(BaseModel):
    question: str
    mode: str = "rerank"
    top_k: int = 4

@app.get("/health")
def health_check():
    return {"status": "ok", "message": "RAG API is running"}

@app.post("/query")
def query_rag(req: QueryRequest):
    try:
        start_time = time.time()
        result = answer_query_with_context(
            question=req.question,
            mode=req.mode,
            top_k=req.top_k
        )
        latency = round((time.time() - start_time) * 1000, 2)
        return {
            "question": req.question,
            "mode": req.mode,
            "latency_ms": latency,
            "answer": result["answer"],
            "sources": result["sources"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))