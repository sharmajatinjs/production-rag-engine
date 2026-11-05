import sys
import os
import re
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from retrieval.dense import search_dense
from retrieval.hybrid import search_hybrid
from retrieval.rerank import search_reranked
from generation.llm_client import generate_answer

STRICT_RAG_PROMPT = """You are a precise technical AI assistant. Answer the user's question using ONLY the provided context chunks below.
If the answer is not in the context, reply with ONLY: "I don't know based on the provided documents."
Include all exact numbers, equations, and technical terms from the context. Cite sources as.

Context Chunks:
{context}

User Question: {question}

Answer:"""

def clean_llm_output(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

def answer_query_with_context(question: str, mode: str = "rerank", top_k: int = 4) -> dict:
    if mode == "dense":
        retrieved_hits = search_dense(question, top_k=top_k)
    elif mode == "hybrid":
        retrieved_hits = search_hybrid(question, top_k=top_k)
    else:
        retrieved_hits = search_reranked(question, initial_k=15, final_k=top_k)
        
    if not retrieved_hits:
        return {
            "answer": "I don't know based on the provided documents.",
            "sources": []
        }
        
    context_str = ""
    for hit in retrieved_hits:
        context_str += f"---\nSource: {hit['source']} | Page: {hit['page']}\nText: {hit['text']}\n"
        
    prompt = STRICT_RAG_PROMPT.format(context=context_str, question=question)
    raw_answer = generate_answer(prompt)
    return {
        "answer": clean_llm_output(raw_answer),
        "sources": retrieved_hits
    }

def answer_query(question: str, mode: str = "rerank", top_k: int = 4) -> str:
    result = answer_query_with_context(question, mode=mode, top_k=top_k)
    return result["answer"]

if __name__ == "__main__":
    test_question = "What are Denoising Diffusion Implicit Models?"
    print("\n--- Running End-to-End RAG Test ---")
    res = answer_query_with_context(test_question, mode="rerank", top_k=3)
    print(f"\nFinal AI Response:\n{res['answer']}")
    print(f"\nRetrieved {len(res['sources'])} source chunks.")