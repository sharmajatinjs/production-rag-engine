import sys
import os
import streamlit as st
import time

# Add src directory to python path directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from generation.prompts import answer_query_with_context

st.set_page_config(page_title="Hybrid RAG Intelligence", layout="wide")
st.title("Production Hybrid RAG Engine")
st.caption("Powered by Qdrant (Dense) + BM25 (Sparse) + Cross-Encoder Re-ranker + Groq Qwen-3")

with st.sidebar:
    st.header("Retrieval Settings")
    mode = st.selectbox(
        "Select Retrieval Pipeline",
        options=["rerank", "hybrid", "dense"],
        index=0,
        help="rerank = Hybrid + Cross-Encoder | hybrid = BM25 + Dense RRF | dense = Vector Only"
    )
    top_k = st.slider("Top-K Chunks", min_value=1, max_value=8, value=4)
    
    st.markdown("---")
    st.subheader("Benchmark Metrics (16-Q Eval)")
    st.markdown("- **Hit Rate@4:** 100.0% (up from 46.2%)")
    st.markdown("- **Answer Correctness:** 84.6% (up from 23.1%)")
    st.markdown("- **Refusal Accuracy:** 100.0%")

question = st.text_input(
    "Ask a question from your DDIM Research Paper or ML/Transformer Notes:",
    placeholder="e.g., Positional encoding formula mein 10,000 ka constant kyun hota hai?"
)

if st.button("Run Query", type="primary") and question.strip():
    with st.spinner(f"Running {mode.upper()} pipeline..."):
        try:
            start_time = time.time()
            result = answer_query_with_context(question, mode=mode, top_k=top_k)
            latency = round((time.time() - start_time) * 1000, 2)
            
            st.subheader("Answer")
            st.write(result["answer"])
            st.info(f"Pipeline Mode: **{mode.upper()}** | Latency: **{latency} ms**")
            
            st.subheader("Retrieved Context Chunks")
            for idx, src in enumerate(result["sources"]):
                score_label = f"Rerank Score: {src.get('rerank_score', src['score']):.4f}"
                with st.expander(f"Chunk {idx+1}: {src['source']} (Page/Section {src['page']}) — {score_label}"):
                    st.write(src["text"])
        except Exception as e:
            st.error(f"Error executing query: {e}")