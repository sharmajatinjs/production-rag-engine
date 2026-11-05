import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "rag_collection"
STORAGE_DIR = "qdrant_storage"

_model = None
_client = None

def get_embedding_model():
    global _model
    if _model is None:
        print(f"Loading embedding model ({EMBEDDING_MODEL})...")
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model

def get_qdrant_client():
    global _client
    if _client is None:
        os.makedirs(STORAGE_DIR, exist_ok=True)
        _client = QdrantClient(path=STORAGE_DIR)
    return _client

def init_collection(client: QdrantClient, vector_size: int):
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )
    print(f"Created fresh Qdrant collection: {COLLECTION_NAME}")

def ingest_chunks_to_qdrant(chunks: list[dict]):
    client = get_qdrant_client()
    model = get_embedding_model()
    
    sample_vector = model.encode("test").tolist()
    init_collection(client, vector_size=len(sample_vector))
    
    points = []
    print(f"Generating embeddings for {len(chunks)} chunks and uploading to Qdrant...")
    for idx, chunk in enumerate(chunks):
        vector = model.encode(chunk["text"]).tolist()
        points.append(
            PointStruct(
                id=idx + 1,
                vector=vector,
                payload={
                    "chunk_id": chunk["chunk_id"],
                    "text": chunk["text"],
                    "source": chunk["metadata"]["source"],
                    "page": chunk["metadata"]["page"]
                }
            )
        )
    
    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"Successfully stored {len(points)} chunks into Qdrant database!")

def search_dense(query: str, top_k: int = 3) -> list[dict]:
    client = get_qdrant_client()
    model = get_embedding_model()
    
    query_vector = model.encode(query).tolist()
    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k
    )
    
    retrieved = []
    for res in response.points:
        retrieved.append({
            "score": res.score,
            "text": res.payload["text"],
            "source": res.payload["source"],
            "page": res.payload["page"],
            "chunk_id": res.payload["chunk_id"]
        })
    return retrieved

if __name__ == "__main__":
    from ingest.parsers import parse_all_pdfs
    from ingest.chunkers import chunk_documents
    
    raw_docs = parse_all_pdfs()
    chunks = chunk_documents(raw_docs)
    ingest_chunks_to_qdrant(chunks)
    
    query = "What is the difference between Explicit Knowledge and Tacit Knowledge?"
    print(f"\nTesting search for query: '{query}'")
    hits = search_dense(query, top_k=2)
    for i, hit in enumerate(hits):
        print(f"\nMatch {i+1} (Score: {hit['score']:.4f}, Source: {hit['source']}, Page: {hit['page']})")
        print(f"Text: {hit['text'][:150]}...")