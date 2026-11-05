import re

def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> list[str]:
    """Text ko fixed size aur overlap ke saath chhote chunks mein todta hai."""
    words = text.split()
    if not words:
        return []
    
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

def chunk_documents(parsed_docs: list[dict], chunk_size: int = 400, overlap: int = 50) -> list[dict]:
    """Saare parsed pages ko chunks mein convert karta hai aur metadata preserve rakhta hai."""
    chunked_docs = []
    
    for doc in parsed_docs:
        text = doc["text"]
        metadata = doc["metadata"]
        
        text_chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        
        for idx, chunk in enumerate(text_chunks):
            chunked_docs.append({
                "chunk_id": f"{metadata['source']}_p{metadata['page']}_c{idx+1}",
                "text": chunk,
                "metadata": {
                    "source": metadata["source"],
                    "page": metadata["page"],
                    "chunk_index": idx + 1
                }
            })
            
    return chunked_docs

if __name__ == "__main__":
    from parsers import parse_all_pdfs
    raw_docs = parse_all_pdfs()
    final_chunks = chunk_documents(raw_docs)
    print(f"\nTotal generated chunks from all PDFs: {len(final_chunks)}")
    if final_chunks:
        print("Sample Chunk ID:", final_chunks[0]["chunk_id"])
        print("Sample Chunk Text:", final_chunks[0]["text"][:200] + "...")