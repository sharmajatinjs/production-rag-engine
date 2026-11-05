import os
import json
import fitz  # PyMuPDF

def parse_pdf(file_path: str) -> list[dict]:
    doc = fitz.open(file_path)
    parsed_pages = []
    file_name = os.path.basename(file_path)
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        if text.strip():
            parsed_pages.append({
                "text": text.strip(),
                "metadata": {
                    "source": file_name,
                    "page": page_num + 1
                }
            })
    return parsed_pages

def parse_excalidraw(file_path: str) -> list[dict]:
    file_name = os.path.basename(file_path)
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading JSON/Excalidraw file {file_name}: {e}")
        return []
        
    elements = data.get("elements", [])
    text_blocks = []
    for el in elements:
        if el.get("type") == "text" and not el.get("isDeleted", False):
            txt = el.get("text", "").strip()
            if txt:
                text_blocks.append(txt)
                
    parsed_sections = []
    group_size = 5
    for i in range(0, len(text_blocks), group_size):
        combined_text = "\n\n".join(text_blocks[i:i + group_size])
        parsed_sections.append({
            "text": combined_text,
            "metadata": {
                "source": file_name,
                "page": (i // group_size) + 1
            }
        })
    print(f"Extracted {len(text_blocks)} text elements from Excalidraw notes.")
    return parsed_sections

def parse_all_pdfs(raw_dir: str = "data/raw") -> list[dict]:
    all_docs = []
    if not os.path.exists(raw_dir):
        print(f"Directory {raw_dir} nahi mili.")
        return all_docs
        
    for file in os.listdir(raw_dir):
        file_path = os.path.join(raw_dir, file)
        if file.lower().endswith(".pdf"):
            print(f"Parsing PDF: {file}...")
            all_docs.extend(parse_pdf(file_path))
        else:
            print(f"Parsing Notes/JSON: {file}...")
            all_docs.extend(parse_excalidraw(file_path))
            
    return all_docs

if __name__ == "__main__":
    docs = parse_all_pdfs()
    print(f"\nTotal extracted pages/sections from all files: {len(docs)}")