import fitz  # PyMuPDF
import chromadb
import ollama
import os

# --- Config ---
PDF_PATHS = ["/Users/vittaldc/economy_bot/Indian_Economy_Ramesh_Singh_7e_0.pdf", "/Users/vittaldc/economy_bot/echapter.pdf"]
CHROMA_PATH = "./chroma_db"
CHUNK_SIZE = 800      # characters per chunk
CHUNK_OVERLAP = 150

def extract_text(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    return full_text

def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
    return chunks

def embed(text):
    response = ollama.embed(model="nomic-embed-text", input=text)
    return response["embeddings"][0]

# --- Main ---
client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection("economy")

doc_id = 0
for pdf in PDF_PATHS:
    print(f"Processing {pdf}...")
    text = extract_text(pdf)
    chunks = chunk_text(text)
    for i, chunk in enumerate(chunks):
        embedding = embed(chunk)
        collection.add(
            ids=[f"doc{doc_id}"],
            embeddings=[embedding],
            documents=[chunk],
            metadatas=[{"source": pdf, "chunk": i}]
        )
        doc_id += 1
        if i % 50 == 0:
            print(f"  {i}/{len(chunks)} chunks done")

print(f"Done! {doc_id} chunks stored.")