from search.chunking import chunk_text
from model.embedding import get_embeddings
from search.faiss_index import FAISSIndex
from search.retriever import Retriever
from rag.generator import generate_answer

def load_pdf(path):
    # pypdf and pdfplumber both struggle with two-column academic PDFs.
    # Using pre-cleaned text extracted manually.
    import pathlib
    clean_path = pathlib.Path(path).with_suffix('.txt')
    if clean_path.exists():
        return clean_path.read_text(encoding='utf-8')
    
    # Fallback to pypdf
    from pypdf import PdfReader
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text


# ---- Setup ----
print("Loading document...")
document = load_pdf("data/paper.pdf")

print("Chunking...")
chunks = chunk_text(document)

print("Embedding...")
embeddings = get_embeddings(chunks)

print("Building FAISS index...")
index = FAISSIndex(len(embeddings[0]))
index.add(embeddings)

retriever = Retriever(index, chunks)

print("✅ System Ready!\n")

# ---- Query Loop ----
while True:
    query = input("Ask a question (or type 'exit' to quit): ")

    if query.lower() == "exit":
        break

    context_chunks = retriever.retrieve(query)

    # ---- Debug: Print retrieved chunks ----
    print("\n📚 Retrieved Chunks:")
    for i, chunk in enumerate(context_chunks):
        print(f"\n--- Chunk {i+1} ---")
        print(chunk)
    print("-" * 50)

    answer = generate_answer(query, context_chunks)

    print("\n📌 Answer:")
    print(answer)
    print("-" * 50)