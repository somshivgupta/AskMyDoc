from search.chunking import chunk_text
from model.embedding import get_embeddings
from search.faiss_index import FAISSIndex
from search.retriever import Retriever
from rag.generator import generate_answer

def load_pdf(path):
    import pathlib
    import re

    def clean_text(text):
        text = re.sub(r'-\n', '', text)           # Fix hyphenated line breaks
        text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)  # Fix mid-word line breaks
        text = re.sub(r' +', ' ', text)            # Remove multiple spaces
        text = re.sub(r'\n{3,}', '\n\n', text)     # Remove excessive newlines
        text = re.sub(r'^\d+$', '', text, flags=re.MULTILINE)  # Remove page numbers
        lines = text.split('\n')
        lines = [l for l in lines if len(l.strip()) > 20]  # Remove short junk lines
        return '\n'.join(lines).strip()

    # ---- Step 1: Use pre-cleaned .txt if it exists ----
    clean_path = pathlib.Path(path).with_suffix('.txt')
    if clean_path.exists():
        print("✅ Using pre-cleaned text file...")
        return clean_path.read_text(encoding='utf-8')

    # ---- Step 2: Try PyMuPDF (best for two-column PDFs) ----
    try:
        import fitz
        print("📄 Extracting with PyMuPDF...")
        doc = fitz.open(path)
        text = ""
        for page in doc:
            blocks = page.get_text("blocks")
            blocks = sorted(blocks, key=lambda b: (round(b[1] / 10), b[0]))
            for block in blocks:
                text += block[4] + "\n"
        text = clean_text(text)
        if len(text.strip()) > 100:
            return text
    except ImportError:
        print("⚠️ PyMuPDF not installed, trying next method...")
    except Exception as e:
        print(f"⚠️ PyMuPDF failed: {e}")

    # ---- Step 3: Fallback to pypdf ----
    print("📄 Extracting with pypdf (fallback)...")
    from pypdf import PdfReader
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return clean_text(text)


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