import fitz
import re
import pathlib
from pypdf import PdfReader
from search.chunking import chunk_text
from model.embedding import get_embeddings
from search.faiss_index import FAISSIndex
from search.retriever import Retriever
from rag.generator import generate_answer


# ──────────────────────────────────────────
# Text Cleaning
# ──────────────────────────────────────────

def clean_text(text):
    text = re.sub(r'-\n', '', text)
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'^\d+$', '', text, flags=re.MULTILINE)
    lines = text.split('\n')
    lines = [l for l in lines if len(l.strip()) > 20]
    return '\n'.join(lines).strip()


# ──────────────────────────────────────────
# Scanned PDF Detection
# ──────────────────────────────────────────

def is_scanned_pdf(path, sample_pages=3):
    try:
        doc = fitz.open(path)
        total_chars = 0
        pages_to_check = min(sample_pages, len(doc))
        for i in range(pages_to_check):
            total_chars += len(doc[i].get_text("text").strip())
        return (total_chars / pages_to_check) < 100
    except Exception:
        return False


# ──────────────────────────────────────────
# OCR
# ──────────────────────────────────────────

def ocr_pdf(path, dpi=300, lang="eng"):
    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        raise ImportError(
            "OCR requires: pip install pytesseract pillow\n"
            "And Tesseract binary: https://github.com/tesseract-ocr/tesseract"
        )

    doc = fitz.open(path)
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pages_text = []

    print(f"🔍 OCR started — {len(doc)} pages at {dpi} DPI...")
    for page_num, page in enumerate(doc):
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        page_text = pytesseract.image_to_string(img, lang=lang, config="--psm 3 --oem 3")
        pages_text.append(page_text)
        print(f"  ✅ Page {page_num + 1}/{len(doc)}")

    return "\n\n".join(pages_text)


# ──────────────────────────────────────────
# PDF Loader (with OCR fallback)
# ──────────────────────────────────────────

def load_pdf(path, ocr_dpi=300, ocr_lang="eng", force_ocr=False):
    # Step 1: Pre-cleaned .txt shortcut
    clean_path = pathlib.Path(path).with_suffix('.txt')
    if clean_path.exists():
        print("✅ Using pre-cleaned text file...")
        return clean_path.read_text(encoding='utf-8')

    # Step 2: Force OCR if requested
    if force_ocr:
        print("🔍 force_ocr=True — skipping text extraction...")
        return clean_text(ocr_pdf(path, dpi=ocr_dpi, lang=ocr_lang))

    # Step 3: Detect scanned PDF early
    if is_scanned_pdf(path):
        print("🖼️  Scanned PDF detected — jumping to OCR...")
        return clean_text(ocr_pdf(path, dpi=ocr_dpi, lang=ocr_lang))

    # Step 4: PyMuPDF extraction
    extracted_text = ""
    try:
        print("📄 Extracting with PyMuPDF...")
        doc = fitz.open(path)
        text = ""
        for page in doc:
            blocks = page.get_text("blocks")
            blocks = sorted(blocks, key=lambda b: (round(b[1] / 10), b[0]))
            for block in blocks:
                text += block[4] + "\n"
        extracted_text = clean_text(text)
    except Exception as e:
        print(f"⚠️  PyMuPDF failed: {e}")

    # Step 5: pypdf fallback
    if len(extracted_text.strip()) < 100:
        print("📄 Poor output — trying pypdf...")
        try:
            reader = PdfReader(path)
            text = "".join(page.extract_text() or "" for page in reader.pages)
            extracted_text = clean_text(text)
        except Exception as e:
            print(f"⚠️  pypdf failed: {e}")

    # Step 6: OCR as last resort
    if len(extracted_text.strip()) < 100:
        print("📄 Text extraction failed — falling back to OCR...")
        extracted_text = clean_text(ocr_pdf(path, dpi=ocr_dpi, lang=ocr_lang))

    if not extracted_text.strip():
        raise ValueError(
            f"Could not extract any text from '{path}'.\n"
            "File may be corrupted, password-protected, or unsupported."
        )

    return extracted_text


# ──────────────────────────────────────────
# Setup
# ──────────────────────────────────────────

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


# ──────────────────────────────────────────
# Query Loop
# ──────────────────────────────────────────

while True:
    query = input("Ask a question (or type 'exit' to quit): ")

    if query.lower() == "exit":
        break

    context_chunks = retriever.retrieve(query)

    print("\n📚 Retrieved Chunks:")
    for i, chunk in enumerate(context_chunks):
        print(f"\n--- Chunk {i+1} ---")
        print(chunk)
    print("-" * 50)

    answer = generate_answer(query, context_chunks)

    print("\n📌 Answer:")
    print(answer)
    print("-" * 50)