# 📄 AskMyDoc — RAG-based Document Q&A

A lightweight **Retrieval-Augmented Generation (RAG)** pipeline that lets you ask natural language questions about any document (PDF) and get context-aware answers using semantic search — with full **OCR support** for scanned documents and **fully local inference** via Ollama.

---

## 🏆 Evaluation Results (DocVQA Benchmark)

Evaluated on 50 samples from the [DocVQA](https://huggingface.co/datasets/HuggingFaceM4/DocumentVQA) benchmark using industry-standard ANLS metric:

| Metric | Score |
|---|---|
| **DocVQA ANLS** | **0.6074 (60.7%)** |
| **Retrieval Hit Rate** | **66.0% (33/50)** |
| **OCR Success Rate** | **94.0% (47/50)** |
| **Avg Search Latency** | **0.14 ms** |

> ANLS (Average Normalized Levenshtein Similarity) is the official DocVQA metric. Random baseline ≈ 0.05, state of the art ≈ 0.92.

---

## 🚀 Demo

```
✅ System Ready!

Ask a question (or type 'exit' to quit): What is the main contribution of the paper?

📌 Answer:
The paper introduces a transformer-based encoder-decoder framework with three components —
an Affective Tracker, Behaviour-aware Generators, and a Polite Generator — to generate
behaviour-aware polite responses in customer-care conversational systems.
```

---

## 🧠 How It Works

![How It Works](https://github.com/user-attachments/assets/0f57dc0f-9ec0-470d-93fd-c90fe0f7cc9a)

A full RAG pipeline in two phases:

**Indexing (runs once at startup)**
1. `load_pdf()` extracts text — tries PyMuPDF → pypdf → OCR fallback automatically
2. `clean_text()` removes junk lines, hyphenated breaks, page numbers
3. `chunk_text()` splits into 200-word overlapping windows
4. `get_embeddings()` converts chunks to vectors using `all-MiniLM-L6-v2`
5. `FAISSIndex` stores all vectors for fast nearest-neighbour search

**Query (runs on every question)**
1. Your question is embedded with the same model
2. FAISS finds the 3 most semantically similar chunks in ~0.14ms
3. Retrieved chunks are passed as context to Ollama (llama3.2)
4. A grounded answer is generated locally — no API calls, no data leaving your machine

---

## 🗂️ Project Structure

```
AskMyDoc/
│
├── app.py                  # Main entry point — query loop
├── download_models.py      # One-time script to cache models locally
├── extract_text.py         # Converts two-column PDFs to clean text
│
├── data/
│   └── paper.pdf           # Your document goes here
│
├── model/
│   └── embedding.py        # Sentence embedding using all-MiniLM-L6-v2
│
├── search/
│   ├── chunking.py         # Splits text into overlapping chunks
│   ├── faiss_index.py      # FAISS vector index builder
│   └── retriever.py        # Semantic retriever — finds top-k chunks
│
├── rag/
│   └── generator.py        # Answer generation via Ollama (llama3.2)
│
└── eval/
    ├── docvqa_eval.py      # DocVQA benchmark evaluation
    ├── anls.py             # ANLS scorer
    └── run_eval.py         # Custom domain evaluation
```

---

## ⚙️ Models Used

| Model | Purpose | Size |
|---|---|---|
| `all-MiniLM-L6-v2` | Semantic embedding & retrieval | ~80 MB |
| `llama3.2` (via Ollama) | Answer generation | ~2 GB |
| `Tesseract OCR` | Text extraction from scanned PDFs | ~50 MB |

All models run **fully offline on CPU** — no GPU required, no API keys needed.

---

## 📄 PDF Extraction Pipeline

AskMyDoc automatically selects the best extraction strategy for your document:

```
PDF Input
    │
    ├── Pre-cleaned .txt exists? ──────────────────→ Use it directly
    │
    ├── is_scanned_pdf() → avg < 100 chars/page? ──→ Jump straight to OCR
    │
    ├── PyMuPDF extraction ─────────────────────────→ Best for two-column layouts
    │       │ failed or empty?
    ├── pypdf fallback ─────────────────────────────→ Standard extraction
    │       │ failed or empty?
    └── Tesseract OCR ──────────────────────────────→ For scanned/image PDFs
            │ 300 DPI render → pytesseract → clean text
```

No manual configuration needed — the right strategy is chosen automatically.

---

## 🛠️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/AskMyDoc.git
cd AskMyDoc
```

### 2. Create a virtual environment

```bash
python -m venv semantic
semantic\Scripts\activate      # Windows
source semantic/bin/activate   # Linux/Mac
```

### 3. Install Python dependencies

```bash
pip install sentence-transformers faiss-cpu pypdf pymupdf pytesseract pillow requests datasets
```

### 4. Install Tesseract binary (for OCR)

**Windows:** Download installer from https://github.com/UB-Mannheim/tesseract/wiki

**Linux:**
```bash
sudo apt install tesseract-ocr
```

**macOS:**
```bash
brew install tesseract
```

### 5. Install Ollama (local LLM)

Download from https://ollama.com/download then pull the model:

```bash
ollama pull llama3.2
```

### 6. Add your PDF

```
data/paper.pdf
```

---

## ▶️ Usage

**Terminal 1 — Start Ollama:**
```bash
ollama serve
```

**Terminal 2 — Run the app:**
```bash
python app.py
```

Then ask your questions:
```
Ask a question (or type 'exit' to quit): What dataset is used in the paper?
📌 Answer: The CYCCD dataset is used, consisting of 140k training, 20k validation, and 40k test conversations.
```

**Force OCR** (if your PDF is scanned):
```python
# In app.py, change:
document = load_pdf("data/paper.pdf", force_ocr=True)
```

---

## 📊 Running the Evaluation

```bash
# Full DocVQA benchmark (50 samples, streams — no full download needed)
python eval/docvqa_eval.py

# Custom evaluation on your own document
python app.py --eval
```

---

## 🔧 Configuration

| Parameter | File | Default | Description |
|---|---|---|---|
| `chunk_size` | `search/chunking.py` | `200` | Words per chunk |
| `overlap` | `search/chunking.py` | `50` | Overlap between chunks |
| `top_k` | `search/retriever.py` | `3` | Chunks retrieved per query |
| `OLLAMA_MODEL` | `rag/generator.py` | `llama3.2` | Generator model |
| `ocr_dpi` | `app.py` | `300` | DPI for OCR rendering |
| `ocr_lang` | `app.py` | `eng` | Tesseract language code |

---

## ⚠️ Known Limitations

- **Avg latency ~34s per query** — Ollama runs on CPU; a GPU reduces this to ~2s
- **OCR accuracy drops** on handwritten or very low quality scans
- **FAISS is in-memory** — very large documents (1000+ pages) may need a persistent vector store like ChromaDB
- **Single document** — one PDF per session; multi-doc support is a planned improvement

---

## 🔮 Future Improvements

- [ ] Support multiple PDFs simultaneously
- [ ] Add a web UI (Streamlit or Gradio)
- [ ] Persistent vector store (ChromaDB) for large document collections
- [ ] GPU acceleration for faster inference
- [ ] Add source chunk citation in answers
- [ ] Support `.txt`, `.docx`, and `.md` input formats
- [ ] Add conversation memory for follow-up questions
- [ ] Re-ranking retrieved chunks before generation

---

## 📦 Full Dependencies

```txt
sentence-transformers
faiss-cpu
pypdf
pymupdf
pytesseract
pillow
requests
datasets
```

---

## 📄 License

MIT License — feel free to use, modify, and distribute.

---

## 🙏 Acknowledgements

- [Sentence Transformers](https://www.sbert.net/) — embedding model
- [FAISS](https://github.com/facebookresearch/faiss) — vector similarity search
- [PyMuPDF](https://pymupdf.readthedocs.io/) — PDF text extraction
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) — scanned document support
- [Ollama](https://ollama.com/) — local LLM inference
- [DocVQA](https://rrc.cvc.uab.es/?ch=17) — evaluation benchmark
