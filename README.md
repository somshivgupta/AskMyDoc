# 📄 AskMyDoc — RAG-based Research Paper Q&A

A lightweight **Retrieval-Augmented Generation (RAG)** pipeline that lets you ask natural language questions about any research paper (PDF) and get context-aware answers using semantic search.

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

```
PDF → Text Extraction → Chunking → Embeddings → FAISS Index
                                                      ↓
               Query → Embed Query → Semantic Search → Top-K Chunks
                                                      ↓
                              Chunks + Query → flan-t5-base → Answer
```

This is a full RAG pipeline:

- **Retrieval** — uses semantic vector search (not keyword matching) to find the most relevant chunks from the document
- **Augmented Generation** — passes retrieved chunks as context to a language model to generate a grounded answer

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
│   └── paper.pdf           # Your research paper goes here
│
├── model/
│   └── embedding.py        # Sentence embedding using all-MiniLM-L6-v2
│
├── search/
│   ├── chunking.py         # Splits text into overlapping chunks
│   ├── faiss_index.py      # FAISS vector index builder
│   └── retriever.py        # Semantic retriever — finds top-k chunks
│
└── rag/
    └── generator.py        # Answer generation using flan-t5-base
```

---

## ⚙️ Models Used

| Model | Purpose | Size |
|---|---|---|
| `all-MiniLM-L6-v2` | Semantic embedding & retrieval | ~80 MB |
| `google/flan-t5-base` | Answer generation | ~990 MB |

Both models run **fully offline** on CPU after the initial download.

---

## 🛠️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/PaperWhisperer.git
cd PaperWhisperer
```

### 2. Install dependencies

```bash
pip install sentence-transformers transformers faiss-cpu pypdf pymupdf torch
```

### 3. Download and cache models (one-time)

```bash
python download_models.py
```

### 4. Add your PDF

Place your research paper in the `data/` folder and name it `paper.pdf`:

```
data/paper.pdf
```

### 5. (Optional) Fix two-column PDF extraction

If your PDF is a two-column academic paper, run this first for cleaner text:

```bash
python extract_text.py
```

This generates `data/paper.txt` which the app will use automatically.

---

## ▶️ Usage

```bash
# Run in offline mode (after models are downloaded)
set TRANSFORMERS_OFFLINE=1   # Windows
export TRANSFORMERS_OFFLINE=1  # Linux/Mac

python app.py
```

Then type your question:

```
Ask a question (or type 'exit' to quit): What dataset is used in the paper?
📌 Answer: The CYCCD dataset is used, consisting of 140k training, 20k validation, and 40k test conversations.
```

Type `exit` to quit.

---

## 📦 Dependencies

```txt
sentence-transformers
transformers
faiss-cpu
pypdf
pymupdf
torch
```

Install all at once:

```bash
pip install sentence-transformers transformers faiss-cpu pypdf pymupdf torch
```

---

## 🔧 Configuration

You can tune these parameters in the respective files:

| Parameter | File | Default | Description |
|---|---|---|---|
| `chunk_size` | `search/chunking.py` | `200` | Words per chunk |
| `overlap` | `search/chunking.py` | `50` | Overlap between chunks |
| `top_k` | `search/retriever.py` | `3` | Number of chunks retrieved |
| `max_new_tokens` | `rag/generator.py` | `150` | Max answer length |
| `context` limit | `rag/generator.py` | `1000` chars | Context passed to model |

---

## ⚠️ Known Limitations

- **Two-column PDFs** — `pypdf` may garble text from multi-column academic papers. Use `extract_text.py` (PyMuPDF) for better extraction.
- **flan-t5-base** is a small model (~250M params) and may copy text rather than synthesize clean answers. Upgrade to `flan-t5-large` or use the Claude API for better results.
- **Token limit** — flan-t5 has a 512 token limit. Long contexts are automatically truncated.
- **CPU only** — runs on CPU by default. GPU support can be enabled via the `device` parameter in the pipeline.

---

## 🔮 Future Improvements

- [ ] Support multiple PDFs at once
- [ ] Add a web UI (Streamlit or Gradio)
- [ ] Upgrade to a better generator model (flan-t5-xl, Mistral, Claude API)
- [ ] Add source chunk citation in answers
- [ ] Support `.txt` and `.docx` input formats
- [ ] Add conversation memory for follow-up questions

---

## 📄 License

MIT License — feel free to use, modify, and distribute.

---

## 🙏 Acknowledgements

- [Sentence Transformers](https://www.sbert.net/) — for the embedding model
- [HuggingFace Transformers](https://huggingface.co/) — for flan-t5
- [FAISS](https://github.com/facebookresearch/faiss) — for vector similarity search
- [PyMuPDF](https://pymupdf.readthedocs.io/) — for robust PDF text extraction
