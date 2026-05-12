import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

from PIL import Image
from datasets import load_dataset
from difflib import SequenceMatcher

from search.chunking import chunk_text
from model.embedding import get_embeddings
from search.faiss_index import FAISSIndex
from search.retriever import Retriever
from rag.generator import generate_answer


# ── ANLS Scorer ──────────────────────────────────────
def anls_score(predicted, ground_truths, threshold=0.5):
    best = 0.0
    predicted_clean = predicted.lower().strip()
    
    for gt in ground_truths:
        gt_clean = gt.lower().strip()
        
        if max(len(predicted_clean), len(gt_clean)) == 0:
            continue
        
        nls = 1 - SequenceMatcher(None, predicted_clean, gt_clean).ratio()
        similarity = (1 - nls) if nls < threshold else 0.0
        best = max(best, similarity)

    return best

def compute_dataset_anls(predictions, ground_truths_list):
    scores = [
        anls_score(pred, gts)
        for pred, gts in zip(predictions, ground_truths_list)
    ]
    return sum(scores) / len(scores)


# ── OCR ──────────────────────────────────────────────
def ocr_image(pil_image):
    text = pytesseract.image_to_string(
        pil_image,
        lang='eng',
        config='--psm 3 --oem 3'
    )
    return text


# ── Build RAG index from text ─────────────────────────
def build_index_from_text(text):
    chunks = chunk_text(text)
    if not chunks:
        return None
    embeddings = get_embeddings(chunks)
    index = FAISSIndex(len(embeddings[0]))
    index.add(embeddings)
    retriever = Retriever(index, chunks)
    return retriever


# ── Main Evaluation ───────────────────────────────────
def run_docvqa_eval(num_samples=50):
    print("Loading DocVQA samples via streaming...")
    dataset = load_dataset(
        "HuggingFaceM4/DocumentVQA",
        split="validation",
        streaming=True
    )
    samples = list(dataset.take(num_samples))
    print(f"Loaded {len(samples)} samples\n")

    predictions        = []
    ground_truths_list = []
    latencies          = []
    ocr_empty_count    = 0
    hit_count          = 0

    for i, sample in enumerate(samples):
        print(f"[{i+1}/{num_samples}] Q: {sample['question']}")

        # Step 1 — OCR the document image
        try:
            text = ocr_image(sample['image'])
        except Exception as e:
            print(f"  ⚠️  OCR failed: {e}")
            predictions.append("")
            ground_truths_list.append(sample['answers'])
            ocr_empty_count += 1
            continue

        if len(text.strip()) < 50:
            print(f"  ⚠️  OCR returned too little text — skipping")
            predictions.append("")
            ground_truths_list.append(sample['answers'])
            ocr_empty_count += 1
            continue

        # Step 2 — Build RAG index
        retriever = build_index_from_text(text)
        if retriever is None:
            print(f"  ⚠️  No chunks generated — skipping")
            predictions.append("")
            ground_truths_list.append(sample['answers'])
            continue

        # Step 3 — Retrieve + Generate
        start  = time.time()
        chunks = retriever.retrieve(sample['question'], k=3)
        answer = generate_answer(sample['question'], chunks)
        latencies.append((time.time() - start) * 1000)

        predictions.append(answer)
        ground_truths_list.append(sample['answers'])

        # Per-sample ANLS
        sample_anls = anls_score(answer, sample['answers'])
        if sample_anls > 0.5:
            hit_count += 1

        print(f"  GT    : {sample['answers']}")
        print(f"  Got   : {answer}")
        print(f"  ANLS  : {sample_anls:.3f}")
        print()

    # ── Final Scores ──
    anls = compute_dataset_anls(predictions, ground_truths_list)

    print("=" * 50)
    print(f"DocVQA ANLS Score : {anls:.4f}  ({anls*100:.1f}%)")
    print(f"Hit Rate (>0.5)   : {hit_count}/{num_samples}  ({hit_count/num_samples*100:.1f}%)")
    print(f"Avg Latency       : {sum(latencies)/len(latencies):.1f} ms" if latencies else "Avg Latency: N/A")
    print(f"OCR failures      : {ocr_empty_count}/{num_samples}")
    print(f"Samples evaluated : {num_samples}")
    print("=" * 50)

    return anls


if __name__ == "__main__":
    run_docvqa_eval(num_samples=50)