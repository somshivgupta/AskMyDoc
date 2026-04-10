from transformers import pipeline

_generator = None

def _get_generator():
    global _generator
    if _generator is None:
        print("Loading generator model...")
        _generator = pipeline("text2text-generation", model="google/flan-t5-base")
        print("✅ Generator ready!")
    return _generator

def generate_answer(query, context_chunks):
    # Limit context to avoid exceeding 512 token limit
    context = " ".join(context_chunks)
    context = context[:1000]  # trim to ~500 tokens worth of text

    prompt = f"""Answer the question using ONLY the context below.

Context:
{context}

Question:
{query}

Answer:"""

    # Only use max_new_tokens, not max_length
    generator = _get_generator()
    result = generator(prompt, max_new_tokens=150, do_sample=False, truncation=True)
    return result[0]["generated_text"]