from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

_model = None
_tokenizer = None

def _get_generator():
    global _model, _tokenizer
    if _model is None:
        print("Loading generator model...")
        _tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
        _model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")
        _model.eval()
        print("✅ Generator ready!")
    return _model, _tokenizer

def generate_answer(query, context_chunks):
    context = " ".join(context_chunks)
    context = context[:1500]  # increased from 1000

    prompt = f"""You are a helpful research assistant. Read the context carefully and write a complete, detailed answer to the question.

Context: {context}

Question: {query}

Write a detailed answer in 2-3 sentences:"""

    model, tokenizer = _get_generator()

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=512,
        truncation=True
    )

    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=200,      # increased from 150
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3,
            length_penalty=2.0       # encourages longer answers
        )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer