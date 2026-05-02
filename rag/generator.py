import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2"

def generate_answer(query, context_chunks):
    if not context_chunks:
        return "I couldn't find relevant information to answer your question."

    context = "\n\n---\n\n".join(context_chunks)

    prompt = f"""You are a helpful research assistant. Answer the question using only the context below. Be detailed and complete.

Context:
{context}

Question: {query}

Answer:"""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "num_predict": 300,
                }
            },
            timeout=120
        )
        response.raise_for_status()
        return response.json()["response"].strip()

    except requests.exceptions.ConnectionError:
        return "❌ Ollama is not running. Start it with: ollama serve"
    except requests.exceptions.Timeout:
        return "❌ Ollama timed out. Try a smaller model like phi3.5."
    except Exception as e:
        return f"❌ Generation failed: {e}"