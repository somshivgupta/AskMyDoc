def chunk_text(text, chunk_size=200, overlap=50):
    # Remove references/bibliography section
    cutoff_keywords = ["references\n", "bibliography\n", "works cited\n"]
    lower_text = text.lower()
    for keyword in cutoff_keywords:
        idx = lower_text.find(keyword)
        if idx != -1:
            text = text[:idx]
            break

    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():  # skip empty chunks
            chunks.append(chunk)

    return chunks