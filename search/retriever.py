from model.embedding import get_embeddings

class Retriever:
    def __init__(self, index, chunks):
        self.index = index
        self.chunks = chunks

    def retrieve(self, query, k=3):
        query_embedding = get_embeddings([query])
        indices = self.index.search(query_embedding, k)

        return [self.chunks[i] for i in indices]