import faiss
import numpy as np

class FAISSIndex:
    def __init__(self, dim):
        self.index = faiss.IndexFlatL2(dim)

    def add(self, embeddings):
        self.index.add(np.array(embeddings))

    def search(self, query_embedding, k=3):
        distances, indices = self.index.search(query_embedding, k)
        return indices[0]