import faiss
import numpy as np
import time

class FAISSIndex:
    def __init__(self, dim):
        self.index = faiss.IndexFlatL2(dim)

    def add(self, embeddings):
        start = time.time()
        self.index.add(np.array(embeddings))
        end = time.time()
        print(f"Add Time: {(end - start)*1000:.2f} ms")

    def search(self, query_embedding, k=3):
        start = time.time()
        distances, indices = self.index.search(query_embedding, k)
        end = time.time()
        
        latency = (end - start) * 1000  # ms
        print(f"Search Latency: {latency:.4f} ms")
        
        return indices[0]