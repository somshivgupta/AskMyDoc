import numpy as np

def semantic_search(query, model, index, documents, k=3):
    query_vec = model.encode([query])
    
    distances, indices = index.search(np.array(query_vec), k)
    
    results = [documents[i] for i in indices[0]]
    
    return results