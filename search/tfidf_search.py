from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class TFIDFSearch:
    def __init__(self, documents):
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform(documents)
        self.documents = documents

    def search(self, query, k=3):
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.tfidf_matrix)
        
        top_indices = scores[0].argsort()[-k:][::-1]
        
        return [self.documents[i] for i in top_indices]