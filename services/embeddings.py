from sentence_transformers import SentenceTransformer

class Embedder:
    def __init__(self, model_name: str):
        self.model_name=model_name
        self.model=SentenceTransformer(model_name)
    def encode(self, texts):
        if not texts: return []
        return self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
