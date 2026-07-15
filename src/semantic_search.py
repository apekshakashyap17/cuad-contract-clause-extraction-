"""
semantic_search.py
-------------------
Bonus feature: semantic search over extracted clauses using sentence
embeddings + FAISS, so you can ask e.g. "which contracts let either party
walk away with 30 days notice?" and get back the most relevant clauses
across the whole processed batch, not just keyword matches.
"""

import os
import pickle

INDEX_DIR = os.path.join(os.path.dirname(__file__), "..", "output", "clause_index")
_MODEL_NAME = "all-MiniLM-L6-v2"  


class ClauseIndex:
    def __init__(self):
        import faiss

        from sentence_transformers import SentenceTransformer

        self._faiss = faiss
        self.model = SentenceTransformer(_MODEL_NAME)
        self.index = None
        self.metadata = []  

    def build(self, records: list):
        """records: list of dicts with contract_id, clause_type, text."""
        import numpy as np

        texts = [r["text"] for r in records if r["text"] and r["text"] != "NOT FOUND"]
        kept = [r for r in records if r["text"] and r["text"] != "NOT FOUND"]
        if not texts:
            return
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        dim = embeddings.shape[1]
        self.index = self._faiss.IndexFlatIP(dim)
        self.index.add(np.array(embeddings, dtype="float32"))
        self.metadata = kept

    def search(self, query: str, top_k: int = 5):
        import numpy as np

        if self.index is None:
            return []
        q_emb = self.model.encode([query], normalize_embeddings=True)
        scores, idxs = self.index.search(np.array(q_emb, dtype="float32"), top_k)
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            meta = self.metadata[idx]
            results.append({**meta, "score": float(score)})
        return results

    def save(self, path: str = INDEX_DIR):
        os.makedirs(path, exist_ok=True)
        self._faiss.write_index(self.index, os.path.join(path, "index.faiss"))
        with open(os.path.join(path, "metadata.pkl"), "wb") as f:
            pickle.dump(self.metadata, f)

    def load(self, path: str = INDEX_DIR):
        self.index = self._faiss.read_index(os.path.join(path, "index.faiss"))
        with open(os.path.join(path, "metadata.pkl"), "rb") as f:
            self.metadata = pickle.load(f)
