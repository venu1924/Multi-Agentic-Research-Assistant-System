from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")


def hybrid_ranking(query, papers, top_k=15):

    texts = [p["title"] + " " + p["abstract"] for p in papers]

    emb = model.encode(texts, convert_to_numpy=True)
    q_emb = model.encode([query], convert_to_numpy=True)[0]

    scores = np.dot(emb, q_emb)

    idx = np.argsort(scores)[::-1][:top_k]

    return [(papers[i], float(scores[i])) for i in idx]