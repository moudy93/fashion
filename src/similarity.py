import numpy as np

def cosine_similarity(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def top_n_recommendations(query_emb, embeddings, image_paths, n=5):
    scores = [cosine_similarity(query_emb, emb) for emb in embeddings]
    ranked = sorted(zip(image_paths, scores), key=lambda x: x[1], reverse=True)
    return ranked[:n]