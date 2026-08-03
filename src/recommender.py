import pickle
import numpy as np
from src.transform import extract_embedding
from src.similarity import top_n_recommendations
from pathlib import Path
base_dir = Path(__file__).resolve().parent.parent
out_dir = base_dir / "data" / "embeddings"
def save_embeddings(embeddings, image_paths, out_dir):
    np.save(f"{out_dir}/image_embeddings.npy", np.array(embeddings))
    with open(f"{out_dir}/image_paths.pkl", "wb") as f:
        pickle.dump(image_paths, f)

def load_embeddings(out_dir):
    embeddings = np.load(f"{out_dir}/image_embeddings.npy")
    with open(f"{out_dir}/image_paths.pkl", "rb") as f:
        image_paths = pickle.load(f)
    return embeddings, image_paths

def recommend(query_image_path, model, embeddings, image_paths, device="cpu", n=5):
    query_emb = extract_embedding(query_image_path, model, device)
    return top_n_recommendations(query_emb, embeddings, image_paths, n=n)