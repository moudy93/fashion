from pathlib import Path
from src.transform import load_model, extract_embedding
from src.recommender import save_embeddings

base_dir = Path(__file__).resolve().parent.parent
CLEAN_IMAGE_DIR = base_dir / "data" / "clean_images"
EMBEDDINGS_DIR = base_dir / "data" / "embeddings"

def main():
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

    model = load_model()
    image_paths = list(CLEAN_IMAGE_DIR.glob("*"))

    embeddings = [extract_embedding(p, model) for p in image_paths]

    save_embeddings(
        embeddings,
        [str(p) for p in image_paths],
        str(EMBEDDINGS_DIR)
    )

if __name__ == "__main__":
    main()