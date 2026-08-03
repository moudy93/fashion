from pathlib import Path
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.transform import load_model, extract_embedding
from src.similarity import top_n_recommendations

BASE_DIR = Path(__file__).resolve().parent.parent
EMBEDDINGS_DIR = BASE_DIR / "data" / "embeddings"
QUERY_IMAGE = BASE_DIR / "data" / "clean_images" / "0000cdba64314d84a49ed1c266589cc0.jpg"
OUTPUT_PLOT = BASE_DIR /"output"/"model_output_sample.png"


def generate_sample_output(query_image_path=QUERY_IMAGE, output_path=OUTPUT_PLOT, top_n=5):
    model = load_model()
    embeddings = np.load(EMBEDDINGS_DIR / "image_embeddings.npy")
    with open(EMBEDDINGS_DIR / "image_paths.pkl", "rb") as handle:
        image_paths = pickle.load(handle)

    query_embedding = extract_embedding(str(query_image_path), model)
    recommendations = top_n_recommendations(query_embedding, embeddings, image_paths, n=top_n)

    fig, axes = plt.subplots(1, top_n + 1, figsize=(18, 4))
    axes[0].imshow(plt.imread(query_image_path))
    axes[0].set_title("Input image")
    axes[0].axis("off")

    for idx, (recommended_image, score) in enumerate(recommendations, start=1):
        axes[idx].imshow(plt.imread(recommended_image))
        axes[idx].set_title(f"Cosine: {score:.4f}")
        axes[idx].axis("off")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return recommendations


if __name__ == "__main__":
    recommendations = generate_sample_output()
    print("Generated sample output with top 5 recommendations:")
    for image_path, score in recommendations:
        print(f"{Path(image_path).name}: {score:.4f}")
