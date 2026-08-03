from flask import Flask, render_template, request, url_for
from pathlib import Path
import numpy as np
import pickle
import shutil
from src.transform import load_model
from src.recommender import recommend

base_dir = Path(__file__).resolve().parent

EMBEDDINGS_DIR = base_dir / "data/embeddings"
UPLOAD_DIR = base_dir / "app/static/uploads"
DATASET_IMAGE_DIR = base_dir / "data/clean_images"
STATIC_RECOMMENDATION_DIR = base_dir / "app/static/recommendations"

app = Flask(__name__, static_folder="app/static", static_url_path="/static")
model = load_model()

embeddings = np.load(f"{EMBEDDINGS_DIR}/image_embeddings.npy")
with open(f"{EMBEDDINGS_DIR}/image_paths.pkl", "rb") as f:
    image_paths = pickle.load(f)

Path(UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
Path(STATIC_RECOMMENDATION_DIR).mkdir(parents=True, exist_ok=True)

for image_path in DATASET_IMAGE_DIR.glob("*"):
    destination = STATIC_RECOMMENDATION_DIR / image_path.name
    if not destination.exists():
        shutil.copyfile(image_path, destination)

@app.route("/", methods=["GET", "POST"])
def index():
    results = []
    uploaded = None
    if request.method == "POST":
        file = request.files.get("image")
        if file and file.filename:
            filename = Path(file.filename).name
            upload_path = UPLOAD_DIR / filename
            file.save(upload_path)
            uploaded = url_for("static", filename=f"uploads/{filename}")
            recommendation_pairs = recommend(str(upload_path), model, embeddings, image_paths, n=5)
            results = [
                {
                    "image": url_for("static", filename=f"recommendations/{Path(image_path).name}"),
                    "score": round(score, 4)
                }
                for image_path, score in recommendation_pairs
            ]
    return render_template("index.html", results=results, uploaded=uploaded)

if __name__ == "__main__":
    app.run()