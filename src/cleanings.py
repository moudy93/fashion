from PIL import Image
from pathlib import Path
import hashlib

def is_valid_image(path):
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False

def hash_image(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def normalize_image(src_path, dst_path, size=(224, 224)):
    with Image.open(src_path) as img:
        img = img.convert("RGB")
        img = img.resize(size)
        img.save(dst_path, quality=95)

def clean_images(raw_dir, clean_dir):
    raw_dir = Path(raw_dir)
    clean_dir = Path(clean_dir)
    clean_dir.mkdir(parents=True, exist_ok=True)

    seen_hashes = set()

    for img_path in raw_dir.iterdir():
        if not img_path.is_file():
            continue
        if not is_valid_image(img_path):
            continue

        h = hash_image(img_path)
        if h in seen_hashes:
            continue
        seen_hashes.add(h)

        out_path = clean_dir / img_path.name
        normalize_image(img_path, out_path)

base_dir = Path(__file__).resolve().parent.parent
raw_dir = base_dir / "data" / "images"
clean_dir = base_dir / "data" / "clean_images"

clean_images(raw_dir, clean_dir)