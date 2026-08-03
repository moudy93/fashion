from torchvision import transforms, models
import torch
import numpy as np
from PIL import Image

def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

def load_model(device="cpu"):
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = torch.nn.Identity()
    model.to(device)
    model.eval()
    return model

def extract_embedding(image_path, model, device="cpu"):
    transform = get_transform()
    image = Image.open(image_path).convert("RGB")
    x = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        emb = model(x).cpu().numpy().squeeze()

    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm
    return emb