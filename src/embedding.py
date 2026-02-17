import numpy as np
import torch
from PIL import Image

from src.models import get_models
from src.preprocessing import generate_variants


def extract_single(img_np):
    # Load models only when needed (lazy loading)
    mtcnn, resnet = get_models()

    img = Image.fromarray(img_np)
    face = mtcnn(img)

    if face is None:
        return None

    with torch.no_grad():
        emb = resnet(face.unsqueeze(0))[0].cpu().numpy()

    return emb / np.linalg.norm(emb)


def extract_multi(img_np):
    embeddings = {}

    for name, variant in generate_variants(img_np).items():
        emb = extract_single(variant)
        if emb is not None:
            embeddings[name] = emb

    return embeddings