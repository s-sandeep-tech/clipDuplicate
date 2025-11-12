"""
clip_faiss_search.py

Requirements:
  pip install pillow torch torchvision ftfy regex tqdm faiss-cpu
  pip install git+https://github.com/openai/CLIP.git

Usage:
  1. Put all images in a folder, e.g., ./images/
  2. Update IMAGE_FOLDER variable
  3. Run: python3 clip_faiss_search.py
"""

import os
import torch
import clip
from PIL import Image
import numpy as np
import faiss

# === CONFIG ===
IMAGE_FOLDER = "./images"   # folder containing images
TOP_K = 3                   # number of nearest neighbors to retrieve

# === Setup device and model ===
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()

# === Load images ===
image_files = [os.path.join(IMAGE_FOLDER, f) for f in os.listdir(IMAGE_FOLDER)
               if f.lower().endswith((".png", ".jpg", ".jpeg"))]

if len(image_files) == 0:
    raise ValueError(f"No images found in {IMAGE_FOLDER}")

print(f"Found {len(image_files)} images.")

# === Compute CLIP embeddings ===
embeddings = []
for img_path in image_files:
    image = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model.encode_image(image)
        emb = emb / emb.norm(dim=-1, keepdim=True)  # normalize
        embeddings.append(emb.cpu().numpy())

embeddings = np.vstack(embeddings).astype("float32")
print("Generated embeddings shape:", embeddings.shape)

# === Build FAISS index ===
dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)  # cosine similarity since embeddings are normalized
index.add(embeddings)
print(f"FAISS index built with {index.ntotal} vectors.")

# === Example similarity search ===
# Use the first image as query
query_embedding = embeddings[0].reshape(1, -1)
distances, indices = index.search(query_embedding, TOP_K)

print(f"\nTop {TOP_K} similar images to {image_files[0]}:")
for rank, (idx, dist) in enumerate(zip(indices[0], distances[0]), start=1):
    print(f"{rank}. {image_files[idx]}  | similarity: {dist:.4f}")
