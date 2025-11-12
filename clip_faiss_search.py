"""
clip_faiss_search_gpu.py

Requirements:
  pip install pillow torch torchvision ftfy regex tqdm faiss-gpu
  pip install git+https://github.com/openai/CLIP.git

Features:
  - Automatic TOP_K adjustment
  - GPU FAISS support
  - Self-similarity removal optional
"""

import os
import torch
import clip
from PIL import Image
import numpy as np
import faiss

# === CONFIG ===
IMAGE_FOLDER = "./images"   # folder containing images
TOP_K = 3                   # desired top-K neighbors
IGNORE_SELF = True          # skip the query image itself in results

# === Setup device and model ===
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()

# === Load images ===
image_files = [os.path.join(IMAGE_FOLDER, f) for f in os.listdir(IMAGE_FOLDER)
               if f.lower().endswith((".png", ".jpg", ".jpeg"))]

num_images = len(image_files)
if num_images == 0:
    raise ValueError(f"No images found in {IMAGE_FOLDER}")

print(f"Found {num_images} images.")

# Adjust TOP_K if more than number of images
actual_top_k = min(TOP_K + 1 if IGNORE_SELF else TOP_K, num_images)

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

# === Build FAISS index on GPU ===
dim = embeddings.shape[1]
cpu_index = faiss.IndexFlatIP(dim)  # Inner product (cosine similarity with normalized vectors)
gpu_res = faiss.StandardGpuResources()
index = faiss.index_cpu_to_gpu(gpu_res, 0, cpu_index)
index.add(embeddings)
print(f"FAISS GPU index built with {index.ntotal} vectors.")

# === Similarity search for each image ===
for i, query_path in enumerate(image_files):
    query_embedding = embeddings[i].reshape(1, -1)
    distances, indices = index.search(query_embedding, actual_top_k)

    if IGNORE_SELF:
        # skip first result (self-similarity)
        distances = distances[0][1:]
        indices = indices[0][1:]
    else:
        distances = distances[0]
        indices = indices[0]

    print(f"\nTop {len(indices)} similar images to {query_path}:")
    for rank, (idx, dist) in enumerate(zip(indices, distances), start=1):
        print(f"{rank}. {image_files[idx]}  | similarity: {dist:.4f}")
