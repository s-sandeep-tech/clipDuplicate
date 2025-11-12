"""
clip_faiss_search_gpu_cached.py

Features:
- GPU CLIP embeddings
- GPU FAISS index
- Stores FAISS index and image filenames for reuse
- Automatically rebuilds embeddings/index if new images are added
- Dynamic TOP_K and optional self-similarity removal
"""

import os
import torch
import clip
from PIL import Image
import numpy as np
import faiss

# === CONFIG ===
IMAGE_FOLDER = "./images"       # folder containing images
TOP_K = 3                       # number of neighbors to retrieve
IGNORE_SELF = True              # skip query image itself
FAISS_INDEX_FILE = "faiss_index.bin"
IMAGE_LIST_FILE = "image_files.npy"

# === Setup device and model ===
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()

# === Load all images from folder ===
image_files = [os.path.join(IMAGE_FOLDER, f) for f in os.listdir(IMAGE_FOLDER)
               if f.lower().endswith((".png", ".jpg", ".jpeg"))]
num_images = len(image_files)
if num_images == 0:
    raise ValueError(f"No images found in {IMAGE_FOLDER}")

# === Check if FAISS index exists ===
rebuild_index = True
if os.path.exists(FAISS_INDEX_FILE) and os.path.exists(IMAGE_LIST_FILE):
    saved_files = np.load(IMAGE_LIST_FILE).tolist()
    # rebuild if new images added
    if set(saved_files) == set(image_files):
        rebuild_index = False
        print("Loading FAISS index from disk...")
        cpu_index = faiss.read_index(FAISS_INDEX_FILE)
        gpu_res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(gpu_res, 0, cpu_index)
        embeddings = None  # embeddings not needed in memory
    else:
        print("New images detected. Rebuilding FAISS index...")

if rebuild_index:
    print(f"Generating CLIP embeddings for {num_images} images...")
    embeddings = []
    for img_path in image_files:
        image = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = model.encode_image(image)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            embeddings.append(emb.cpu().numpy())

    embeddings = np.vstack(embeddings).astype("float32")
    print("Generated embeddings shape:", embeddings.shape)

    # === Build GPU FAISS index ===
    dim = embeddings.shape[1]
    cpu_index = faiss.IndexFlatIP(dim)
    gpu_res = faiss.StandardGpuResources()
    index = faiss.index_cpu_to_gpu(gpu_res, 0, cpu_index)
    index.add(embeddings)
    print(f"FAISS GPU index built with {index.ntotal} vectors.")

    # Save index and image filenames
    faiss.write_index(faiss.index_gpu_to_cpu(index), FAISS_INDEX_FILE)
    np.save(IMAGE_LIST_FILE, np.array(image_files))
    print("FAISS index and image list saved to disk.")

# === Determine actual TOP_K ===
actual_top_k = min(TOP_K + 1 if IGNORE_SELF else TOP_K, num_images)

# === Perform similarity search for each image ===
for i, query_path in enumerate(image_files):
    # Use embeddings if available; otherwise retrieve from index directly
    if embeddings is not None:
        query_embedding = embeddings[i].reshape(1, -1)
    else:
        # Build a temporary query embedding on the fly
        image = preprocess(Image.open(query_path).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            q_emb = model.encode_image(image)
            query_embedding = (q_emb / q_emb.norm(dim=-1, keepdim=True)).cpu().numpy()

    distances, indices = index.search(query_embedding, actual_top_k)

    if IGNORE_SELF:
        distances = distances[0][1:]
        indices = indices[0][1:]
    else:
        distances = distances[0]
        indices = indices[0]

    print(f"\nTop {len(indices)} similar images to {query_path}:")
    for rank, (idx, dist) in enumerate(zip(indices, distances), start=1):
        print(f"{rank}. {image_files[idx]}  | similarity: {dist:.4f}")
