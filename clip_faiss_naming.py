"""
clip_faiss_naming.py

Features:
- GPU CLIP embeddings
- GPU FAISS index
- Cache FAISS index and image list on disk
- Dynamic TOP_K
- Self-similarity removal
- Query new image not in folder without rebuilding index
- Suggest jewelry design names from candidate list
- Outputs CSV with results
"""

import os
import torch
import clip
from PIL import Image
import numpy as np
import faiss
import csv

# === CONFIG ===
IMAGE_FOLDER = "./images"          # folder containing jewelry images
TOP_K = 3                          # number of neighbors to retrieve
IGNORE_SELF = True                 # skip query image itself
FAISS_INDEX_FILE = "faiss_index.bin"
IMAGE_LIST_FILE = "image_files.npy"
CSV_OUTPUT = "image_similarity_names.csv"

# Candidate names for jewelry designs
CANDIDATE_NAMES = [
    "Elegant Ring",
    "Floral Necklace",
    "Geometric Earrings",
    "Diamond Pendant",
    "Gold Bracelet",
    "Stud Earrings",
    "Gemstone Ring"
]

# === Setup device and model ===
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()

# === Prepare text embeddings for candidate names ===
text_tokens = clip.tokenize(CANDIDATE_NAMES).to(device)
with torch.no_grad():
    text_embeddings = model.encode_text(text_tokens)
    text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)

# === Load images from folder ===
image_files = [os.path.join(IMAGE_FOLDER, f) for f in os.listdir(IMAGE_FOLDER)
               if f.lower().endswith((".png", ".jpg", ".jpeg"))]
num_images = len(image_files)
if num_images == 0:
    raise ValueError(f"No images found in {IMAGE_FOLDER}")

# === Load or rebuild FAISS index ===
rebuild_index = True
if os.path.exists(FAISS_INDEX_FILE) and os.path.exists(IMAGE_LIST_FILE):
    saved_files = np.load(IMAGE_LIST_FILE).tolist()
    if set(saved_files) == set(image_files):
        rebuild_index = False
        print("Dataset unchanged. Loading FAISS index from disk...")
        cpu_index = faiss.read_index(FAISS_INDEX_FILE)
        gpu_res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(gpu_res, 0, cpu_index)
        embeddings = None
    else:
        print("Dataset changed. Rebuilding FAISS index...")

if rebuild_index:
    print(f"Generating CLIP embeddings for {num_images} images...")
    embeddings = []
    suggested_names_list = []
    for img_path in image_files:
        # Encode image
        image = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = model.encode_image(image)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            embeddings.append(emb.cpu().numpy())

            # Suggest design name
            similarity = (emb @ text_embeddings.T).squeeze(0)   # cosine similarity
            best_idx = similarity.argmax().item()
            suggested_name = CANDIDATE_NAMES[best_idx]
            suggested_names_list.append(suggested_name)
            print(f"Suggested name for {os.path.basename(img_path)}: {suggested_name}")

    embeddings = np.vstack(embeddings).astype("float32")
    print("Generated embeddings shape:", embeddings.shape)

    # Build GPU FAISS index
    dim = embeddings.shape[1]
    cpu_index = faiss.IndexFlatIP(dim)
    gpu_res = faiss.StandardGpuResources()
    index = faiss.index_cpu_to_gpu(gpu_res, 0, cpu_index)
    index.add(embeddings)
    print(f"FAISS GPU index built with {index.ntotal} vectors.")

    # Save index and image list
    faiss.write_index(faiss.index_gpu_to_cpu(index), FAISS_INDEX_FILE)
    np.save(IMAGE_LIST_FILE, np.array(image_files))
    print("FAISS index and image list saved to disk.")
else:
    # If dataset unchanged and loading existing index, generate suggested names for output
    suggested_names_list = ["Unknown"] * len(image_files)

# === Determine actual TOP_K ===
actual_top_k = min(TOP_K + 1 if IGNORE_SELF else TOP_K, len(image_files))

# === Helper: search a single query embedding ===
def search_query(query_embedding):
    distances, indices = index.search(query_embedding, actual_top_k)
    if IGNORE_SELF:
        distances = distances[0][1:]
        indices = indices[0][1:]
    else:
        distances = distances[0]
        indices = indices[0]
    return list(zip(indices, distances))

# === Collect results for CSV ===
csv_rows = []

# === 1) Search each image in folder ===
for i, query_path in enumerate(image_files):
    if embeddings is not None:
        query_embedding = embeddings[i].reshape(1, -1)
    else:
        # build embedding on the fly
        image = preprocess(Image.open(query_path).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            q_emb = model.encode_image(image)
            query_embedding = (q_emb / q_emb.norm(dim=-1, keepdim=True)).cpu().numpy().astype("float32")
    results = search_query(query_embedding)

    print(f"\nTop {len(results)} similar images to {query_path}:")
    for rank, (idx, dist) in enumerate(results, start=1):
        print(f"{rank}. {image_files[idx]}  | similarity: {dist:.4f}")
    csv_rows.append({
        "Image": os.path.basename(query_path),
        "Suggested Name": suggested_names_list[i],
        "Top Matches": ", ".join([os.path.basename(image_files[idx]) for idx, _ in results])
    })

# === 2) Optional: search a new image not in folder ===
# new_image_path = "./query.jpg"  # set path to new query image
# if os.path.exists(new_image_path):
#     image = preprocess(Image.open(new_image_path).convert("RGB")).unsqueeze(0).to(device)
#     with torch.no_grad():
#         q_emb = model.encode_image(image)
#         query_embedding = (q_emb / q_emb.norm(dim=-1, keepdim=True)).cpu().numpy().astype("float32")
#     results = search_query(query_embedding)
#     similarity = (q_emb @ text_embeddings.T).squeeze(0)
#     best_idx = similarity.argmax().item()
#     suggested_name = CANDIDATE_NAMES[best_idx]
#     print(f"\nSuggested name for {os.path.basename(new_image_path)}: {suggested_name}")
#     print(f"Top {len(results)} similar images:")
#     for rank, (idx, dist) in enumerate(results, start=1):
#         print(f"{rank}. {image_files[idx]}  | similarity: {dist:.4f}")
#     csv_rows.append({
#         "Image": os.path.basename(new_image_path),
#         "Suggested Name": suggested_name,
#         "Top Matches": ", ".join([os.path.basename(image_files[idx]) for idx, _ in results])
#     })

# === Write CSV ===
with open(CSV_OUTPUT, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["Image", "Suggested Name", "Top Matches"])
    writer.writeheader()
    for row in csv_rows:
        writer.writerow(row)

print(f"\nResults saved to {CSV_OUTPUT}")
