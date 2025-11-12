import torch
import clip
from PIL import Image

# Load the model to GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Example image & text
image = preprocess(Image.open("sample.jpg")).unsqueeze(0).to(device)
text = clip.tokenize(["a photo of a cat", "a photo of a dog"]).to(device)

# Run CLIP
with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    similarity = (image_features @ text_features.T).softmax(dim=-1)

print("Similarity scores:", similarity)


  git config --global user.email "sandeeps@kaluanjewellers.tech"
  git config --global user.name "s-sandeep-tech"