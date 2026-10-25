import torch
from PIL import Image
from transformers import pipeline


# Use Apple Silicon GPU when available
device = "mps" if torch.backends.mps.is_available() else "cpu"

print("Using device:", device)

classifier = pipeline(
    "image-classification",
    model="microsoft/resnet-50",
    device=device
)

image_path = input("Enter image path: ")

image = Image.open(image_path).convert("RGB")

results = classifier(image)

print("\nImage Classification Results:\n")

for result in results[:5]:
    print(
        f"{result['label']}: "
        f"{result['score'] * 100:.2f}%"
    )