import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

MODEL_NAME = "mesabo/agri-plant-disease-resnet50"

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

print("Using device:", device)
print("Loading agriculture disease model...")

processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)

model = model.to(device)
model.eval()

image_path = input("\nEnter image path: ").strip()

image = Image.open(image_path).convert("RGB")

inputs = processor(images=image, return_tensors="pt")

inputs = {
    key: value.to(device)
    for key, value in inputs.items()
}

with torch.no_grad():
    outputs = model(**inputs)

probabilities = torch.nn.functional.softmax(
    outputs.logits,
    dim=-1
)[0]

top_results = torch.topk(probabilities, 5)

print("\n🌱 AGRICULTURE AI RESULTS")
print("=" * 45)

for score, index in zip(
    top_results.values,
    top_results.indices
):

    label = model.config.id2label[index.item()]
    confidence = score.item() * 100

    print(
        f"{label}: {confidence:.2f}%"
    )