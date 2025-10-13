# classifier.py
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch


# โหลดโมเดล CLIP
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


ALL_PLACES = [
   "Ratchadamnoen Avenue – Democracy Monument",
   "Sala Chalermkrung Royal Theatre",
   "Giant Swing – Wat Suthat",
   "Khao San Road",
   "Phra Sumen Fort – Santichaiprakan Park",
   "National Museum Bangkok",
   "Yaowarat (Chinatown)",
   "Sanam Luang (Royal Field)"
]



def check_image_category(image_path, expected_place):
    image = Image.open(image_path).convert("RGB")
    texts = ALL_PLACES
    inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1)
    return probs[0][texts.index(expected_place)].item()