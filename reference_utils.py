# reference_utils.py
import os
import random
from PIL import Image


DATASET_ROOT = "dataset"  # โฟลเดอร์เก็บ reference images แยกสถานที่


def get_random_reference(place_name):
   """สุ่มเลือก reference image ของสถานที่"""
   folder_path = os.path.join(DATASET_ROOT, place_name)
   if not os.path.exists(folder_path):
       raise ValueError(f"No reference folder for {place_name}")


   images = [
       os.path.join(folder_path, f) for f in os.listdir(folder_path)
       if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
   ]
   if not images:
       raise ValueError(f"No reference images in folder {folder_path}")


   return random.choice(images)




def compute_similarity(image_path1, image_path2, model, processor):
   """ใช้ CLIP วัดความใกล้เคียงของภาพ"""
   from PIL import Image
   import torch


   img1 = Image.open(image_path1).convert("RGB")
   img2 = Image.open(image_path2).convert("RGB")


   inputs = processor(images=[img1, img2], return_tensors="pt", padding=True)
   outputs = model.get_image_features(**inputs)
  
   # normalize
   outputs = outputs / outputs.norm(p=2, dim=-1, keepdim=True)
  
   similarity = torch.matmul(outputs[0], outputs[1].T).item()
   return similarity