# reference_prompt_builder.py
import os
import random
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from reference_utils import compute_similarity, get_random_reference
from classifier import model as clip_model, processor as clip_processor

# โหลด BLIP caption model สำหรับอธิบายภาพ
caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

DATASET_ROOT = "dataset"  # โฟลเดอร์ dataset หลัก

# Mapping ของชื่อสถานที่ → โฟลเดอร์ใน dataset
PLACE_NAME_TO_FOLDER = {
    "Ratchadamnoen Avenue – Democracy Monument": "Ratchadamnoen Avenue – Democracy Monument",
    "Sala Chalermkrung Royal Theatre": "Sala Chalermkrung Royal Theatre",
    "Giant Swing – Wat Suthat": "Giant Swing - Wat Suthat",
    "Phra Sumen Fort – Santichaiprakan Park": "Phra Sumen Fort – Santichaiprakarn Park",
    "National Museum Bangkok": "Phra Nakhon National Museum",
    "Yaowarat (Chinatown)": "Yaowarat (Chinatown)",
    "Sanam Luang (Royal Field)": "Sanam Luang (Royal_Field)"
}

def describe_images_from_folder(folder, limit=3):
    """สรุปคำบรรยายภาพจาก folder"""
    captions = []
    images_list = [os.path.join(folder, f) for f in os.listdir(folder)
                   if f.lower().endswith((".jpg",".jpeg",".png",".webp"))]
    imgs = random.sample(images_list, min(limit, len(images_list)))
    for img_path in imgs:
        raw_image = Image.open(img_path).convert('RGB')
        inputs = caption_processor(raw_image, return_tensors="pt")
        out = caption_model.generate(**inputs, max_length=50)
        caption = caption_processor.decode(out[0], skip_special_tokens=True)
        captions.append(caption)
    return " ".join(captions)

def build_prompt(place_name, user_image_path=None):
    """
    สร้าง prompt สำหรับ OpenAI โดยอิง dataset และ Art Objects
    - place_name: ชื่อสถานที่
    - user_image_path: path ของภาพผู้ใช้ (optional)
    """
    # แปลงชื่อสถานที่เป็นชื่อโฟลเดอร์ dataset
    folder_name = PLACE_NAME_TO_FOLDER.get(place_name)
    if not folder_name:
        raise ValueError(f"No dataset mapping for {place_name}")
    place_folder = os.path.join(DATASET_ROOT, folder_name)

    if not os.path.exists(place_folder):
        raise ValueError(f"No dataset folder found at {place_folder}")

    # 1a. เลือก reference images
    selected_refs = []
    if user_image_path:
        similarities = []
        for f in os.listdir(place_folder):
            img_path = os.path.join(place_folder, f)
            if img_path.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                sim = compute_similarity(user_image_path, img_path, clip_model, clip_processor)
                similarities.append((sim, img_path))
        similarities.sort(reverse=True)
        selected_refs = [img_path for _, img_path in similarities[:3]]
    else:
        selected_refs = [get_random_reference(place_folder)]

    # 1b. อธิบายภาพ reference
    place_desc = describe_images_from_folder(place_folder)

    # 2. Art Objects
    art_folder = os.path.join(DATASET_ROOT, "Art Objects")
    art_desc = describe_images_from_folder(art_folder)

    # 3. Prompt ระดับสูงสำหรับสร้างบรรยากาศยุค 1960s
    prompt = (
        f"Transform the uploaded photo to depict {place_name} in Bangkok during the 1960s with high realism. "
        f"Use the following reference visual cues and textures: {place_desc}. "
        f"Integrate subtle architectural and artistic elements inspired by {art_desc}. "
        "Preserve natural lighting and colors (do NOT use sepia, vintage filters, or artificial color tinting). "
        "Minimize modern elements: remove visible modern vehicles, street signs, advertisements, or contemporary clothing if present. "
        "Keep people minimal and realistic; include only if historically accurate for the 1960s street scene. "
        "Maintain perspective, depth, and proportion as in the original uploaded image. "
        "The final output should look like a natural, authentic 1960s urban street photograph: "
        "accurate materials, shopfronts, signage, building textures, and urban atmosphere, without adding excessive objects or elements not present in the original scene."
    )
    return prompt