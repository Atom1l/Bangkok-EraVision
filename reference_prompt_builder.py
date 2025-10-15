# reference_prompt_builder.py
import os
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from reference_utils import compute_similarity
from classifier import model as clip_model, processor as clip_processor

# โหลด BLIP caption model สำหรับอธิบายภาพ (ยังคงใช้สำหรับ dynamic part)
caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

DATASET_ROOT = "dataset"

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

# --- คลัง PROMPT เฉพาะสถานที่ (ส่วนที่ 1: เพิ่มเข้ามาใหม่) ---
LOCATION_SPECIFIC_PROMPTS = {
    # ====== ราชดำเนินกลาง-อนุสาวรีย์ประชาธิปไตย ======
    "Ratchadamnoen Avenue – Democracy Monument": """
Your task is to **modify and transform** the uploaded photo, aiming for a result that is both historically accurate and faithful to the original composition.

**Rule 1: The Democracy Monument (The Preservation Rule)**
- **Strictly preserve the monument's entire architectural form, placement, and perspective.**
- **Crucially, retain and enhance ALL intricate surface details, INCLUDING specifically the unique bas-relief carvings on the wings AND the distinctive lion-head figures with a cobra emerging from their mouths at the base of the monument's pillars.** These existing details must be preserved and clarified, NOT replaced or recreated.
- For other non-carving details like the **red doors or golden top**, incorporate additional visual cues from: '{landmark_details_from_dataset}'.

**Rule 2: Surrounding Area - A Tale of Two Zones**
- **Your primary goal is to accurately represent the varied scale and style of buildings around the monument based on their location.**
- **Crucially, the placement, scale, and perspective of all transformed buildings must match the original structures in the uploaded photo to avoid distortion.**

- **Zone A: Buildings at the Immediate Traffic Circle:**
  - Any buildings positioned **directly adjacent to the monument's roundabout** must be transformed into **smaller, 2-3 story commercial buildings or shophouses.**
  - This area should reflect a mix of uses from the era, such as **car showrooms (like the Mercedes-Benz sign seen in photos), small shops, and offices.**
  - **AVOID placing the large, uniform reddish-orange blocks directly at the circle's edge.**

- **Zone B: Buildings Along the Main Avenue (Further in the background):**
  - The iconic, large-scale **mid-20th century Bangkok Modernist buildings** should only appear **further down the avenue, lining the main boulevards in the background.**
  - The architectural transformation for these specific buildings must follow these precise physical characteristics:
    - They must have a distinct, **blocky, and rectilinear form with a flat roofline.**
    - Their most critical feature is the emphasis on **strong horizontal lines, created by prominent concrete ledges and continuous balconies that wrap around the facade between floors.**
    - The facade is not flat but features a **rhythmic pattern of slightly protruding vertical sections.**
    - The **ground floor should be visually distinct from the upper floors,** featuring larger glass windows for shopfronts with simple, non-ornate frames.

- **General Details for All Buildings and the Street:**
  - All buildings must have a **muted, deep reddish-orange or terracotta color (สีส้มอิฐหม่นอมแดง)**.
  - All buildings must look like **real, aged concrete structures, not clean 3D renders.** Introduce realistic imperfections like weathering and subtle water stains.
  - **Windows on ALL buildings MUST be simple, rectangular insets with sharp, 90-degree corners.** AVOID any form of curved or rounded window frames.
  - The street must feature the iconic, ornate, **tall white streetlights topped with a Kinnara (กินรี) figure** and a realistic mix of 1960s vehicles.

- **Use visual cues for texture and fine detail from:** '{surrounding_details_from_dataset}'.

**Atmosphere:** The final image must **match the ambient lighting, weather, and time of day of the original uploaded photo.** Apply a vintage aesthetic that authentically replicates the look of 1960s color film, including its unique color science, saturation, and natural grain. Avoid artificial sepia filters.
""",

    # ====== เยาวราช ======
    "Yaowarat (Chinatown)": """
Your task is to edit the uploaded photo with two distinct rules:

**Rule 1: Main Building/Landmark in User's Photo**
- **Strictly preserve the architectural form of the main building the user has focused on.**
- **For its textures and fine details, draw inspiration from these visual descriptions:** '{landmark_details_from_dataset}'.

**Rule 2: The rest of the Yaowarat Street Scene**
- **Transform the entire surrounding environment into a bustling 1960s Yaowarat street.**
- **Base the style of the shophouses, signs, and street elements on these specific visual cues:** '{surrounding_details_from_dataset}'.
- Add numerous vertical neon signs in both Chinese and Thai characters, and crowd the street with pedicabs (samlors) and vintage cars.

**Atmosphere:** The final image must be energetic and vibrant, capturing the spirit of 1960s Chinatown.
""",

}

def describe_specific_images(image_paths):
    if not image_paths:
        return ""
    captions = []
    for img_path in image_paths:
        try:
            raw_image = Image.open(img_path).convert('RGB')
            inputs = caption_processor(raw_image, return_tensors="pt")
            out = caption_model.generate(**inputs, max_length=50)
            caption = caption_processor.decode(out[0], skip_special_tokens=True)
            captions.append(caption)
        except Exception:
            continue
    return " ".join(captions)

# --- ตัวสร้าง Prompt หลัก (ส่วนที่ 2: ปรับปรุงใหม่ทั้งหมด) ---
# --- ตัวสร้าง Prompt หลัก (ฉบับแก้ไขใหม่ทั้งหมด) ---
def build_prompt(place_name, user_image_path=None):
    """
    สร้าง prompt โดยเลือกจากคลัง และดึงข้อมูลเสริมจาก dataset ที่ง่ายขึ้น
    """
    # 1. เลือก Prompt หลักจากคลัง
    base_prompt = LOCATION_SPECIFIC_PROMPTS.get(place_name)
    if not base_prompt:
        raise ValueError(f"No specific prompt template found for {place_name}")

    # 2. เตรียมตัวแปรสำหรับเก็บคำอธิบาย (ลดเหลือ 2 ส่วน)
    landmark_details_desc = "the monument's original textures like its doors and golden top"
    surrounding_details_desc = "typical 1960s Bangkok street scenes"

    if user_image_path:
        folder_name = PLACE_NAME_TO_FOLDER.get(place_name)
        if folder_name:
            base_place_folder = os.path.join(DATASET_ROOT, folder_name)

            def get_description_from_folder(subfolder_name):
                # ... (โค้ดในฟังก์ชันนี้เหมือนเดิม) ...
                pass # Placeholder for brevity

            # 2a. ค้นหาและบรรยายรายละเอียดของ Landmark (เฉพาะส่วนที่ไม่ใช่ลายแกะสลัก)
            desc = get_description_from_folder("landmark_details")
            if desc: landmark_details_desc = desc

            # 2b. ค้นหาและบรรยายรายละเอียดของอาคารรอบๆ
            desc = get_description_from_folder("surrounding_details")
            if desc: surrounding_details_desc = desc

    # 3. เติมข้อมูลทั้งหมดลงใน Prompt หลัก
    final_prompt = base_prompt.format(
        landmark_details_from_dataset=landmark_details_desc,
        surrounding_details_from_dataset=surrounding_details_desc
    )

    return final_prompt