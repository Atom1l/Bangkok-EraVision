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
Your task is to modify the uploaded photo. Adherence to the following rules is absolute and mandatory. **Your PRIMARY GOAL is to preserve the existing layout and follow the explicit structural rules below.**

**Rule 1: The Monument (Preservation)**
- The monument's form, placement, and perspective MUST be preserved.
- ALL existing surface details (carvings, lion-heads) MUST be retained and enhanced, NOT replaced.
- Remove the brush or small plants at the monument's base.
- For subtle texture inspiration ONLY (do NOT alter structure based on this), consider visual cues like: '{landmark_details_from_dataset}'.

**Rule 2: Surrounding Area - Transformation Based on Original Composition**
- Transform EXISTING elements following the strict rules below.

- **Area A: The Urban Layout (Conditional Rules)**
  - (Keep all conditional layout rules as they are - Central Median, Lampposts, Road Surface)

- **Area B: Building Architecture by Zone (Strict Structural Rules)**
  - **Architectural Uniformity Rule:** Zone 2 buildings MUST be a continuous, uniform block.
  - **Facade Rhythm Rule:** Zone 2 facade is NOT flat, must have rhythmic protruding/recessed bays.
  - **Zone 1 (Circle/Corners):** Buildings MUST be smaller, 2-3 story commercial.
  - **Zone 2 (Main Avenue):** Buildings MUST be the large, blocky, reddish-orange Modernist style described above.
  - **Zone 3 (Deep Background):** MUST transition to a dense, low-rise area.

- **General Details for All Buildings:**
  - Buildings MUST look aged and imperfect.
  - Windows MUST be simple, sharp, 90-degree rectangles.
  - The street MUST be populated with 1960s vehicles.
  - **For subtle texture and weathering ONLY (do NOT change structure/color based on this),** consider visual cues like: '{surrounding_details_from_dataset}'.

**Atmosphere:**
- (Keep atmosphere rules as they are)
"""
# (Apply similar "subtle texture inspiration ONLY" wording to other prompts)
,

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

    # ====== ศาลาเฉลิมกรุง ======
    "Sala Chalermkrung Royal Theatre": """
Your task is to modify the uploaded photo. Adherence to the following rules is absolute and mandatory.

**Rule 1: Core Architecture (Preservation)**
- The fundamental architectural form, perspective, and placement of the Sala Chalermkrung theatre building MUST be strictly preserved as it appears in the uploaded photo.
- Key unchangeable features include its modernist blocky structure, the distinct corner entrance, the rows of upper-floor windows, and the circular decorative medallions below the roofline.
- For fine details like the texture of the concrete and the design of the rooftop neon sign structure, draw inspiration from these visual cues: '{landmark_details_from_dataset}'.

**Rule 2: Theatrical Facade Decoration (Transformation)**
- This is your primary transformation task. The building's facade MUST be adorned as a classic 1960s Thai movie palace.
- **Hand-Painted Billboards:** Large sections of the upper facade MUST be covered with massive, hand-painted movie billboards. These billboards must depict scenes and actors characteristic of 1960s Thai cinema, featuring rich colors and a painterly style.
- **Character Cut-outs:** It is MANDATORY to add at least one large, painted wooden cut-out figure of a movie star. This figure should be placed prominently, either on the marquee above the entrance or standing at street level in front of the theatre.
- **Marquee & Banners:** The marquee above the entrance must be decorated with smaller, hand-lettered banners and signs announcing the current film's title and stars in Thai script.

**Rule 3: The Surrounding Street Scene (Contextual Transformation)**
- The street and sidewalks must be transformed to reflect a 1960s Bangkok setting.
- **Vehicles:** The street MUST be populated with a realistic mix of 1960s vehicles, such as Datsun Bluebird-style sedans, older American cars, and three-wheeled pedicabs (samlors). It is FORBIDDEN to show any modern cars, vans, or motorcycles.
- **Pedestrians & Street Furniture:** The sidewalks should feature pedestrians in 1960s attire. Simple, low metal crowd control barriers may be present in front of the theatre. The road surface must be aged asphalt.
- For the specific style of cars, clothing, and street textures, use these visual cues: '{surrounding_details_from_dataset}'.

**Atmosphere:** The final image must **match the ambient lighting, weather, and time of day of the original uploaded photo.** Apply a vintage aesthetic that authentically replicates the look of 1960s color film, including its unique color science, saturation, and natural grain. Avoid artificial sepia filters or overly aggressive aging effects.
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



# ================== No dataset required version ================== #
# ================== No dataset required version ================== #

# # reference_prompt_builder.py
# import os

# # (ลบ imports ที่ไม่จำเป็นออก: PIL, Transformers, BLIP, CLIP, compute_similarity)

# DATASET_ROOT = "dataset" # เก็บไว้เผื่อใช้อ้างอิง แต่โค้ดนี้ไม่ได้ใช้แล้ว

# # Mapping ของชื่อสถานที่ → โฟลเดอร์ใน dataset (ใช้แค่ตรวจสอบว่ามีสถานที่นี้จริงหรือไม่)
# PLACE_NAME_TO_FOLDER = {
#     "Ratchadamnoen Avenue – Democracy Monument": "Ratchadamnoen Avenue – Democracy Monument",
#     "Sala Chalermkrung Royal Theatre": "Sala Chalermkrung Royal Theatre",
#     "Giant Swing – Wat Suthat": "Giant Swing - Wat Suthat",
#     "Phra Sumen Fort – Santichaiprakan Park": "Phra Sumen Fort – Santichaiprakarn Park",
#     "National Museum Bangkok": "Phra Nakhon National Museum",
#     "Yaowarat (Chinatown)": "Yaowarat (Chinatown)",
#     "Sanam Luang (Royal Field)": "Sanam Luang (Royal_Field)"
# }

# # --- คลัง PROMPT (Hard-coded Prompts) ---
# LOCATION_SPECIFIC_PROMPTS = {
#     # ====== ราชดำเนินกลาง-อนุสาวรีย์ประชาธิปไตย (ฉบับ Hard-coded) ======
#     "Ratchadamnoen Avenue – Democracy Monument": """
# Your task is to modify the uploaded photo. Adherence to the following rules is absolute and mandatory.

# **Rule 1: The Monument (Preservation)**
# - The monument's form, placement, and perspective MUST be preserved.
# - ALL existing surface details (carvings, lion-heads) MUST be retained and enhanced, NOT replaced.
# - Ensure the central turret displays **its historic red doors and the golden constitution sculpture on top.**

# **Rule 2: Surrounding Area - Transformation Based on Original Composition**
# - **Your PRIMARY GOAL is to preserve the existing layout of the uploaded photo.** The following rules are for transforming EXISTING elements ONLY if they do not contradict the original composition.

# - **Area A: The Urban Layout (ผังเมืองและถนน - Conditional Rules)**
#   - **Central Median Strip (เกาะกลางถนน):**
#     - **IF a median strip is visible in the original photo,** it MUST be transformed into a chain of distinct, separate rounded rectangular islands with wide gaps (rendered as road surface) in between.
#     - **IF NO median strip is visible in the original photo, IT IS FORBIDDEN to add one.**
#   - **Lamppost Placement (เสาไฟ):**
#     - **Presence Rule: Add lampposts ONLY IF they are already present in the original photo.** If the original photo has no lampposts, DO NOT add them.
#     - **Placement Rule:** IF lampposts are present, Kinnara lampposts are ONLY for central median islands (arranged in facing pairs); simpler 1960s lampposts are ONLY for sidewalks.
#     - **Exclusion Zone:** It is FORBIDDEN to place any lampposts on the monument's immediate roundabout or its steps.
#   - **Road Surface Rule:** The road surface MUST be plain, aged asphalt. It is FORBIDDEN to add modern lane dividers.

# - **Area B: Building Architecture by Zone (สถาปัตยกรรมตามโซน)**
#   - **Architectural Uniformity Rule:** Buildings lining the main avenue (Zone 2) must be a continuous, uniform block with consistent height.
#   - **Facade Rhythm Rule:** The facade is NOT a flat wall. It must feature a distinct rhythmic pattern of protruding vertical sections alternating with recessed bays.
#   - **Zone 1 (Immediate Circle & Dinso Rd. Corners):** Buildings directly at the circle must be smaller, 2-3 story commercial buildings.
#   - **Zone 2 (Main Avenue - Foreground):** The large-scale mid-20th century Bangkok Modernist buildings (blocky, **muted deep reddish-orange color**) must adhere to the uniformity and rhythm rules above.
#   - **Zone 3 (Deep Background):** Visible behind the main avenue buildings, the cityscape must transition into a dense, low-rise area of smaller, mixed-style shophouses.

# - **General Details for All Buildings:**
#   - All buildings must look aged and imperfect (weathering, stains), not like a clean 3D render.
#   - Windows on ALL buildings MUST be simple, rectangular insets with sharp, 90-degree corners.
#   - The street MUST be populated with a realistic mix of 1960s vehicles.

# **Atmosphere:**
# - The final image MUST **match the ambient lighting, weather, and time of day of the original photo.**
# - Apply a vintage aesthetic that replicates 1960s color film (color science, saturation, grain). Avoid artificial sepia filters.
# """,

#     # ====== ศาลาเฉลิมกรุง (ตัวอย่าง - คุณต้องเขียน Prompt นี้เองโดยละเอียด) ======
#     "Sala Chalermkrung Royal Theatre": """
# Your task is to modify the uploaded photo. Adherence to the following rules is absolute and mandatory.

# **Rule 1: Core Architecture (Preservation)**
# - The fundamental architectural form, perspective, and placement of the Sala Chalermkrung theatre building MUST be strictly preserved.
# - Ensure key features like the modernist structure, corner entrance, window rows, and roofline medallions are retained.

# **Rule 2: Theatrical Facade Decoration (Transformation)**
# - The building's facade MUST be adorned as a classic 1960s Thai movie palace.
# - **Hand-Painted Billboards:** Large sections MUST be covered with massive, hand-painted 1960s Thai movie billboards.
# - **Character Cut-outs:** It is MANDATORY to add at least one large, painted wooden cut-out figure of a movie star.
# - **Marquee & Banners:** The marquee must have hand-lettered banners in Thai script.

# **Rule 3: The Surrounding Street Scene (Contextual Transformation)**
# - The street MUST reflect a 1960s Bangkok setting.
# - **Vehicles:** Populate with 1960s vehicles (Datsuns, older American cars, samlors). FORBIDDEN modern vehicles.
# - **Pedestrians & Details:** Sidewalks should feature pedestrians in 1960s attire. Road surface must be aged asphalt.

# **Atmosphere:**
# - The final image MUST **match the ambient lighting, weather, and time of day of the original photo.**
# - Apply a vintage aesthetic that replicates 1960s color film. Avoid artificial sepia filters.
# """,

#     # --- (เพิ่ม Prompt สำหรับสถานที่อื่นๆ ที่นี่ โดยใช้วิธี Hard-coding) ---
#     # "Yaowarat (Chinatown)": """ ... """
#     # "Giant Swing – Wat Suthat": """ ... """
#     # ... etc ...

# }

# # --- (ลบฟังก์ชัน describe_specific_images ทิ้ง) ---

# # --- ตัวสร้าง Prompt หลัก (ฉบับแก้ไข - ไม่ใช้ Dataset) ---
# def build_prompt(place_name, user_image_path=None): # user_image_path ไม่ได้ใช้แล้ว แต่เก็บไว้เผื่ออนาคต
#     """
#     สร้าง prompt โดยเลือกจากคลัง (Hard-coded Prompts)
#     """
#     # 1. เลือก Prompt หลักจากคลัง
#     base_prompt = LOCATION_SPECIFIC_PROMPTS.get(place_name)
#     if not base_prompt:
#         # ตรวจสอบว่ามีชื่อสถานที่ใน mapping แต่ไม่มีใน prompt หรือไม่
#         if place_name not in PLACE_NAME_TO_FOLDER:
#              raise ValueError(f"Unknown location selected: {place_name}")
        
#         # ถ้ามีสถานที่ แต่ยังไม่ได้เขียน Prompt
#         raise ValueError(f"No specific prompt template written for {place_name}. Please add it to LOCATION_SPECIFIC_PROMPTS.")

#     # 2. คืนค่า Prompt โดยตรง
#     final_prompt = base_prompt
    
#     return final_prompt
