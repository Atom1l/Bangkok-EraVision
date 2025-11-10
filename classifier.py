# # classifier.py
# from PIL import Image
# from transformers import CLIPProcessor, CLIPModel
# import torch


# # โหลดโมเดล CLIP
# model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
# processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


# ALL_PLACES = [
#    "Ratchadamnoen Avenue – Democracy Monument",
#    "Sala Chalermkrung Royal Theatre",
#    "Giant Swing – Wat Suthat",
#    "Khao San Road",
#    "Phra Sumen Fort – Santichaiprakan Park",
#    "National Museum Bangkok",
#    "Yaowarat (Chinatown)",
#    "Sanam Luang (Royal Field)"
# ]



# def check_image_category(image_path, expected_place):
#     image = Image.open(image_path).convert("RGB")
#     texts = ALL_PLACES
#     inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)
#     with torch.no_grad():
#         outputs = model(**inputs)
#     probs = outputs.logits_per_image.softmax(dim=1)
#     return probs[0][texts.index(expected_place)].item()


import os
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch
import torch.nn.functional as F

# --- 1. การตั้งค่าหลัก (Shared) ---
print("กำลังโหลดโมเดล CLIP...")
MODEL_NAME = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(MODEL_NAME)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# ชื่อสถานที่หลัก (สำหรับโฟลเดอร์และผลลัพธ์)
ALL_PLACES = [
    "Ratchadamnoen Avenue – Democracy Monument",
    "Sala Chalermkrung Royal Theatre",
    "Giant Swing – Wat Suthat",
    "Khao San Road",
    "Phra Sumen Fort – Santichaiprakan Park",
    "National Museum Bangkok",
    "Yaowarat (Chinatown)",
    "Sanam Luang (Royal Field)",
    "Other"
]

# ข้อความสำหรับ Method 1 (Text-based)
TEXT_PROMPTS = [
    "a photo of Ratchadamnoen Avenue and the Democracy Monument, a structure with four large wings, in a traffic circle on Ratchadamnoen Avenue",
    "a photo of Sala Chalermkrung Royal Theatre building",
    "a photo of the giant red swing (Sao Ching Cha) in front of Wat Suthat temple in Bangkok",
    "a photo of Khao San Road, a party street with many bars, English signs, and tourists",
    "a photo of Phra Sumen Fort and Santichaiprakan Park by the river",
    "a photo of exhibits inside the Bangkok National Museum",
    "a photo of Yaowarat Chinatown, with bright red neon signs, Chinese characters, and gold shops",
    "a photo of the large grassy field of Sanam Luang near the Grand Palace"
]

# ที่อยู่โฟลเดอร์สำหรับ Method 2 (Reference-based)
REFERENCE_DIR = "reference_images"


# --- 2. ฟังก์ชันสำหรับ Method 1 (Text-based) ---
def classify_by_text(image_path, processor, model, device):
    """
    จำแนกภาพโดยเทียบกับ Text Prompts (เร็ว)
    คืนค่า (predicted_place, confidence_score)
    """
    try:
        image = Image.open(image_path).convert("RGB")
        
        # ใช้ TEXT_PROMPTS ที่เราปรับปรุงแล้ว
        inputs = processor(text=TEXT_PROMPTS, images=image, return_tensors="pt", padding=True).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        probs = outputs.logits_per_image.softmax(dim=1)[0] # ได้ [8]
        
        # หาค่าสูงสุด
        best_prob, best_index = torch.max(probs, dim=0)
        
        # คืน "ชื่อสถานที่" จาก ALL_PLACES (ไม่ใช่ Text Prompt)
        predicted_place = ALL_PLACES[best_index.item()]
        confidence_score = best_prob.item()
        
        return predicted_place, confidence_score
        
    except Exception as e:
        print(f"Error in classify_by_text: {e}")
        return None, 0.0

# --- 3. ฟังก์ชันสำหรับ Method 2 (Reference-based) ---
# (เหมือนในโค้ดก่อนหน้า)

def preprocess_reference_images(reference_dir, places, processor, model, device):
    """
    โหลดและประมวลผลภาพอ้างอิงทั้งหมดล่วงหน้า
    """
    print("กำลังประมวลผลภาพอ้างอิง (Pre-processing)...")
    all_features = []
    all_labels = []

    with torch.no_grad():
        for place in places:
            place_dir = os.path.join(reference_dir, place)
            if not os.path.isdir(place_dir):
                print(f"Warning: ไม่พบโฟลเดอร์ {place_dir}")
                continue

            for image_name in os.listdir(place_dir):
                image_path = os.path.join(place_dir, image_name)
                try:
                    image = Image.open(image_path).convert("RGB")
                    inputs = processor(images=image, return_tensors="pt").to(device)
                    image_features = model.get_image_features(**inputs)
                    all_features.append(image_features)
                    all_labels.append(place)
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")

    if not all_features:
        raise ValueError("ไม่พบภาพอ้างอิงเลย! กรุณาตรวจสอบโฟลเดอร์ reference_images")

    reference_features_tensor = torch.cat(all_features, dim=0)
    print(f"ประมวลผลภาพอ้างอิงเสร็จสิ้น: พบ {len(all_labels)} ภาพ")
    return reference_features_tensor, all_labels

def classify_by_reference(image_path, reference_features, reference_labels, processor, model, device):
    """
    จำแนกภาพโดยเทียบกับคลังภาพอ้างอิง (แม่นยำ)
    คืนค่า (predicted_place, confidence_score)
    """
    try:
        user_image = Image.open(image_path).convert("RGB")
        with torch.no_grad():
            inputs = processor(images=user_image, return_tensors="pt").to(device)
            user_feature = model.get_image_features(**inputs)

        similarities = F.cosine_similarity(user_feature, reference_features, dim=1)
        best_score, best_index = torch.max(similarities, dim=0)
        
        predicted_place = reference_labels[best_index.item()]
        confidence_score = best_score.item()
        
        return predicted_place, confidence_score

    except Exception as e:
        print(f"Error in classify_by_reference: {e}")
        return None, 0.0

# --- 4. ส่วนหลัก: Hybrid (Cascading) Classifier ---

def hybrid_classifier(image_path, text_threshold, ref_threshold, reference_features, reference_labels):
    """
    ฟังก์ชันหลักในการจำแนกแบบลูกผสม (Hybrid)
    """
    print(f"\n--- 1. เริ่มจำแนกภาพ: {image_path} ---")
    
    # --- ด่านที่ 1: ตรวจสอบด้วย Text (เร็ว) ---
    text_place, text_score = classify_by_text(image_path, processor, model, device)
    print(f"[Method 1: Text]   ทายว่า: {text_place} (ความมั่นใจ: {text_score:.4f})")
    
    if text_score >= text_threshold:
        print(f"-> ผลลัพธ์: ยอมรับ ✅ (ผ่านเกณฑ์ Text-Based > {text_threshold})")
        return text_place, text_score, "Method 1: Text-Based"
    
    # --- ด่านที่ 2: ตรวจสอบด้วย Reference (แม่นยำ) ---
    print(f"-> (คะแนน Text ต่ำ, ส่งต่อให้ Method 2 ตรวจสอบ...)")
    ref_place, ref_score = classify_by_reference(
        image_path, reference_features, reference_labels, processor, model, device
    )
    print(f"[Method 2: Reference] ทายว่า: {ref_place} (ความมั่นใจ: {ref_score:.4f})")

    if ref_score >= ref_threshold:
        print(f"-> ผลลัพธ์: ยอมรับ ✅ (ผ่านเกณฑ์ Reference-Based > {ref_threshold})")
        return ref_place, ref_score, "Method 2: Reference-Based"
    else:
        print(f"-> ผลลัพธ์: ไม่รู้จัก ❌ (ไม่ผ่านทั้ง 2 เกณฑ์)")
        return "Unknown", ref_score, "None"


# --- 5. ส่วน 실행 (Execution) ---
if __name__ == "__main__":
    
    # 5.1 โหลดข้อมูลอ้างอิง (ทำครั้งเดียว)
    reference_features, reference_labels = preprocess_reference_images(
        REFERENCE_DIR, ALL_PLACES, processor, model, device
    )

    # 5.2 ตั้งค่าเกณฑ์ความมั่นใจ (สำคัญมาก! ต้องลองจูนเอง)
    # เกณฑ์สำหรับ Text (ควรตั้งไว้สูงหน่อย เพราะมันไม่แม่นกับเคสยากๆ)
    TEXT_CONFIDENCE_THRESHOLD = 0.85 
    # เกณฑ์สำหรับ Reference (ตั้งต่ำกว่าได้ เพราะมันแม่นกว่า)
    REF_CONFIDENCE_THRESHOLD = 0.30 

    # --- ทดสอบ ---
    
    # ทดสอบภาพที่ 1 (สมมติว่าเป็นภาพเสาชิงช้าชัดๆ)
    USER_IMAGE_PATH_1 = 'PATH_TO_YOUR_GIANT_SWING_IMAGE.jpg' 
    if os.path.exists(USER_IMAGE_PATH_1):
        place, score, method = hybrid_classifier(
            USER_IMAGE_PATH_1, 
            TEXT_CONFIDENCE_THRESHOLD, 
            REF_CONFIDENCE_THRESHOLD, 
            reference_features, 
            reference_labels
        )
        print(f"--> ผลลัพธ์สุดท้าย: {place} (โดย {method})")

    # ทดสอบภาพที่ 2 (สมมติว่าเป็นภาพเยาวราชตอนกลางคืน)
    USER_IMAGE_PATH_2 = 'PATH_TO_YOUR_YAOWARAT_IMAGE.jpg'
    if os.path.exists(USER_IMAGE_PATH_2):
        place, score, method = hybrid_classifier(
            USER_IMAGE_PATH_2, 
            TEXT_CONFIDENCE_THRESHOLD, 
            REF_CONFIDENCE_THRESHOLD, 
            reference_features, 
            reference_labels
        )
        print(f"--> ผลลัพธ์สุดท้าย: {place} (โดย {method})")

    # ทดสอบภาพที่ 3 (สมมติว่าเป็นภาพแมว)
    USER_IMAGE_PATH_3 = 'PATH_TO_YOUR_CAT_IMAGE.jpg'
    if os.path.exists(USER_IMAGE_PATH_3):
        place, score, method = hybrid_classifier(
            USER_IMAGE_PATH_3, 
            TEXT_CONFIDENCE_THRESHOLD, 
            REF_CONFIDENCE_THRESHOLD, 
            reference_features, 
            reference_labels
        )
        print(f"--> ผลลัพธ์สุดท้าย: {place} (โดย {method})")