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

# ที่อยู่โฟลเดอร์สำหรับ Method (Reference-based)
REFERENCE_DIR = "reference_images"

# --- 1. ฟังก์ชันสำหรับ Method (Reference-based) ---
# (เหมือนในโค้ดก่อนหน้า)

def preprocess_reference_images(reference_dir, places, processor, model, device):
    """
    [MODIFIED!] โหลด, ประมวลผล และคำนวณ "ค่าเฉลี่ย" ของ Vector ภาพอ้างอิงล่วงหน้า
    """
    print("กำลังประมวลผลภาพอ้างอิง (คำนวณค่าเฉลี่ย)...")
    all_mean_features = []
    all_labels_for_mean = [] # <-- จะเก็บแค่ 9 สถานที่

    with torch.no_grad():
        for place in places: # places คือ ALL_PLACES (9 สถานที่)
            place_dir = os.path.join(reference_dir, place)
            if not os.path.isdir(place_dir):
                print(f"Warning: ไม่พบโฟลเดอร์ {place_dir}")
                continue

            place_features_list = []
            for image_name in os.listdir(place_dir):
                image_path = os.path.join(place_dir, image_name)
                try:
                    image = Image.open(image_path).convert("RGB")
                    inputs = processor(images=image, return_tensors="pt").to(device)
                    image_features = model.get_image_features(**inputs)
                    place_features_list.append(image_features)
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")
            
            if not place_features_list:
                print(f"Warning: ไม่พบภาพใน {place_dir}")
                continue

            # --- [CRITICAL CHANGE] ---
            # คำนวณค่าเฉลี่ยของ Vector ทั้งหมดในหมวดหมู่นี้
            mean_feature = torch.mean(torch.cat(place_features_list, dim=0), dim=0, keepdim=True)
            all_mean_features.append(mean_feature)
            all_labels_for_mean.append(place)
            # --- [END CHANGE] ---

    if not all_mean_features:
        raise ValueError("ไม่พบภาพอ้างอิงเลย! กรุณาตรวจสอบโฟลเดอร์ reference_images")

    reference_features_tensor = torch.cat(all_mean_features, dim=0) # <-- Tensor นี้จะมีแค่ 9 แถว
    print(f"ประมวลผลค่าเฉลี่ยเสร็จสิ้น: พบ {len(all_labels_for_mean)} สถานที่")
    return reference_features_tensor, all_labels_for_mean # <-- คืนค่าเฉลี่ยและชื่อสถานที่ (9 รายการ)

def classify_by_reference(image_path, reference_mean_features, reference_mean_labels, processor, model, device):
    """
    [MODIFIED!] จำแนกภาพโดยเทียบกับ "ค่าเฉลี่ย" ของคลังภาพอ้างอิง
    """
    try:
        user_image = Image.open(image_path).convert("RGB")
        with torch.no_grad():
            inputs = processor(images=user_image, return_tensors="pt").to(device)
            user_feature = model.get_image_features(**inputs) # Vector ของภาพผู้ใช้ (1 แถว)

        # --- [CRITICAL CHANGE] ---
        # เปรียบเทียบ Vector ผู้ใช้ (1 แถว) กับ Vector ค่าเฉลี่ย (9 แถว)
        similarities = F.cosine_similarity(user_feature, reference_mean_features, dim=1)
        best_score, best_index = torch.max(similarities, dim=0)
        
        predicted_place = reference_mean_labels[best_index.item()]
        confidence_score = best_score.item()
        # --- [END CHANGE] ---
        
        return predicted_place, confidence_score

    except Exception as e:
        print(f"Error in classify_by_reference: {e}")
        return None, 0.0

# --- (ลบ Hybrid Classifier และ Text Classifier ที่ไม่ได้ใช้ออก) ---

# --- 5. ส่วน 실행 (Execution) ---
if __name__ == "__main__":
    
    # 5.1 โหลดข้อมูลอ้างอิง (ทำครั้งเดียว)
    reference_features, reference_labels = preprocess_reference_images(
        REFERENCE_DIR, ALL_PLACES, processor, model, device
    )

    # 5.2 ทดสอบ
    USER_IMAGE_PATH_2 = 'PATH_TO_YOUR_YAOWARAT_IMAGE.jpg'
    if os.path.exists(USER_IMAGE_PATH_2):
        place, score = classify_by_reference(
            USER_IMAGE_PATH_2, 
            reference_features, 
            reference_labels, 
            processor, 
            model, 
            device
        )
        print(f"--> ผลลัพธ์สุดท้าย: {place} (Score: {score:.4f})")