# classifier.py
import os
from google.cloud import vision

# --- 1. Setup Credentials ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CREDENTIALS_PATH = os.path.join(CURRENT_DIR, "credentials.json")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = CREDENTIALS_PATH

client = None
try:
    if os.path.exists(CREDENTIALS_PATH):
        client = vision.ImageAnnotatorClient()
        print("✅ Google Vision Client Ready!")
    else:
        print(f"❌ CRITICAL: Credentials file not found at {CREDENTIALS_PATH}")
except Exception as e:
    print(f"❌ Error initializing client: {e}")

# --- 2. รายชื่อสถานที่ (สำหรับ app.py) ---
PLACE_LABELS = [
    "Ratchadamnoen Avenue – Democracy Monument",
    "Sala Chalermkrung Royal Theatre",
    "Giant Swing – Wat Suthat",
    "Khao San Road",
    "Phra Sumen Fort – Santichaiprakan Park",
    "National Museum Bangkok",
    "Yaowarat (Chinatown)",
    "Sanam Luang (Royal Field)"
]

# --- 3. Keyword Mapping ---
KEYWORD_MAPPING = {
    "Ratchadamnoen Avenue – Democracy Monument": ["Democracy Monument", "Ratchadamnoen", "Anusawari"],
    "Sala Chalermkrung Royal Theatre": ["Sala Chalermkrung", "Royal Theatre", "Chalermkrung"],
    "Giant Swing – Wat Suthat": ["Giant Swing", "Sao Ching Cha", "Wat Suthat", "Suthat"],
    "Khao San Road": ["Khao San", "Khaosan", "Buddy Lodge", "Thanon Khao San"],
    "Phra Sumen Fort – Santichaiprakan Park": ["Phra Sumen", "Santichaiprakan", "Phra Arthit"],
    "National Museum Bangkok": ["National Museum", "Wang Na", "Bangkok National Museum"],
    "Yaowarat (Chinatown)": ["Yaowarat", "Chinatown", "China Town", "Canton House"],
    "Sanam Luang (Royal Field)": ["Sanam Luang", "Royal Field", "Pramane Ground"]
}

# --- 4. Bad Composition Rules ---

# 4.1 Global Rules: กฎเหล็กที่ห้ามทุกกรณี (Technical & Composition)
GLOBAL_BAD_LABELS = [
    # กลุ่มพื้นผิว/ซูม/มุมมองไม่ดี
    "Pattern", "Texture", "Material", "Wall", "Floor", "Brick", 
    "Close-up", "Macro photography", "Symmetry", "Circle", "Rectangular",
    
    # กลุ่มสิ่งที่ไม่ใช่รูปถ่ายสถานที่จริง
    "Text", "Font", "Screenshot", "Document", "Poster", "Drawing", "Map", "Plot", "Sketch",
    
    # กลุ่มคน (Selfie/Portrait)
    "Selfie", "Face", "Skin", "Eyewear", "Cool"
]

# 4.2 Context-Aware Rules: กฎเฉพาะสถานที่ (ห้ามมีสิ่งนี้ในที่นี้)
PER_PLACE_BAD_LABELS = {
    "Ratchadamnoen Avenue – Democracy Monument": [
        # 1. กลุ่มชุมนุมและฝูงชน (Crowd Control)
        "Protest", "Demonstration", "Crowd", "Riot", "Banner", "Flag",
        
        # 2. ยานพาหนะ (Vehicles) 
        "Traffic jam", "Bus", "Truck", "Lorry", "Van", "Vehicle", 
        "Motorcycle", "Scooter", "Bicycle",
        
        # 3. สิ่งก่อสร้างและสิ่งรกตา (Clutter & Modern Objects)
        "Scaffolding", "Construction", "Signage", "Billboard", "Advertising",
        "Barrier", "Traffic cone", "CCTV", "Overpass", "Bridge",
        
        # 4. ธรรมชาติ (Nature Obstruction)
        "Flower", "Tree", "Plant",
        
        # 5. มุมกล้องและแสง (Composition & Lighting)
        # เอาไว้กันรูปถ่ายเจาะเฉพาะจุด
        "Close-up", "Detail", "Macro", "Sculpture", "Statue", "Relief", "Wing", 
        # เอาไว้กัน Generate ออกมาเพี้ยนเนื่องจาก Dataset มีแต่ตอนกลางวัน
        "Night", "Evening", "Darkness", "Sunset"

        # 6. มุมกล้องสูงและผังเมือง (High Angle & Urban Layout)
        "Aerial photography", "Aerial view", "Bird's-eye view", "High-angle shot", "Drone",
        "Urban design", "Metropolitan area", "Cityscape", "Skyline",
    ],

    #==========================================================================================#
    #==========================================================================================#
    #==========================================================================================#

    "Sala Chalermkrung Royal Theatre": [
        # 1. ภายใน (อันนี้ต้องกันไว้เหมือนเดิม)
        "Interior", "Inside", "Audience", "Seat", "Movie theater", 
        "Stage", "Curtain", "Lobby", "Ceiling",
        
        # 2. Modern & Tech (จอ LED หน้าตึก)
        "Screen", "Monitor", "Display", "Digital screen", "LED display", "Television",
        
        # 3. Crowd & People (คนเยอะเกินไปบังตึก)
        "Crowd", "Selfie", "Mobile phone",
        
        # 4. Close-up (ถ่ายเจาะ)
        "Close-up", "Detail", "Macro",

        # --- 5. Side View Killers (โฟกัสแค่ "ทรงตึก" เท่านั้น) ---
        # ❌ ลบ Sky, Tree, Road, Traffic light ออกให้หมด! (เพราะมุมหน้าก็มี)
        
        # ✅ ตัวแยกมุม: "หน้าต่าง" และ "มุมตึก"
        "Window", "Windows", # ด้านข้างมีหน้าต่าง ด้านหน้ามีแต่ป้าย
        "Corner", # มุมหัวมุม
        "Apartment", "Office building", "Condominium", # AI มักมองหน้าต่างด้านข้างว่าเป็นอพาร์ทเมนต์
        "Residential area", # บางทีด้านข้างดูเหมือนตึกที่พักอาศัย
        
        # ✅ กันมุมเสย (ที่ทำให้ตัวหนังสือเบี้ยว)
        "Low angle shot", "Worm's-eye view"
    ],

    "Giant Swing – Wat Suthat": [
        # 1. Modern Cityscape
        "Skyscraper", "Cityscape", "Modern building", "Tower", "Office building",
        "Hotel", "Apartment", "Condominium",
        
        # 2. Event & Clutter
        "Tent", "Canopy", "Stage", "Performance", "Concert",
        "Fence", "Barrier", "Traffic cone", "Plastic chair",
        
        # 3. Traffic & Vehicles
        "Traffic jam", "Bus", "Truck", "Van", "Pickup truck",
        "Vehicle", "Car",
        
        # 4. Activities
        "Sport", "Basketball", "Playground", "Exercise", "Aerobics",
        "Market", "Street food", "Vendor",
        
        # 5. Composition Flaws
        "Close-up", "Detail", "Selfie", # ห้ามเซลฟี่หน้าเสา
        "Pigeon", "Bird", # นกเยอะเกินไปจะดูสกปรก
        "Power line", "Cable", "Wire", # สายไฟยุคนี้รกกว่ายุคก่อน
        
        # 6. Tourists
        "Tourist", "Backpack", "Group", "Tour guide"

        # 7. Angles & Views
        "Aerial photography", "Aerial view", "Bird's-eye view", "High-angle shot", "Drone"
    ],

    "Khao San Road": [
        # 1. Framing & Angle
        "Close-up", "Macro", "Detail", 
        "Selfie", "Face", "Portrait", 
        "Ground", "Floor", "Asphalt", 
        "Sky",

        # 2. Specific Objects that Ruin Structure 
        "Dish", "Bowl", "Food", "Cuisine", "Meal", 
        "Drink", "Beverage", "Glass", "Bottle",
        "Table", 
        "Umbrella", 

        # 3. Overwhelming Obstructions 
        "Truck", "Van", "Bus", 
        "Wall", "Fence", "Barrier" 
    ],

    #======================= ยังไม่ชัวร์ =======================#
    "Phra Sumen Fort – Santichaiprakan Park": [
        # 1. The "Park" Elements 
        # "Pavilion", "Gazebo", "Thai pavilion", "Sala", # พระที่นั่งสันติชัยปราการ (ไม่มีในยุค 60s)
        "Park", "Garden", "Lawn", "Manicured grass", # สนามหญ้าเรียบกริบ (ในรูปเก่าคือหญ้ารกๆ)
        "Bench", "Seating area", "Lamp post", # ม้านั่งและโคมไฟสวนสาธารณะ
        "Playground", "Exercise equipment", # เครื่องเล่นเด็ก/เครื่องออกกำลังกาย
        
        # 2. Modern Anachronisms (สิ่งที่ผิดยุคอย่างรุนแรง)
        "Bridge", "Suspension bridge", "Cable-stayed bridge", # สะพานพระราม 8 (ห้ามติดมาเด็ดขาด)
        "Skyscraper", "Condominium", "Modern building", # ตึกสูงฝั่งธนฯ
        
        # 3. Modern Activities (กิจกรรมคนกรุงยุคใหม่)
        "Aerobics", "Yoga", "Exercise", "Jogging", "Running", # คนมาเต้นแอโรบิก
        "Picnic", "Mat", "Camping", # คนปูเสื่อนั่งเล่น
        "Skateboard", "Rollerblades", "Bicycle", "Cycling", # เด็กสเก็ต
        "Music", "Guitar", "Band", "Performance", # ดนตรีในสวน
        
        # 4. River & View (มุมที่ไม่ได้โฟกัสตัวป้อม)
        "River", "Water", "Boat", "Ship", # ถ้าถ่ายแต่วิวแม่น้ำ ไม่เห็นป้อม (Dataset เน้นกำแพงป้อม)
        "Pier", "Dock", # ท่าเรือพระอาทิตย์สมัยใหม่
        
        # 5. Composition
        "Tree", "Plant", "Foliage", # ต้นลำพูต้นใหญ่บังป้อมมิด (ควรเห็นกำแพงป้อมชัดๆ)
        "Close-up", "Texture", "Brick", # ถ่ายเจาะกำแพง
        "Selfie", "Portrait"
    ],

    "National Museum Bangkok": [
        # 1. Museum Exhibits & Interior 
        "Glass", "Display case", "Artifact", "Interior", "Showcase",
        "Statue", "Sculpture", "Buddha", "Amulet", "Ceramic", "Weapon", "Painting",
        "Exhibit", "Gallery", "Room", "Ceiling", "Light fixture",

        # 2. Mass Tourism & Modern Entrance
        "Bus", "Tour bus", "Van", "Ticket", "Ticket counter",
        "Crowd", "Tourist", "Group", "Tour guide", "Student", "Uniform",
        "Camera", "Tripod", "Selfie", "Backpack",
        
        # 3. Modern Signs & Event
        "Banner", "Poster", "Signage", "Billboard", "Tent", "Marquee",
        
        # 4. Composition
        "Close-up", "Detail", "Macro", 
        "Garden", "Tree", "Plant",
    ],

    "Yaowarat (Chinatown)": [
        # 1. Visual Blockers: Large Vehicles
        "Bus", "Double-decker bus", "Tour bus",
        "Truck", "Lorry", "Container", "Van", "Pickup truck",
        "Traffic jam",
        
        # 2. Visual Blockers: Street Furniture
        "Umbrella", "Beach umbrella", "Canopy", "Tent",
        "Barrier", "Fence", "Construction",
        
        # 3. Perspective Blockers: Crowd
        "Crowd", "Procession", "Demonstration", "Group", 
        "Tourist guide", "Flag", "Street food stall"
        
        # 4. Wrong Subject Focus
        "Dish", "Bowl", "Food", "Cuisine", "Meal",
        "Selfie", "Face", "Portrait",
        "Close-up", "Macro", "Detail",
        "Gold", "Jewellery"
        
        # --- สิ่งที่ปลดล็อก (ยอมให้มีได้) ---
        # Signage, LED, Neon, Monitor -> ปล่อยผ่าน (ให้ AI แปลง Style เอา)
        # Tuk tuk, Taxi, Car -> ปล่อยผ่าน (ให้ AI แปลงเป็นรถเก่า)
        # Street food stall (ตัวรถเข็นเตี้ยๆ) -> ปล่อยผ่าน (ถ้าไม่กางร่มบังตึก)
    ],

    "Sanam Luang (Royal Field)": [
        # 1. The "Wall" of Vehicles (รถบัสบังวิว - เหมือนเดิม)
        "Bus", "Tour bus", "Double-decker bus", "Coach",
        "Truck", "Van", "Traffic jam", 
        
        # 2. Modern Event Structures (เต็นท์งานพิธี - เหมือนเดิม)
        "Tent", "Canopy", "Marquee", "White tent",
        "Stage", "Truss", "Scaffolding", "Loudspeaker",
        "Fence", "Barrier", "Metal fence", "Barricade",
        
        # 3. Close-up People & Activities (ด่านใหม่! แก้มือตรงนี้)
        "Sitting", "Lying down", "Picnic", # คนนั่ง/นอนเล่นบังวิว
        "Legs", "Foot", "Shoe", # ถ่ายเจาะขา/รองเท้าตัวเอง
        "Student", "Uniform", "Group", # กลุ่มนักศึกษา/คนมานั่งจับกลุ่มคุยกัน
        "Selfie", "Portrait", # ถ่ายหน้าคนชัดๆ
        
        # 4. Clutter & Cleanliness (เหมือนเดิม)
        "Garbage", "Waste", "Trash", "Plastic bag",
        "Market", "Stall", "Vendor", "Flea market",
        "Homeless", "Sleeping", "Tree", "Bench", "Lamp post", "Chair"
    ]
}

# --- 4.3 Required Landmarks (กฎเหล็ก: ต้องเห็น Landmark หรือบริบทเหล่านี้) ---
REQUIRED_LANDMARKS = {
    "Giant Swing – Wat Suthat": [
        # English
        "Giant Swing", "Sao Ching Cha",
        "Wat Suthat", "Suthat Temple",

        # Thai
        "เสาชิงช้า",
        "วัดสุทัศน์", "วัดสุทัศนเทพวราราม"
    ],

    "Sanam Luang (Royal Field)": [
        # English
        "Grand Palace", "Royal Palace", 
        "Wat Phra Kaew", "Emerald Buddha", "Temple of the Emerald Buddha",
        "Sanam Luang", "Phra Mane Ground", 
        
        # Thai
        "พระบรมมหาราชวัง", "พระราชวัง",
        "วัดพระแก้ว", "วัดพระศรีรัตนศาสดาราม",
        "สนามหลวง", "ท้องสนามหลวง"
    ]
}

# --- 4.4 Score Thresholds (เกณฑ์คะแนนขั้นต่ำแยกรายที่) ---
# ค่า Default คือ 0.50
SCORE_THRESHOLDS = {
    "Sala Chalermkrung Royal Theatre": 0.90,
    "Khao San Road": 0.70,
    "Phra Sumen Fort – Santichaiprakan Park": 0.65,
    "Yaowarat (Chinatown)": 0.70,
    "Sanam Luang (Royal Field)": 0.65,
}

def calculate_area(vertices):
    """คำนวณพื้นที่ของ Bounding Box (ค่า Normalized 0.0 - 1.0)"""
    x_coords = [v.x for v in vertices]
    y_coords = [v.y for v in vertices]
    width = max(x_coords) - min(x_coords)
    height = max(y_coords) - min(y_coords)
    return width * height

def classify_image(image_path):
    if client is None:
        return "Error (Setup Issue)", 0.0, False

    try:
        with open(image_path, "rb") as image_file:
            content = image_file.read()
        image = vision.Image(content=content)

        features = [
            vision.Feature(type_=vision.Feature.Type.LANDMARK_DETECTION),
            vision.Feature(type_=vision.Feature.Type.WEB_DETECTION),
            vision.Feature(type_=vision.Feature.Type.LABEL_DETECTION),
            vision.Feature(type_=vision.Feature.Type.OBJECT_LOCALIZATION),
        ]
        request = vision.AnnotateImageRequest(image=image, features=features)
        response = client.batch_annotate_images(requests=[request]).responses[0]

        print("\n--- 🎬 Image Analysis Director Mode ---")

        # =========================================================
        # STEP 1: Global Quality Check
        # =========================================================
        found_global_bad = []
        for label in response.label_annotations:
            if label.description in GLOBAL_BAD_LABELS and label.score > 0.75:
                found_global_bad.append(label.description)
        
        if found_global_bad:
            reason = ", ".join(found_global_bad)
            print(f"❌ REJECTED: Global Bad Composition ({reason})")
            return f"Rejected (Bad Composition: {reason})", 0.0, False

        total_person_area = 0.0
        if response.localized_object_annotations:
            for obj in response.localized_object_annotations:
                if obj.name == "Person":
                    area = calculate_area(obj.bounding_poly.normalized_vertices)
                    total_person_area += area
            
            if total_person_area > 0.40:
                print(f"❌ REJECTED: Too much person area ({total_person_area:.0%})")
                return f"Rejected (Subject is Person: {total_person_area:.0%})", 0.0, False

        # =========================================================
        # STEP 2: Place Identification & Data Gathering
        # =========================================================
        detected_keywords = []
        all_found_context = set() 

        # A. Landmark
        if response.landmark_annotations:
            for landmark in response.landmark_annotations:
                detected_keywords.append((landmark.description, landmark.score + 0.3))
                all_found_context.add(landmark.description.lower())

        # B. Web Entities
        if response.web_detection and response.web_detection.web_entities:
            for entity in response.web_detection.web_entities[:10]:
                detected_keywords.append((entity.description, entity.score))
                if entity.description:
                    all_found_context.add(entity.description.lower())

        # Matching Logic
        best_match_place = None
        best_match_score = 0.0

        for place_key, target_keywords in KEYWORD_MAPPING.items():
            for target in target_keywords:
                for detected_word, confidence in detected_keywords:
                    if not detected_word: continue
                    if target.lower() in detected_word.lower():
                        if confidence > best_match_score:
                            best_match_score = confidence
                            best_match_place = place_key

        if best_match_score < 0.5:
             best_match_place = None

        # 🛡️ Threshold Check
        required_score = SCORE_THRESHOLDS.get(best_match_place, 0.50)
        if best_match_score < required_score:
             if best_match_place:
                 print(f"❌ REJECTED: {best_match_place} found but score {best_match_score:.2f} < {required_score}")
             best_match_place = None

        # =========================================================
        # STEP 3: Context-Aware Filtering
        # =========================================================
        if best_match_place and best_match_place in PER_PLACE_BAD_LABELS:
            specific_bad_list = PER_PLACE_BAD_LABELS[best_match_place]
            found_specific_bad = []
            
            # 3.1 Check Labels (เช็คป้ายกำกับตามปกติกับทุกที่)
            for label in response.label_annotations:
                if label.description in specific_bad_list and label.score > 0.65:
                    found_specific_bad.append(f"Label: {label.description}")

            # 3.2 Check Objects (⭐ เงื่อนไขพิเศษ: เช็ควัตถุเฉพาะศาลาเฉลิมกรุง ⭐)
            if best_match_place == "Sala Chalermkrung Royal Theatre":
                if response.localized_object_annotations:
                    for obj in response.localized_object_annotations:
                        # เช็คว่ามีวัตถุใน Bad list หรือไม่ (เช่น Traffic light, Car)
                        if obj.name in specific_bad_list and obj.score > 0.5: 
                            found_specific_bad.append(f"Object: {obj.name}")

            if found_specific_bad:
                reason = ", ".join(found_specific_bad)
                print(f"❌ REJECTED: Context Mismatch for {best_match_place} ({reason})")
                return f"Rejected (Invalid content for this place: {reason})", 0.0, False
        
        # =========================================================
        # STEP 3.5: Required Landmark Check
        # =========================================================
        if best_match_place and best_match_place in REQUIRED_LANDMARKS:
            required_list = REQUIRED_LANDMARKS[best_match_place]
            landmark_found = False
            
            print(f"   🏛️ Special Requirement Check for {best_match_place}")
            for required_lm in required_list:
                if any(required_lm.lower() in found_ctx for found_ctx in all_found_context):
                    landmark_found = True
                    print(f"      ✅ Requirement Met: Found '{required_lm}' in context.")
                    break
            
            if not landmark_found:
                print(f"❌ REJECTED: {best_match_place} identified, but visual confirmation missing.")
                print(f"      (System saw: {list(all_found_context)[:5]}...)")
                return f"Rejected (Missing visual landmark: {best_match_place})", 0.0, False

        # =========================================================
        # STEP 4: Final Result
        # =========================================================
        if best_match_place:
            final_score = min(best_match_score, 0.99)
            print(f"✅ ACCEPTED: {best_match_place} (Score: {final_score:.2f})")
            return best_match_place, final_score, True
        else:
            top_web = response.web_detection.web_entities[0].description if response.web_detection.web_entities else "Unknown"
            print(f"⚠️ UNKNOWN: Could not identify place (Top guess: {top_web})")
            return f"Other ({top_web})", 0.0, False

    except Exception as e:
        print(f"🔴 Error calling Google API: {e}")
        return "Error (API Issue)", 0.0, False