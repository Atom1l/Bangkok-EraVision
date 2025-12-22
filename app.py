# app.py
import os
import glob
from io import BytesIO
from flask import Flask, request, render_template, jsonify, url_for
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

# --- 1. Import ระบบ Classifier ---
from classifier import classify_image, PLACE_LABELS

# --- 2. Import ระบบ ML Transformer (ของเพื่อน) ---
try:
    from ml_transformer import EraVisionTransformer
    ML_ENABLED = True
except ImportError:
    print("⚠️ Warning: ไม่พบไฟล์ ml_transformer.py หรือ library ไม่ครบ")
    ML_ENABLED = False
    ml_transformer = None

# โหลด environment variables
load_dotenv()

# --- ตั้งค่า Flask ---
app = Flask(__name__)
# ใช้ static folder ในการเก็บไฟล์เพื่อให้เข้าถึงผ่าน URL ได้ง่าย
app.config['UPLOAD_FOLDER'] = "static/uploads"
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], "images_database"), exist_ok=True)

# --- Config Model Path ---
# แก้ไข Path ให้ตรงกับที่เก็บโมเดลจริง
ML_MODEL_PATH = "models/democracy_monument_1960s" 

# --- โหลด ML Model (Load Once) ---
if ML_ENABLED:
    try:
        print("⏳ Loading ML Model... (Please wait)")
        ml_transformer = EraVisionTransformer(ML_MODEL_PATH)
        print("✅ ML Model Ready!")
    except Exception as e:
        print(f"❌ Error loading ML Model: {e}")
        ML_ENABLED = False

# --- Helper Functions ---
def get_next_filename(folder, prefix="BangkokEra", ext=".png"):
    files = glob.glob(os.path.join(folder, f"{prefix}*{ext}"))
    if not files:
        return os.path.join(folder, f"{prefix}001{ext}")
    numbers = []
    for f in files:
        try:
            num_part = os.path.splitext(f)[0].split(prefix)[-1]
            numbers.append(int(num_part))
        except ValueError:
            continue
    next_num = max(numbers) + 1 if numbers else 1
    return os.path.join(folder, f"{prefix}{next_num:03d}{ext}")

def process_image_with_ml(image_path, place_name):
    if not ML_ENABLED or ml_transformer is None:
        raise ValueError("ML Model is not loaded.")

    print(f"🎨 Generating 1960s style for: {place_name}")
    # เรียกใช้ transform_to_1960s จาก class ของเพื่อน
    result_pil = ml_transformer.transform_to_1960s(image_path, place_name)
    
    if result_pil is None:
        raise ValueError("ML Model returned None.")

    # แปลงเป็น bytes เพื่อบันทึก
    buffered = BytesIO()
    result_pil.save(buffered, format="PNG")
    return buffered.getvalue()

# --- ROUTES ---

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", places=PLACE_LABELS)

# ==========================================
# 🟢 STEP 1: รับไฟล์ + ตรวจสอบ (ยังไม่เจนภาพ)
# ==========================================
@app.route("/step1_classify", methods=["POST"])
def step1_classify():
    try:
        # 1. เช็ค Input
        place_selected = request.form.get("location")
        if not place_selected or "image" not in request.files:
            return jsonify({"error": "Missing location or image"}), 400
            
        file = request.files["image"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        # 2. บันทึกไฟล์ชั่วคราว
        filename = secure_filename(file.filename)
        temp_filename = f"temp_{filename}"
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
        file.save(temp_path)

        print(f"\n--- 🔍 Step 1: Classification ({place_selected}) ---")
        
        # 3. เรียก Classifier
        predicted_place, score, is_valid = classify_image(temp_path)
        print(f"AI Result: {predicted_place} ({score*100:.2f}%)")

        # 4. เช็คเงื่อนไข (Logic Gate)
        
        # กรณี Rejected (เช่น ติดคน, มุมกล้องแย่)
        if "Rejected" in predicted_place:
            error_msg = f"Image Rejected: {predicted_place.replace('Rejected', '').strip('() ')}"
            return jsonify({"success": False, "error": error_msg}), 400

        # 👇👇 แก้ไขตรงนี้ครับ: กรณี Unknown (AI ไม่รู้ที่ไหน) 👇👇
        if "Other" in predicted_place or "Unknown" in predicted_place:
             # ดึงสิ่งที่ AI เห็นออกมา (เช่น Other (Cat) -> Cat)
             ai_guess = predicted_place.replace("Other", "").strip("() ")
             if not ai_guess or ai_guess == "Unknown":
                 ai_guess = "Nothing recognizable"
                 
             return jsonify({
                 "success": False, 
                 "error": f"❌ Could not identify the location. AI sees: '{ai_guess}'"
             }), 400
        # 👆👆 จบส่วนแก้ไข 👆👆

        # กรณีสถานที่ผิด (Mismatch)
        if predicted_place != place_selected:
            return jsonify({
                "success": False,
                "error": f"Mismatch! Selected '{place_selected}' but AI sees '{predicted_place}' ({score*100:.1f}%)."
            }), 400

        # ✅ ผ่าน Step 1: ส่งผลกลับไปบอกหน้าเว็บก่อน!
        # ส่งชื่อไฟล์ชั่วคราวกลับไปด้วย เพื่อเอาไปใช้ใน Step 2
        return jsonify({
            "success": True,
            "message": f"Verified: {predicted_place}",
            "score_percent": f"{score*100:.1f}", # ส่งเป็นตัวเลขเปอร์เซ็นต์
            "temp_filename": temp_filename,       # สำคัญ! ต้องใช้ชื่อนี้ไปเจนต่อ
            "place_name": predicted_place
        })

    except Exception as e:
        print(f"Error Step 1: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


# ==========================================
# 🎨 STEP 2: รับคำสั่งเจนภาพ (ต่อจากตะกี้)
# ==========================================
@app.route("/step2_generate", methods=["POST"])
def step2_generate():
    temp_path = None # ประกาศตัวแปรไว้ก่อนเพื่อใช้ใน finally
    try:
        # รับข้อมูลจาก JSON
        data = request.json
        temp_filename = data.get("temp_filename")
        place_name = data.get("place_name")

        if not temp_filename or not place_name:
            return jsonify({"error": "Missing data for generation"}), 400

        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
        
        # เช็คว่าไฟล์ยังอยู่ไหม
        if not os.path.exists(temp_path):
             return jsonify({"error": "Temporary file lost. Please upload again."}), 400

        print(f"\n--- 🎨 Step 2: Generation Start ({place_name}) ---")

        # เรียก ML Model
        img_bytes = process_image_with_ml(temp_path, place_name=place_name)

        # บันทึกรูปผลลัพธ์ (อันนี้เก็บไว้โชว์ ไม่ลบ)
        images_db_folder = os.path.join(app.config['UPLOAD_FOLDER'], "images_database")
        output_img_path = get_next_filename(images_db_folder, ext=".png")
        
        with open(output_img_path, "wb") as f:
            f.write(img_bytes)
        
        # ส่ง URL กลับ
        web_path = output_img_path.replace("\\", "/")
        
        return jsonify({
            "success": True,
            "img_url": web_path,
            "message": "Generation Complete!"
        })

    except Exception as ml_error:
        print(f"ML Error: {ml_error}")
        return jsonify({"success": False, "error": f"Generation failed: {str(ml_error)}"}), 500

    finally:
        # 👇👇 เพิ่มส่วนนี้: ลบไฟล์ temp ทิ้งทุกกรณี (ไม่ว่าจะสำเร็จหรือ Error) 👇👇
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
                print(f"🗑️ Cleaned up temp file: {temp_path}")
            except Exception as e:
                print(f"⚠️ Could not delete temp file: {e}")
        # 👆👆 จบส่วนลบไฟล์ 👆👆

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=False)