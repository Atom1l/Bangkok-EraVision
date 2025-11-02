import os
import glob
import base64
import requests
from io import BytesIO
from PIL import Image
from flask import Flask, request, render_template, send_file, jsonify # <-- เพิ่ม jsonify
from werkzeug.utils import secure_filename # <-- import นี้เพื่อความปลอดภัย
from dotenv import load_dotenv

# --- Import ระบบ ML ของเรา ---
from ml_transformer import EraVisionTransformer
# from classifier import check_image_category # (ยังไม่ใช้)
# from reference_prompt_builder import build_prompt # (ไม่จำเป็น ถ้าใช้ ML)

# --- โหลด environment variables (.env) ---
load_dotenv()

# --- 1. แก้ไข Path ของ Model ---
# เปลี่ยนจาก "path/to/drive/..." มาเป็น path ในโปรเจกต์ของคุณ
ML_MODEL_PATH = "models/democracy_monument_1960s" 

# --- ตั้งค่า Flask ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "static/uploads" # <-- แนะนำให้เก็บใน static
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- โหลด Model ตอนเริ่มแอป (ใช้ VRAM) ---
print("กำลังโหลด EraVision ML Model... (โปรดรอ)")
ml_transformer = EraVisionTransformer(ML_MODEL_PATH)
print("✅ Model พร้อมใช้งาน")

# --- API Keys (สำหรับส่วนวิดีโอในอนาคต) ---
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") # (ไม่ใช้สำหรับการสร้างภาพแล้ว)
RUNWAY_API_KEY = os.getenv("RUNWAY_API_KEY")
if RUNWAY_API_KEY:
    try:
        from runwayml import RunwayML
        runway_client = RunwayML(api_key=RUNWAY_API_KEY)
        print("✅ RunwayML client พร้อมใช้งาน")
    except ImportError:
        runway_client = None
        print("⚠️ ไม่ได้ติดตั้ง RunwayML library, ส่วนวิดีโอจะไม่ทำงาน")
else:
    runway_client = None
    print("⚠️ ไม่พบ RUNWAY_API_KEY, ส่วนวิดีโอจะไม่ทำงาน")

PROMPT_VIDEO = "Short 5-second video, gentle camera motion, vintage 1960s street style"

# --- 2. นี่คือฟังก์ชันที่ถูกต้อง (อันเดียว) ---
def convert_image_to_1960s(image_path, place_name):
    """
    ใช้ ML model (ControlNet + LoRA) ที่เราเทรนมา
    """
    print(f"กำลังแปลงภาพด้วย ML Model... สถานที่: {place_name}")
    # ใช้ ML model
    result_pil = ml_transformer.transform_to_1960s(image_path, place_name)
    
    if result_pil is None:
        raise ValueError("ML Model ไม่สามารถประมวลผลภาพได้")

    # แปลงเป็น bytes
    buffered = BytesIO()
    result_pil.save(buffered, format="PNG")
    buffered.seek(0) # <-- สำคัญมาก: ย้าย pointer กลับไปที่จุดเริ่มต้น
    return buffered.getvalue()

# (ฟังก์ชัน OpenAI ที่ซ้ำซ้อน ถูกลบออกจากตรงนี้แล้ว)

def get_next_filename(folder, prefix="BangkokEra", ext=".png"):
    """หาชื่อไฟล์ถัดไปในโฟลเดอร์ (เช่น BangkokEra001.png)"""
    os.makedirs(folder, exist_ok=True)
    files = glob.glob(os.path.join(folder, f"{prefix}*{ext}"))
    if not files:
        return os.path.join(folder, f"{prefix}001{ext}")
    numbers = [int(os.path.splitext(f)[0].split(prefix)[-1]) for f in files]
    next_num = max(numbers) + 1
    return os.path.join(folder, f"{prefix}{next_num:03d}{ext}")

def generate_video_from_image(img_bytes, output_path="output.mp4"):
    """(ส่วนนี้สำหรับอนาคต) สร้างวิดีโอจาก RunwayML"""
    if not runway_client:
        raise ValueError("RunwayML client ไม่ได้ตั้งค่าไว้")

    img_b64 = base64.b64encode(img_bytes).decode("utf-8")
    task = runway_client.image_to_video.create(
        model="gen4_turbo",
        prompt_image=f"data:image/png;base64,{img_b64}",
        prompt_text=PROMPT_VIDEO,
        ratio="1280:720",
        duration=5
    ).wait_for_task_output()

    if not task.output:
        raise ValueError("Runway ไม่ส่ง output วิดีโอมาให้")

    video_url = task.output[0] if isinstance(task.output[0], str) else task.output[0].get("url")
    if not video_url:
        raise ValueError("Runway ไม่ส่ง URL วิดีโอมาให้")

    r = requests.get(video_url)
    if r.status_code != 200:
        raise ValueError("ไม่สามารถดาวน์โหลดวิดีโอจาก Runway ได้")

    with open(output_path, "wb") as f:
        f.write(r.content)
    return output_path

# --- ส่วนควบคุมหน้าเว็บ (Routes) ---

@app.route("/", methods=["GET"])
def index():
    """แสดงหน้าเว็บหลัก"""
    # ล้างค่าเก่า (ถ้ามี)
    return render_template("index.html", message="", img_file=None, video_file=None)

@app.route("/upload", methods=["POST"])
def upload_and_process():
    """
    รับไฟล์ที่อัปโหลด, ประมวลผล, และส่งผลลัพธ์กลับไป
    """
    message = ""
    img_file_url = None # เราจะส่ง URL กลับไปแทน path
    video_file_url = None

    try:
        place_selected = request.form.get("location")
        if not place_selected:
            raise ValueError("กรุณาเลือกสถานที่")

        if "image" not in request.files:
            raise ValueError("ไม่พบไฟล์ที่อัปโหลด")
            
        file = request.files["image"]
        if file.filename == "":
            raise ValueError("กรุณาเลือกไฟล์")

        # 1. บันทึกไฟล์ที่อัปโหลดชั่วคราว
        # ใช้ secure_filename เพื่อป้องกันปัญหาชื่อไฟล์แปลกๆ
        filename = secure_filename(file.filename)
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(temp_path)
        
        # (ส่วนของ Classifier ที่คอมเมนต์ไว้)
        # confidence = check_image_category(temp_path, place_selected)
        # ...

        # 2. เรียกใช้ ML Model
        # ฟังก์ชันนี้จะเรียกตัวที่ถูกต้อง (ML Model)
        img_bytes = convert_image_to_1960s(
            temp_path, 
            place_name=place_selected
        )

        # 3. บันทึกภาพผลลัพธ์
        images_folder = os.path.join(app.config['UPLOAD_FOLDER'], "images_database")
        output_img_path = get_next_filename(images_folder, ext=".png")
        
        with open(output_img_path, "wb") as f:
            f.write(img_bytes)
        
        # สร้าง URL ที่ template จะเรียกใช้ได้
        # (เปลี่ยนจาก path ตรงๆ เป็น URL ที่ปลอดภัยกว่า)
        img_file_url = f"/{output_img_path}"
        message = "สร้างภาพสำเร็จ!"

        # (ส่วนสร้างวิดีโอ ยังคอมเมนต์ไว้เหมือนเดิม)
        # videos_folder = os.path.join(app.config['UPLOAD_FOLDER'], "videos_database")
        # output_video_path = get_next_filename(videos_folder, ext=".mp4")
        # video_file = generate_video_from_image(img_bytes, output_video_path)
        # video_file_url = f"/{output_video_path}"

    except Exception as e:
        print(f"เกิดข้อผิดพลาด: {e}")
        message = f"Error: {str(e)}"
        # คืนค่า error กลับไปให้
        return jsonify({"error": message}), 500

    # คืนค่าผลลัพธ์เป็น JSON
    return jsonify({
        "message": message,
        "img_url": img_file_url,
        "video_url": video_file_url
    })

# (เราไม่ต้องการ route /image และ /video อีกต่อไป
# เพราะเราส่ง URL กลับไปใน JSON แล้ว 
# Flask จะจัดการไฟล์ static ให้อัตโนมัติ)

if __name__ == "__main__":
    # app.run(debug=True) # debug=True อาจทำให้โมเดลโหลดซ้ำ
    app.run(host='0.0.0.0', port=5000, debug=False)
