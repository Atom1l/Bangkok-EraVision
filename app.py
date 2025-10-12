from flask import Flask, request, render_template, send_file
import openai
from io import BytesIO
import os
from PIL import Image
import base64
import requests
import glob
import random
from runwayml import RunwayML, TaskFailedError
from dotenv import load_dotenv


from classifier import check_image_category


# โหลด environment variables
load_dotenv()


# ตั้งค่า Flask
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "uploads"
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


# --- API Keys ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
RUNWAY_API_KEY = os.getenv("RUNWAY_API_KEY")


openai.api_key = OPENAI_API_KEY
runway_client = RunwayML(api_key=RUNWAY_API_KEY)


# --- PROMPTS ---
BASE_PROMPT_IMAGE = (
   "A realistic street scene from the 1960s, as seen through the eyes of a local person living in that time. "
   "Show authentic mid-20th-century urban atmosphere architecture, signage, shopfronts, and textures typical of that era. Use natural daylight colors (no sepia or vintage filters). "
   "Keep the environment calm and authentic: minimal people, few or no cars unless historically accurate. Realistic, historically accurate, detailed lighting and materials."
)
PROMPT_VIDEO = "Short 5-second video, gentle camera motion, vintage 1960s street style"


# --- ฟังก์ชันช่วยสร้างชื่อไฟล์เรียงลำดับ ---
def get_next_filename(folder, prefix="BangkokEra", ext=".png"):
   os.makedirs(folder, exist_ok=True)
   files = glob.glob(os.path.join(folder, f"{prefix}*{ext}"))
   if not files:
       return os.path.join(folder, f"{prefix}001{ext}")
   numbers = [int(os.path.splitext(f)[0].split(prefix)[-1]) for f in files]
   next_num = max(numbers) + 1
   return os.path.join(folder, f"{prefix}{next_num:03d}{ext}")


# --- ฟังก์ชันแปลงภาพ OpenAI (พร้อม reference dataset) ---
def convert_image_to_1960s(image_path, reference_folder=None):
   allowed_exts = (".png", ".jpg", ".jpeg", ".webp")
   if not image_path.lower().endswith(allowed_exts):
       raise ValueError("Only PNG, JPG, JPEG, or WebP images are supported.")


   # เปิดภาพหลัก
   img = Image.open(image_path).convert("RGB")
   buffered = BytesIO()
   img.save(buffered, format="PNG")
   buffered.seek(0)


   # --- สร้าง reference prompt ---
   ref_prompt = ""
   if reference_folder and os.path.exists(reference_folder):
       ref_images = glob.glob(os.path.join(reference_folder, "*"))
       if ref_images:
           ref_img = random.choice(ref_images)
           ref_prompt = f"Use the lighting, color tone, and architecture style similar to {os.path.basename(ref_img)}. "


   full_prompt = (
       f"{ref_prompt}"
       "A realistic street scene from the 1960s, as seen through the eyes of a local person living in that time. "
       "Show authentic mid-20th-century urban atmosphere architecture, signage, shopfronts, and textures typical of that era. Use natural daylight colors (no sepia or vintage filters). "
       "Keep the environment calm and authentic: minimal people, few or no cars unless historically accurate. Realistic, historically accurate, detailed lighting and materials."
   )


   # เรียก OpenAI API
   response = openai.images.edit(
       model="gpt-image-1",
       image=("input.png", buffered, "image/png"),
       prompt=full_prompt,
       size="1024x1024"
   )


   if response.data and response.data[0].b64_json:
       return base64.b64decode(response.data[0].b64_json)
   else:
       raise ValueError("OpenAI did not return valid image data.")


# --- ฟังก์ชันสร้างวิดีโอ Runway ---
def generate_video_from_image(img_bytes, output_path="output.mp4"):
   img_b64 = base64.b64encode(img_bytes).decode("utf-8")


   task = runway_client.image_to_video.create(
       model="gen4_turbo",
       prompt_image=f"data:image/png;base64,{img_b64}",
       prompt_text=PROMPT_VIDEO,
       ratio="1280:720",
       duration=5
   ).wait_for_task_output()


   if not task.output:
       raise ValueError("Runway did not return a valid video URL.")


   video_url = task.output[0] if isinstance(task.output[0], str) else task.output[0].get("url")
   if not video_url:
       raise ValueError("Runway did not return a valid video URL.")


   r = requests.get(video_url)
   if r.status_code != 200:
       raise ValueError("Failed to download video from Runway.")


   with open(output_path, "wb") as f:
       f.write(r.content)


   return output_path


# --- ROUTES ---
@app.route("/", methods=["GET", "POST"])
def index():
   message = ""
   img_file = None
   video_file = None


   if request.method == "POST":
       place_selected = request.form.get("location")


       if "image" not in request.files:
           message = "No file uploaded."
       else:
           file = request.files["image"]
           if file.filename == "":
               message = "No file selected."
           else:
               try:
                   # --- Save temp file ---
                   temp_path = os.path.join(app.config['UPLOAD_FOLDER'], "temp_upload.png")
                   file.save(temp_path)


                   # --- ตรวจสถานที่ ---
                   confidence = check_image_category(temp_path, place_selected)
                   threshold = 0.8


                   if confidence < threshold:
                       message = f"ภาพนี้อาจไม่ใช่ {place_selected} (ความมั่นใจ {confidence:.2f})"
                       img_file = temp_path
                   else:
                       # --- ใช้ dataset เป็น reference ---
                       ref_folder = os.path.join("dataset", place_selected.replace(" ", "_"))
                       img_bytes = convert_image_to_1960s(temp_path, reference_folder=ref_folder)


                       # --- บันทึกภาพและวิดีโอ ---
                       images_folder = os.path.join(app.config['UPLOAD_FOLDER'], "images_database")
                       videos_folder = os.path.join(app.config['UPLOAD_FOLDER'], "videos_database")
                       os.makedirs(images_folder, exist_ok=True)
                       os.makedirs(videos_folder, exist_ok=True)


                       output_img_path = get_next_filename(images_folder, ext=".png")
                       with open(output_img_path, "wb") as f:
                           f.write(img_bytes)
                       img_file = output_img_path


                       output_video_path = get_next_filename(videos_folder, ext=".mp4")
                       video_file = generate_video_from_image(img_bytes, output_video_path)


               except Exception as e:
                   message = f"Error: {str(e)}"


   return render_template("index.html", message=message, img_file=img_file, video_file=video_file)




@app.route("/image")
def image():
   images_folder = os.path.join(app.config['UPLOAD_FOLDER'], "images_database")
   latest_image = get_next_filename(images_folder, ext=".png")
   latest_image = os.path.join(images_folder, f"BangkokEra{int(latest_image[-7:-4])-1:03d}.png")
   return send_file(latest_image, mimetype="image/png")




@app.route("/video")
def video():
   videos_folder = os.path.join(app.config['UPLOAD_FOLDER'], "videos_database")
   latest_video = get_next_filename(videos_folder, ext=".mp4")
   latest_video = os.path.join(videos_folder, f"BangkokEra{int(latest_video[-7:-4])-1:03d}.mp4")
   return send_file(latest_video, mimetype="video/mp4")




if __name__ == "__main__":
   app.run(debug=True)