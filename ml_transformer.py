import torch
import cv2
import numpy as np
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UNet2DConditionModel
from peft import PeftModel

# ตรวจสอบว่ามี GPU หรือไม่
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"ML Transformer: Using device {DEVICE}")

class EraVisionTransformer:
    """
    คลาสนี้ทำหน้าที่ห่อหุ้ม (wrap) โมเดล ControlNet + LoRA
    เพื่อให้ app.py เรียกใช้งานได้ง่ายๆ
    """

    def __init__(self, lora_model_path: str):
        """
        โหลดโมเดลทั้งหมดตอนเริ่มต้นแอป (โหลดครั้งเดียว)

        Args:
            lora_model_path: Path ไปยังโฟลเดอร์ที่เก็บ LoRA adapter
        """
        print(f"Loading base models...")
        # 1. โหลด ControlNet (Canny)
        self.controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11p_sd15_canny",
            torch_dtype=torch.float16
        )

        # 2. โหลด Base Pipeline (Stable Diffusion 1.5)
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=self.controlnet,
            torch_dtype=torch.float16,
            safety_checker=None
        )

        print(f"Loading LoRA adapter from {lora_model_path}...")
        # 3. โหลดและ "สวม" LoRA adapter (ไฟล์ .safetensors)
        # นี่คือส่วนที่สำคัญที่สุด
        self.pipe.unet = PeftModel.from_pretrained(
            self.pipe.unet,
            lora_model_path,
            torch_dtype=torch.float16
        )

        # 4. ย้ายทุกอย่างไปที่ GPU
        self.pipe = self.pipe.to(DEVICE)
        print("✅ ML Model loaded and ready.")

    def _get_canny_edge(self, pil_image: Image.Image) -> Image.Image:
        """
        ฟังก์ชัน Helper เพื่อสร้าง Canny edge map จาก PIL Image
        (นำมาจากโค้ด Colab Cell 6/7)
        """
        # แปลง PIL Image เป็น numpy array
        img_array = np.array(pil_image)

        # Canny edge detection
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        # แปลงกลับเป็น PIL Image
        return Image.fromarray(edges)

    def transform_to_1960s(self, image_path: str, place_name: str) -> Image.Image:
        """
        ฟังก์ชันหลักในการแปลงภาพ

        Args:
            image_path: Path ไปยังไฟล์ภาพที่ผู้ใช้อัปโหลด
            place_name: ชื่อสถานที่ (เช่น 'Democracy Monument')

        Returns:
            PIL Image ของภาพที่แปลงแล้ว
        """
        try:
            # 1. โหลดภาพต้นฉบับ
            original_image = Image.open(image_path).convert('RGB')
            original_image = original_image.resize((512, 512))

            # 2. สร้าง Canny edge (Control signal)
            control_image = self._get_canny_edge(original_image)

            # 3. สร้าง Prompt (ควรจะอิงตาม place_name)
            # คุณสามารถปรับปรุง logic นี้ได้ในอนาคต
            if "democracy" in place_name.lower():
                prompt = "Democracy Monument Bangkok, 1960s vintage photograph, historical architecture, old cars, retro atmosphere"
                negative_prompt = "modern, contemporary, 2020s, new buildings, modern cars, smartphone, digital"
            else:
                # Prompt ทั่วไป
                prompt = "Bangkok 1960s vintage photograph, historical architecture, retro"
                negative_prompt = "modern, contemporary, 2020s"

            print(f"Generating image with prompt: {prompt}")

            # 4. รัน Pipeline!
            output = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=control_image,  # นี่คือ Canny edge
                num_inference_steps=30,
                guidance_scale=7.5,
            ).images[0]

            return output

        except Exception as e:
            print(f"Error during transformation: {e}")
            return None