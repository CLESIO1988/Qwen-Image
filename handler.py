print("✅ handler.py loaded successfully")

import os
import io
import base64
from PIL import Image
import torch
from diffusers import QwenImageEditPipeline

# Global model cache
pipe = None

def load_model():
    global pipe
    if pipe is None:
        print("Loading Qwen/Qwen-Image-Edit...")
        pipe = QwenImageEditPipeline.from_pretrained(
            "Qwen/Qwen-Image-Edit",
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
        )
        pipe.to("cuda")
        print("Model loaded.")
    return pipe

def handler(event):
    try:
        # Parse input
        image_data = event.get("image")
        prompt = event.get("prompt", "")
        negative_prompt = event.get("negative_prompt", " ")
        steps = event.get("steps", 50)
        seed = event.get("seed", 0)

        if not image_data or not prompt:
            return {"error": "Missing image or prompt"}

        # Decode image
        image_bytes = base64.b64decode(image_data)
        input_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Load model
        pipe = load_model()

        # Generate
        generator = torch.manual_seed(seed)
        with torch.inference_mode():
            output = pipe(
                image=input_image,
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=steps,
                generator=generator,
                true_cfg_scale=4.0
            )
            output_image = output.images[0]

        # Encode result
        buf = io.BytesIO()
        output_image.save(buf, format="PNG")
        img_str = base64.b64encode(buf.getvalue()).decode()

        return {"image": img_str}

    except Exception as e:
        print(f"Error: {str(e)}")
        return {"error": str(e)}