import runpod
import os
import io
import base64
from PIL import Image
import torch
from diffusers import QwenImageEditPipeline

print("✅ handler.py loaded successfully")

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
        # RunPod expects input under the "input" key
        job_input = event["input"]
        image_data = job_input.get("image")
        prompt = job_input.get("prompt", "")
        negative_prompt = job_input.get("negative_prompt", " ")
        steps = job_input.get("steps", 50)
        seed = job_input.get("seed", 0)

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

# This line is required for RunPod serverless workers
runpod.serverless.start({"handler": handler})
