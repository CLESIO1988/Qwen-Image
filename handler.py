import runpod
import os
import io
import base64
from PIL import Image
import torch
from diffusers import QwenImageEditPipeline
import traceback

print("✅ handler.py loaded successfully")

# Global model cache
pipe = None

def load_model():
    global pipe
    if pipe is None:
        try:
            print("🔄 Loading Qwen/Qwen-Image-Edit...")
            pipe = QwenImageEditPipeline.from_pretrained(
                "Qwen/Qwen-Image-Edit",
                torch_dtype=torch.bfloat16,
                use_safetensors=True,
                cache_dir="/tmp/cache"
            )
            pipe.to("cuda")
            print("✅ Model loaded successfully.")
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            print(traceback.format_exc())
            raise e
    return pipe

def handler(event):
    try:
        print(f"📥 Received event: {event}")
        
        # RunPod expects input under the "input" key
        job_input = event.get("input", {})
        print(f"📋 Job input: {job_input}")
        
        image_data = job_input.get("image")
        prompt = job_input.get("prompt", "")
        negative_prompt = job_input.get("negative_prompt", " ")
        steps = job_input.get("steps", 50)
        seed = job_input.get("seed", 0)
        
        print(f"🎯 Prompt: {prompt}")
        print(f"🔢 Steps: {steps}, Seed: {seed}")

        if not image_data or not prompt:
            error_msg = "Missing image or prompt"
            print(f"❌ {error_msg}")
            return {"error": error_msg}

        # Decode image
        print("🖼️ Decoding image...")
        try:
            image_bytes = base64.b64decode(image_data)
            input_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            print(f"📐 Image size: {input_image.size}")
        except Exception as e:
            error_msg = f"Error decoding image: {str(e)}"
            print(f"❌ {error_msg}")
            return {"error": error_msg}

        # Load model
        print("🤖 Loading model...")
        try:
            pipe = load_model()
        except Exception as e:
            error_msg = f"Error loading model: {str(e)}"
            print(f"❌ {error_msg}")
            return {"error": error_msg}

        # Generate
        print("🎨 Starting image generation...")
        try:
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
            print("✅ Image generation completed")
        except Exception as e:
            error_msg = f"Error during generation: {str(e)}"
            print(f"❌ {error_msg}")
            print(traceback.format_exc())
            return {"error": error_msg}

        # Encode result
        print("📤 Encoding output image...")
        try:
            buf = io.BytesIO()
            output_image.save(buf, format="PNG")
            img_str = base64.b64encode(buf.getvalue()).decode()
            print("✅ Image encoded successfully")
        except Exception as e:
            error_msg = f"Error encoding output: {str(e)}"
            print(f"❌ {error_msg}")
            return {"error": error_msg}

        print("🎉 Handler completed successfully")
        return {"image": img_str}

    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        print(f"❌ {error_msg}")
        print(traceback.format_exc())
        return {"error": error_msg}

# This line is required for RunPod serverless workers
if __name__ == "__main__":
    print("🚀 Starting RunPod serverless worker...")
    runpod.serverless.start({"handler": handler})        if not image_data or not prompt:
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
