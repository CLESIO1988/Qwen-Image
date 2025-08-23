#!/usr/bin/env python3

import os
import sys
import traceback

print("=" * 50)
print("🚀 Handler starting...")
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")
print(f"Current working directory: {os.getcwd()}")
print(f"Files in current directory: {os.listdir('.')}")
print("=" * 50)

try:
    import runpod
    print("✅ runpod imported successfully")
except Exception as e:
    print(f"❌ Failed to import runpod: {e}")
    traceback.print_exc()
    sys.exit(1)

try:
    import io
    import base64
    from PIL import Image
    import torch
    print("✅ Basic imports successful")
except Exception as e:
    print(f"❌ Failed basic imports: {e}")
    traceback.print_exc()
    sys.exit(1)

print("🔄 Loading inference module...")

# Global model cache
pipe = None

def load_model():
    global pipe
    if pipe is None:
        try:
            print("🔄 Attempting to load model...")
            
            # Try multiple import strategies
            try:
                from diffusers import QwenImageEditPipeline
                print("✅ QwenImageEditPipeline found")
                
                pipe = QwenImageEditPipeline.from_pretrained(
                    "Qwen/Qwen-Image-Edit",
                    torch_dtype=torch.bfloat16,
                    use_safetensors=True,
                    cache_dir="/tmp/cache"
                )
                pipe.to("cuda" if torch.cuda.is_available() else "cpu")
                print("✅ QwenImageEditPipeline loaded successfully")
                
            except Exception as qwen_error:
                print(f"⚠️ QwenImageEditPipeline failed: {qwen_error}")
                print("🔄 Trying fallback pipeline...")
                
                from diffusers import StableDiffusionInstructPix2PixPipeline
                pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
                    "timbrooks/instruct-pix2pix",
                    torch_dtype=torch.float16,
                    use_safetensors=True,
                    cache_dir="/tmp/cache"
                )
                pipe.to("cuda" if torch.cuda.is_available() else "cpu")
                print("✅ Fallback pipeline loaded successfully")
                
        except Exception as e:
            print(f"❌ Failed to load any pipeline: {e}")
            traceback.print_exc()
            raise e
    return pipe

def handler(event):
    """RunPod handler function"""
    try:
        print(f"📥 Received event: {event}")
        
        # RunPod expects input under the "input" key
        job_input = event.get("input", {})
        print(f"📋 Job input keys: {list(job_input.keys())}")
        
        # Extract parameters
        image_data = job_input.get("image")
        prompt = job_input.get("prompt", "")
        negative_prompt = job_input.get("negative_prompt", "")
        steps = job_input.get("steps", 20)  # Reduced default steps
        seed = job_input.get("seed", 0)
        
        print(f"🎯 Prompt: '{prompt}'")
        print(f"🔢 Steps: {steps}, Seed: {seed}")

        # Validate required inputs
        if not image_data:
            error_msg = "Missing 'image' in input"
            print(f"❌ {error_msg}")
            return {"error": error_msg}
            
        if not prompt:
            error_msg = "Missing 'prompt' in input"
            print(f"❌ {error_msg}")
            return {"error": error_msg}

        # Decode image
        print("🖼️ Decoding image...")
        try:
            image_bytes = base64.b64decode(image_data)
            input_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            print(f"📐 Input image size: {input_image.size}")
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
                # Try QwenImageEditPipeline parameters first
                try:
                    output = pipe(
                        image=input_image,
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        num_inference_steps=steps,
                        generator=generator,
                        true_cfg_scale=4.0
                    )
                except TypeError:
                    # Fallback to InstructPix2Pix parameters
                    output = pipe(
                        image=input_image,
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        num_inference_steps=steps,
                        generator=generator,
                        guidance_scale=7.5,
                        image_guidance_scale=1.5
                    )
                
                output_image = output.images[0]
            print("✅ Image generation completed")
        except Exception as e:
            error_msg = f"Error during generation: {str(e)}"
            print(f"❌ {error_msg}")
            traceback.print_exc()
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
        traceback.print_exc()
        return {"error": error_msg}

# Test function for debugging
def test_imports():
    """Test all imports"""
    print("🧪 Testing imports...")
    
    try:
        import torch
        print(f"✅ torch: {torch.__version__}")
    except Exception as e:
        print(f"❌ torch: {e}")
        return False
    
    try:
        import diffusers
        print(f"✅ diffusers: {diffusers.__version__}")
    except Exception as e:
        print(f"❌ diffusers: {e}")
        return False
    
    try:
        from diffusers import QwenImageEditPipeline
        print("✅ QwenImageEditPipeline available")
    except Exception as e:
        print(f"⚠️ QwenImageEditPipeline: {e}")
        try:
            from diffusers import StableDiffusionInstructPix2PixPipeline
            print("✅ StableDiffusionInstructPix2PixPipeline available as fallback")
        except Exception as e2:
            print(f"❌ No fallback available: {e2}")
            return False
    
    return True

if __name__ == "__main__":
    print("🏁 Starting RunPod serverless worker...")
    
    # Test imports first
    if not test_imports():
        print("❌ Import tests failed, exiting...")
        sys.exit(1)
    
    print("🚀 Starting RunPod serverless...")
    try:
        runpod.serverless.start({"handler": handler})
    except Exception as e:
        print(f"❌ Failed to start RunPod serverless: {e}")
        traceback.print_exc()
        sys.exit(1)# Test import to catch issues early
RUN python -c "import runpod; import torch; import diffusers; print('✅ All imports successful')"

# Test model import specifically
RUN python -c "from diffusers import QwenImageEditPipeline; print('✅ QwenImageEditPipeline import successful')"

# The container will be started by RunPod
