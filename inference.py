import os
import uuid
import torch
from PIL import Image
import base64
from io import BytesIO
import traceback

print("🚀 Loading image editing model...")

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"📱 Using device: {device}")

# Try to import QwenImageEditPipeline with fallback
pipeline = None
pipeline_type = None

try:
    from diffusers import QwenImageEditPipeline
    print("✅ QwenImageEditPipeline found, attempting to load...")
    
    pipeline = QwenImageEditPipeline.from_pretrained(
        "Qwen/Qwen-Image-Edit",
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
        cache_dir="/tmp/cache"
    )
    pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)
    pipeline_type = "qwen"
    print("✅ QwenImageEditPipeline loaded successfully")
    
except Exception as e:
    print(f"❌ Failed to load QwenImageEditPipeline: {str(e)}")
    print("🔄 Attempting fallback to StableDiffusionInstructPix2PixPipeline...")
    
    try:
        from diffusers import StableDiffusionInstructPix2PixPipeline
        pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            "timbrooks/instruct-pix2pix",
            torch_dtype=torch.float16,
            use_safetensors=True,
            cache_dir="/tmp/cache"
        )
        pipeline.to(device)
        pipeline.set_progress_bar_config(disable=True)
        pipeline_type = "instructpix2pix"
        print("✅ StableDiffusionInstructPix2PixPipeline loaded as fallback")
        
    except Exception as e2:
        print(f"❌ Failed to load fallback pipeline: {str(e2)}")
        print("🔄 Attempting final fallback to StableDiffusionImg2ImgPipeline...")
        
        try:
            from diffusers import StableDiffusionImg2ImgPipeline
            pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=torch.float16,
                use_safetensors=True,
                cache_dir="/tmp/cache"
            )
            pipeline.to(device)
            pipeline.set_progress_bar_config(disable=True)
            pipeline_type = "img2img"
            print("✅ StableDiffusionImg2ImgPipeline loaded as final fallback")
            
        except Exception as e3:
            print(f"❌ All pipelines failed to load: {str(e3)}")
            raise RuntimeError("No compatible image editing pipeline could be loaded")

def edit_image_from_base64(image_b64: str, prompt: str, **kwargs) -> str:
    """Edit image from base64 string using available pipeline"""
    try:
        print(f"🔤 Decoding base64 image...")
        image_bytes = base64.b64decode(image_b64)
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        print(f"📐 Image loaded: {img.size}")
        
        return edit_image_from_pil(img, prompt, **kwargs)
        
    except Exception as e:
        print(f"❌ Error in edit_image_from_base64: {str(e)}")
        print(traceback.format_exc())
        raise e

def edit_image_from_pil(img: Image.Image, prompt: str, **kwargs) -> str:
    """Edit PIL Image using available pipeline"""
    try:
        # Parameters based on pipeline type
        if pipeline_type == "qwen":
            params = {
                "image": img,
                "prompt": prompt,
                "generator": torch.manual_seed(kwargs.get("seed", 0)),
                "true_cfg_scale": kwargs.get("true_cfg_scale", 4.0),
                "negative_prompt": kwargs.get("negative_prompt", ""),
                "num_inference_steps": kwargs.get("num_inference_steps", 50),
            }
        elif pipeline_type == "instructpix2pix":
            params = {
                "image": img,
                "prompt": prompt,
                "generator": torch.manual_seed(kwargs.get("seed", 0)),
                "guidance_scale": kwargs.get("true_cfg_scale", 7.5),
                "image_guidance_scale": kwargs.get("image_guidance_scale", 1.5),
                "negative_prompt": kwargs.get("negative_prompt", ""),
                "num_inference_steps": kwargs.get("num_inference_steps", 50),
            }
        elif pipeline_type == "img2img":
            params = {
                "image": img,
                "prompt": prompt,
                "generator": torch.manual_seed(kwargs.get("seed", 0)),
                "guidance_scale": kwargs.get("true_cfg_scale", 7.5),
                "negative_prompt": kwargs.get("negative_prompt", ""),
                "num_inference_steps": kwargs.get("num_inference_steps", 50),
                "strength": kwargs.get("strength", 0.8),
            }
        else:
            raise RuntimeError("No pipeline available")
        
        # Remove non-pipeline parameters
        params.pop("seed", None)
        
        print(f"🎨 Editing image with {pipeline_type} pipeline")
        print(f"🎯 Prompt: '{prompt}'")
        print(f"⚙️  Steps: {params.get('num_inference_steps', 'N/A')}")
        
        # Run inference
        with torch.inference_mode():
            output = pipeline(**params)
        
        # Get the edited image
        edited_img = output.images[0]
        print("✅ Image generation completed")
        
        # Save the result
        out_path = f"/tmp/outputs/{uuid.uuid4()}.png"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        edited_img.save(out_path)
        
        print(f"💾 Image saved to: {out_path}")
        return out_path
        
    except Exception as e:
        print(f"❌ Error in edit_image_from_pil: {str(e)}")
        print(traceback.format_exc())
        raise e

def edit_image_return_base64(img: Image.Image, prompt: str, **kwargs) -> str:
    """Edit PIL Image and return as base64 string"""
    try:
        # Use the same parameters as edit_image_from_pil
        if pipeline_type == "qwen":
            params = {
                "image": img,
                "prompt": prompt,
                "generator": torch.manual_seed(kwargs.get("seed", 0)),
                "true_cfg_scale": kwargs.get("true_cfg_scale", 4.0),
                "negative_prompt": kwargs.get("negative_prompt", ""),
                "num_inference_steps": kwargs.get("num_inference_steps", 50),
            }
        elif pipeline_type == "instructpix2pix":
            params = {
                "image": img,
                "prompt": prompt,
                "generator": torch.manual_seed(kwargs.get("seed", 0)),
                "guidance_scale": kwargs.get("true_cfg_scale", 7.5),
                "image_guidance_scale": kwargs.get("image_guidance_scale", 1.5),
                "negative_prompt": kwargs.get("negative_prompt", ""),
                "num_inference_steps": kwargs.get("num_inference_steps", 50),
            }
        elif pipeline_type == "img2img":
            params = {
                "image": img,
                "prompt": prompt,
                "generator": torch.manual_seed(kwargs.get("seed", 0)),
                "guidance_scale": kwargs.get("true_cfg_scale", 7.5),
                "negative_prompt": kwargs.get("negative_prompt", ""),
                "num_inference_steps": kwargs.get("num_inference_steps", 50),
                "strength": kwargs.get("strength", 0.8),
            }
        else:
            raise RuntimeError("No pipeline available")
        
        # Remove non-pipeline parameters
        params.pop("seed", None)
        
        print(f"🎨 Editing image with {pipeline_type} pipeline")
        print(f"🎯 Prompt: '{prompt}'")
        print(f"⚙️  Steps: {params.get('num_inference_steps', 'N/A')}")
        
        # Run inference
        with torch.inference_mode():
            output = pipeline(**params)
        
        # Get the edited image
        edited_img = output.images[0]
        print("✅ Image generation completed")
        
        # Convert to base64
        buffer = BytesIO()
        edited_img.save(buffer, format="PNG")
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        print("📤 Image converted to base64")
        return img_str
        
    except Exception as e:
        print(f"❌ Error in edit_image_return_base64: {str(e)}")
        print(traceback.format_exc())
        raise e

# Legacy compatibility function
def edit_image_from_url(image_url: str, prompt: str, **kwargs) -> str:
    """Edit image from URL - legacy compatibility"""
    try:
        import requests
        print(f"🌐 Downloading image from: {image_url}")
        response = requests.get(image_url, timeout=30)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert("RGB")
        return edit_image_from_pil(img, prompt, **kwargs)
    except Exception as e:
        print(f"❌ Error downloading image: {str(e)}")
        raise e

if __name__ == "__main__":
    print(f"🧪 Testing {pipeline_type} pipeline...")
    print(f"Device: {device}")
    if pipeline_type == "qwen":
        print(f"Model dtype: {pipeline.unet.dtype}")
    print("✅ Pipeline test completed")
