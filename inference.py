import os
import uuid
import torch
from diffusers import QwenImageEditPipeline
from PIL import Image
import requests
from io import BytesIO
import base64
import traceback

print("🚀 Loading Qwen Image Edit model...")

# Load model globally (saves GPU warmup time)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"📱 Using device: {device}")

try:
    pipeline = QwenImageEditPipeline.from_pretrained(
        "Qwen/Qwen-Image-Edit",
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
        cache_dir="/tmp/cache"
    )
    pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)  # Disable progress bar for cleaner logs
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {str(e)}")
    print(traceback.format_exc())
    raise e

def edit_image_from_url(image_url: str, prompt: str, **kwargs) -> str:
    """
    Edit image from URL using Qwen-Image-Edit pipeline
    
    Args:
        image_url: URL of the image to edit
        prompt: Text description of the desired edit
        **kwargs: Additional parameters for the pipeline
    
    Returns:
        str: Path to the saved edited image
    """
    try:
        print(f"🌐 Downloading image from: {image_url}")
        # Download the image
        response = requests.get(image_url, timeout=30)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert("RGB")
        print(f"📐 Image loaded: {img.size}")
        
        return edit_image_from_pil(img, prompt, **kwargs)
        
    except Exception as e:
        print(f"❌ Error in edit_image_from_url: {str(e)}")
        print(traceback.format_exc())
        raise e

def edit_image_from_base64(image_b64: str, prompt: str, **kwargs) -> str:
    """
    Edit image from base64 string using Qwen-Image-Edit pipeline
    
    Args:
        image_b64: Base64 encoded image
        prompt: Text description of the desired edit
        **kwargs: Additional parameters for the pipeline
    
    Returns:
        str: Path to the saved edited image
    """
    try:
        print(f"🔤 Decoding base64 image...")
        # Decode base64 image
        image_bytes = base64.b64decode(image_b64)
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        print(f"📐 Image loaded: {img.size}")
        
        return edit_image_from_pil(img, prompt, **kwargs)
        
    except Exception as e:
        print(f"❌ Error in edit_image_from_base64: {str(e)}")
        print(traceback.format_exc())
        raise e

def edit_image_from_pil(img: Image.Image, prompt: str, **kwargs) -> str:
    """
    Edit PIL Image using Qwen-Image-Edit pipeline
    
    Args:
        img: PIL Image object
        prompt: Text description of the desired edit
        **kwargs: Additional parameters for the pipeline
    
    Returns:
        str: Path to the saved edited image
    """
    try:
        # Default parameters (can be overridden by kwargs)
        default_params = {
            "image": img,
            "prompt": prompt,
            "generator": torch.manual_seed(kwargs.get("seed", 0)),
            "true_cfg_scale": kwargs.get("true_cfg_scale", 4.0),
            "negative_prompt": kwargs.get("negative_prompt", ""),
            "num_inference_steps": kwargs.get("num_inference_steps", 50),
        }
        
        # Update with any additional kwargs
        inputs = {**default_params, **kwargs}
        # Remove non-pipeline parameters
        inputs.pop("seed", None)
        inputs.pop("return_base64", None)  # Custom parameter for output format
        
        print(f"🎨 Editing image with prompt: '{prompt}'")
        print(f"⚙️  Parameters: cfg_scale={inputs['true_cfg_scale']}, steps={inputs['num_inference_steps']}")
        
        # Run inference
        with torch.inference_mode():
            output = pipeline(**inputs)
        
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
    """
    Edit PIL Image and return as base64 string
    
    Args:
        img: PIL Image object
        prompt: Text description of the desired edit
        **kwargs: Additional parameters for the pipeline
    
    Returns:
        str: Base64 encoded edited image
    """
    try:
        # Default parameters (can be overridden by kwargs)
        default_params = {
            "image": img,
            "prompt": prompt,
            "generator": torch.manual_seed(kwargs.get("seed", 0)),
            "true_cfg_scale": kwargs.get("true_cfg_scale", 4.0),
            "negative_prompt": kwargs.get("negative_prompt", ""),
            "num_inference_steps": kwargs.get("num_inference_steps", 50),
        }
        
        # Update with any additional kwargs
        inputs = {**default_params, **kwargs}
        # Remove non-pipeline parameters
        inputs.pop("seed", None)
        
        print(f"🎨 Editing image with prompt: '{prompt}'")
        print(f"⚙️  Parameters: cfg_scale={inputs['true_cfg_scale']}, steps={inputs['num_inference_steps']}")
        
        # Run inference
        with torch.inference_mode():
            output = pipeline(**inputs)
        
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

# For backward compatibility
def edit_image(image_url: str, prompt: str, **kwargs) -> str:
    """Legacy function - use edit_image_from_url instead"""
    return edit_image_from_url(image_url, prompt, **kwargs)

if __name__ == "__main__":
    # Test the model loading
    print("🧪 Testing model...")
    print(f"Device: {device}")
    print(f"Model dtype: {pipeline.unet.dtype}")
    print("✅ Model test completed")
