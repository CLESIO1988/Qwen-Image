import os
import uuid
import torch
from diffusers import QwenImageEditPipeline
from PIL import Image
import requests
from io import BytesIO

# Load model globally (saves GPU warmup time)
device = "cuda" if torch.cuda.is_available() else "cpu"
pipeline = QwenImageEditPipeline.from_pretrained("Qwen/Qwen-Image-Edit")
pipeline.to(torch.bfloat16)
pipeline.to(device)
pipeline.set_progress_bar_config(disable=None)

def edit_image(image_url: str, prompt: str, **kwargs) -> str:
    """
    Edit image using Qwen-Image-Edit pipeline
    
    Args:
        image_url: URL of the image to edit
        prompt: Text description of the desired edit
        **kwargs: Additional parameters for the pipeline
    """
    try:
        # Download the image
        response = requests.get(image_url, timeout=30)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert("RGB")
        
        # Default parameters (can be overridden by kwargs)
        default_params = {
            "image": img,
            "prompt": prompt,
            "generator": torch.manual_seed(kwargs.get("seed", 0)),
            "true_cfg_scale": kwargs.get("true_cfg_scale", 4.0),
            "negative_prompt": kwargs.get("negative_prompt", " "),
            "num_inference_steps": kwargs.get("num_inference_steps", 50),
        }
        
        # Update with any additional kwargs
        inputs = {**default_params, **kwargs}
        # Remove non-pipeline parameters
        inputs.pop("seed", None)
        
        print(f"Editing image with prompt: {prompt}")
        print(f"Parameters: cfg_scale={inputs['true_cfg_scale']}, steps={inputs['num_inference_steps']}")
        
        # Run inference
        with torch.inference_mode():
            output = pipeline(**inputs)
        
        # Get the edited image
        edited_img = output.images[0]
        
        # Save the result
        out_path = f"/app/outputs/{uuid.uuid4()}.png"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        edited_img.save(out_path)
        
        print(f"Image saved to: {out_path}")
        return out_path
        
    except Exception as e:
        print(f"Error in edit_image: {str(e)}")
        raise e