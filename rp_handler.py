import runpod
import os
import traceback
from inference import edit_image

def handler(job):
    """
    RunPod handler for Qwen-Image-Edit
    
    Expected input format:
    {
        "input": {
            "image_url": "https://example.com/img.jpg",
            "prompt": "Make the shirt red",
            "seed": 42,  # optional
            "true_cfg_scale": 4.0,  # optional
            "num_inference_steps": 50,  # optional
            "negative_prompt": " "  # optional
        }
    }
    """
    try:
        inp = job.get("input", {})
        
        # Required parameters
        image_url = inp.get("image_url")
        prompt = inp.get("prompt")
        
        if not image_url or not prompt:
            return {
                "error": "Both 'image_url' and 'prompt' are required.",
                "status": "failed"
            }
        
        print(f"Processing job with prompt: {prompt}")
        print(f"Image URL: {image_url}")
        
        # Optional parameters
        optional_params = {
            "seed": inp.get("seed", 0),
            "true_cfg_scale": inp.get("true_cfg_scale", 4.0),
            "num_inference_steps": inp.get("num_inference_steps", 50),
            "negative_prompt": inp.get("negative_prompt", " ")
        }
        
        # Run the image edit
        output_path = edit_image(image_url, prompt, **optional_params)
        
        return {
            "output_image": output_path,
            "status": "completed",
            "prompt_used": prompt,
            "parameters": optional_params
        }
        
    except requests.exceptions.RequestException as e:
        error_msg = f"Failed to download image: {str(e)}"
        print(error_msg)
        return {"error": error_msg, "status": "failed"}
        
    except torch.cuda.OutOfMemoryError as e:
        error_msg = f"GPU out of memory: {str(e)}"
        print(error_msg)
        return {"error": error_msg, "status": "failed"}
        
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        print(error_msg)
        print(traceback.format_exc())
        return {"error": error_msg, "status": "failed"}

# Start serverless
runpod.serverless.start({"handler": handler})