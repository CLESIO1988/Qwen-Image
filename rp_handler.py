import runpod
import os
from inference import edit_image  # your repo’s function for editing

def handler(job):
    """
    job: {
        "input": {
            "image_url": "https://example.com/img.jpg",
            "prompt": "Make the shirt red"
        }
    }
    """
    inp = job["input"]

    image_url = inp.get("image_url")
    prompt = inp.get("prompt")

    if not image_url or not prompt:
        return {"error": "Both image_url and prompt are required."}

    try:
        # Run the image edit (your implementation inside inference.py)
        output_path = edit_image(image_url, prompt)

        # TODO: Upload output_path to storage (e.g. S3, RunPod volume)
        # For now just return local path
        return {"output_image": output_path}

    except Exception as e:
        return {"error": str(e)}

# Start serverless
runpod.serverless.start({"handler": handler})
