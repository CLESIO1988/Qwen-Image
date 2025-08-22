import os
import uuid
from qwen_image_edit import QwenImageEditor  # adjust to your repo

# Load once globally (saves GPU warmup time)
model = QwenImageEditor.from_pretrained("Qwen/Qwen-Image-Edit", device="cuda")

def edit_image(image_url: str, prompt: str) -> str:
    # Download the image
    import requests
    from PIL import Image
    from io import BytesIO

    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))

    # Run model
    result = model.edit(img, prompt)

    # Save locally
    out_path = f"/app/outputs/{uuid.uuid4()}.png"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    result.save(out_path)

    return out_path
