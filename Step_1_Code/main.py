from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
import io
import base64

model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_single_file(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

def pil_to_b64(image: Image.Image) -> str:
    # Create a BytesIO object to hold the image data
    buffered = io.BytesIO()
    
    # Save the image to the BytesIO object in PNG format
    image.save(buffered, format="PNG")
    
    # Get the byte data from the BytesIO object
    img_bytes = buffered.getvalue()
    
    # Encode the byte data to base64
    img_base64 = base64.b64encode(img_bytes).decode("utf-8")
    
    return img_base64

from fastapi import FastAPI

app = FastAPI()

# Define a simple endpoint
@app.get("/")
def read_root():
    return {"message": "Hello, World!"}

@app.post("/api/gen_image")
def gen_image(prompt: str):
    image = pipe(prompt).images[0]  
    image.save(f"{prompt}.jpg")
    image_b64 = pil_to_b64(image)
    return image_b64

if __name__=='__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=80)