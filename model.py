from transformers import pipeline as hf_pipeline
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
import torch
from PIL import Image
import numpy as np
import io
import random
from torchvision.transforms import ToTensor


# Constants
CONTROL_NET_MODEL = "lllyasviel/control_v11f1p_sd15_depth"
SD_MODEL = "runwayml/stable-diffusion-v1-5"

INFERENCE_STEPS = 50
PROMPT_IMPROVMENT = "8k, RAW photo, best quality, masterpiece, highly detailed, realistic style, uhd, DSLR, soft lighting, film grain, high dynamic range, Historical Portrait"

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

def prompt_builder(data):
    prompt = "A portrait of a young person named "
    if "name" in data and data["name"]:
        prompt += f" {data['name']},"
    
    if "occupation" in data and data["occupation"]:
        prompt += f" a {data['occupation']},"
    
    if "culture" in data and data["culture"]:
        prompt += f" from {data['culture']},"
    
    if "dateOrDescription" in data and data["dateOrDescription"]:
        prompt += f" in the {data['dateOrDescription']},"
    
    prompt += f" {PROMPT_IMPROVMENT}"
    return prompt

def depth_processor(image):
    depth_estimator = hf_pipeline('depth-estimation')
    depth = depth_estimator(image)['depth']
    depth = np.array(depth)
    depth = depth[:, :, None]
    depth = np.concatenate([depth, depth, depth], axis=2)
    depth_image = Image.fromarray(np.uint8(depth))  # Convert to PIL Image
    depth_image = depth_image.resize((512, 512))  # Resize
    return depth_image

class ImageProcessor:
    def __init__(self):
        self.control_net = ControlNetModel.from_pretrained(CONTROL_NET_MODEL, torch_dtype=torch.float16)
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(SD_MODEL, controlnet=self.control_net, torch_dtype=torch.float16, safety_checker=None)
        self.pipe.to(DEVICE)

    def run(self, data, uploaded_image=None):
        if uploaded_image is None:
            raise ValueError("No image uploaded for processing")

        depth_map = depth_processor(uploaded_image)
        prompt = prompt_builder(data)
        print("Running on", DEVICE)
        print("Generated Prompt:", prompt)

        all_images = []
        for _ in range(4):  # Generate 4 images
            # Each iteration uses a different random seed
            generator = torch.Generator(device=DEVICE).manual_seed(random.randint(0, 10000))
            generated_images = self.pipe(prompt=prompt, image=depth_map, negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality", num_inference_steps=INFERENCE_STEPS, generator=generator).images
            all_images.extend(generated_images)

        return all_images
