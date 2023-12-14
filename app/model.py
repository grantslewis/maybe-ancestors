import torch
import matplotlib.pyplot as plt
import os
from huggingface_hub import HfApi
from pathlib import Path
from diffusers.utils import load_image
from PIL import Image
import numpy as np
from controlnet_aux import OpenposeDetector
import random
import cv2
from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    UniPCMultistepScheduler,
)

## GAN
# import gan_model

CONTROL_NET_MODEL = "lllyasviel/control_v11p_sd15_canny"
SD_MODEL = "runwayml/stable-diffusion-v1-5"

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
IMAGE_COUNT = 4
# INITIAL_STEPS = 10
INFERENCE_STEPS = 10
HIGH_NOISE_FRAC = 0.7
PROMPT_IMPROVMENT = "8k, RAW photo, best quality, masterpiece, highly detailed, realistic style, uhd, DSLR, soft lighting, film grain, high dynamic range," # photo-realistic,
PROMPT_BASE = "A headshot of a"
NEGATIVE_PROMPT = "soft line, lowres, text, sketch, bad hands, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, artist name, blurry, ugly, logo, pixelated, oversharpened, high contrast, NSFW"

SET_SEEDS = []

def prompt_builder(data):
    prompt = PROMPT_BASE #+ prompt
    if "culture" in data:
        prompt += f" {data['culture']}"
    
    if "gender" in data:
        prompt += f" {data['gender']}"
    else:
        prompt += " person"
    
    attributes = []
    if "eyeColor" in data:
        attributes.append(f"{data['eyeColor']} eyes")
        
    hair = []
    if "hairStyle" in data:
        hair.append(data['hairStyle'])
    if "hairColor" in data:
        hair.append(data['hairColor'])
    if len(hair) > 0:
        hair_parts = " ".join(hair)
        attributes.append(f"{hair_parts} hair")
    if "skinTone" in data:
        attributes.append(f"{data['skinTone']} skin")
    
    attributes_combined = ""
    if len(attributes) > 2:
        attributes_combined = ", ".join(attributes[:-1]) + ", and " + attributes[-1]
    # If the list has 3 or fewer items, just join them with spaces.
    elif len(attributes) == 2:
        attributes_combined = f"{attributes[0]} and {attributes[1]}"
    elif len(attributes) == 1:
        attributes_combined = attributes[0]
        # return ' '.join(attributes)
    if len(attributes) > 0:
        attributes_combined = f" with {attributes_combined}"
    prompt += attributes_combined
    
    if len(data["location"]) > 0:
        prompt += f" living {data['location']}"
    if len(data["occupation"]) > 0:
        prompt += f" working as a {data['occupation']}"
    if len(data["dateOrDescription"]) > 0:
        prompt += f" living during the {data['dateOrDescription']}"
    prompt += ". "
    
    if data["inputText"] is not None and len(data["inputText"]) > 0:
        prompt = ""
        
    prompt += data["inputText"]
    
    prompt_base = prompt
    prompt = f"{PROMPT_IMPROVMENT} {prompt}"
    print('PROMPT:', prompt)
    return prompt
    


def canny_processor(image, lower_thresh, higher_thresh):
    np_image = np.array(image)
    
    gray = cv2.cvtColor(np_image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, lower_thresh, higher_thresh)
    
    pil_image = None
    if len(control_image.shape) == 2:
        # Directly convert to PIL Image
        pil_image = Image.fromarray(control_image)
    else:
        # Convert from BGR to RGB
        control_image_rgb = cv2.cvtColor(control_image, cv2.COLOR_BGR2RGB)
        # Convert to PIL Image
        pil_image = Image.fromarray(control_image_rgb)
    return pil_image

# def control_net_processor(image, control_net):
#     control_image = canny_processor(image, 100, 200)
#     control_image = load_image(control_image, 256, 256)
#     control_image = control_image.unsqueeze(0).to(DEVICE)
#     control_image = control_net.encode_image(control_image)
#     return control_image


class ImageProcessor:
    def __init__(self):
        self.control_net = None
        self.pipe = None
        
        
        
        # self.refiner = None
        # self.vae = None
        self.start_model()
        

    def start_model(self):
        self.control_net = ControlNetModel.from_pretrained(CONTROL_NET_MODEL, torch_dtype=torch.float16)
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(SD_MODEL, control_net=self.control_net, torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
        self.pipe = self.pipe.to(DEVICE)
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        # self.pipe.scheduler = UniPCMultistepScheduler(self.pipe.model, 0.99, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01)
        self.pipe.enable_model_cpu_offload()
        
        
        # self.vae = AutoencoderKL.from_pretrained(self.VAE_PATH, torch_dtype=torch.float16)
        # self.pipe = StableDiffusionXLPipeline.from_pretrained(self.DIFFUSER_PATH, vae=self.vae, torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
        # self.pipe = self.pipe.to(self.DEVICE)
        # self.pipe.enable_model_cpu_offload()

        # self.refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16)
        # self.refiner = self.refiner.to(self.DEVICE)
    
    def ctrl_net_processor(self, image):
        print('TODO: control_net_processor')
        # control_image = canny_processor(image, 100, 200)
        return image
        

    def make_generators(self, count):
        seeds = SET_SEEDS
        if len(seeds) < count:
            seeds = seeds + [random.randint(0, 1000) for _ in range(count)]
        seeds = seeds[:count]
        
        generators = [torch.Generator(device=self.DEVICE).manual_seed(i) for i in seeds]
        return seeds, generators

    def run(self, prompt=None, image=None):
        if not self.pipe:
            raise Exception("Models are not initialized. Please call start_model first.")
        
        if not isinstance(prompt, str):
            prompt = prompt_builder(prompt)
        

        # seeds, generators = self.make_generators(self.IMAGE_COUNT)
        
        ## GAN
        # images = gan_model.main(image)
        
        ctrl_image = self.ctrl_net_processor(image)

        # # If an image is provided, use Img2Img pipeline, otherwise use text-to-image pipeline
        # if image:
        #     # Process the image here
            
        # else:
        #     # Process the prompt here
        #     pass

        # Your code to generate images goes here
        seeds, generators = self.make_generators(IMAGE_COUNT)
        images = self.pipe(prompt, num_inference_steps=INFERENCE_STEPS, generator=generator, image=ctrl_image).images
        

        # Return results
        return {
            "images": images,  # Replace with generated images
            "seeds": seeds
        }