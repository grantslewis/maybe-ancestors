import torch
from PIL import Image
import cv2
import numpy as np

def caption_image(processor, model, image, text="", device="cuda"):
    inputs = processor(image, text, return_tensors="pt").to(device, torch.float16)
    out = model.generate(**inputs, max_new_tokens = 20)
    prompt = processor.decode(out[0], skip_special_tokens=True)
    return prompt

def canny_generation(image, t_lower=100, t_upper=200):
    image = np.array(image)
    image = cv2.Canny(image, t_lower, t_upper)
    return Image.fromarray(image, 'L')

def generate_avatar(pipe, prompt, controlnet_conditioning_scale, image, negative_prompt=''):
    control_image = canny_generation(image)
    generated_image = pipe(prompt, controlnet_conditioning_scale=controlnet_conditioning_scale, image=control_image, num_inference_steps = 20, negative_prompt=negative_prompt).images[0]
    return generated_image