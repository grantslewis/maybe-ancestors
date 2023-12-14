import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

from flask import Flask, render_template, request, jsonify
import torch
import base64
import io
from io import BytesIO
from PIL import Image
# from transformers import BlipProcessor, BlipForConditionalGeneration
# from diffusers import DiffusionPipeline, StableDiffusionXLControlNetPipeline, ControlNetModel, AutoencoderKL
from diffusers import StableDiffusionXLPipeline, AutoencoderKL, StableDiffusionXLImg2ImgPipeline
# from generation import avatar_generation as avatar_generation
# import avatar_generation
import random

# Initialize the Flask application and configurations
app = Flask(__name__)

caption_path = "Salesforce/blip-image-captioning-large"
caption_text = ""
# control_net_path = "diffusers/controlnet-canny-sdxl-1.0"
VAE_PATH = "madebyollin/sdxl-vae-fp16-fix"
DIFFUSER_PATH = "stabilityai/stable-diffusion-xl-base-1.0"
DEVICE = "cuda:0"






IMAGE_COUNT = 4
INITIAL_STEPS = 10
HIGH_NOISE_FRAC = 0.7

PROMPT_IMPROVMENT = "8k, RAW photo, best quality, masterpiece, highly detailed, realistic style, uhd, DSLR, soft lighting, film grain, high dynamic range," # photo-realistic, 
PROMPT_BASE = "A headshot of a"

NEGATIVE_PROMPT = "soft line, lowres, text, sketch, bad hands, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, artist name, blurry, ugly, logo, pixelated, oversharpened, high contrast, NSFW"



def make_generators(count):
    seeds = [random.randint(0, 1000) for _ in range(count)]
    generators = [torch.Generator(device=DEVICE).manual_seed(i) for i in seeds]
    # for i in range(count):
    #     generators.append(random.randint(0, 100))
    return seeds, generators

seeds, generators = make_generators(IMAGE_COUNT)


# generators = 

# Initialize models outside routes
# cap_processor = BlipProcessor.from_pretrained(caption_path)
# cap_model = BlipForConditionalGeneration.from_pretrained(caption_path, torch_dtype=torch.float16).to(device)
# controlnet = ControlNetModel.from_pretrained(control_net_path, torch_dtype=torch.float16)
vae = AutoencoderKL.from_pretrained(VAE_PATH, torch_dtype=torch.float16)
pipe = StableDiffusionXLPipeline.from_pretrained(DIFFUSER_PATH, vae=vae, torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
pipe = pipe.to(DEVICE)
# pipe = StableDiffusionXLControlNetPipeline.from_pretrained(DIFFUSER_PATH, controlnet=controlnet, vae=vae, torch_dtype=torch.float16)
pipe.enable_model_cpu_offload()

refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16)
refiner = refiner.to(DEVICE)




@app.route('/')
def index():
    return render_template('index.html')

@app.route('/transform', methods=['POST'])
def transform_image():
    try:
        seeds, generators = make_generators(IMAGE_COUNT)
        data = request.json
        
        print(request)
        # data = request.get_json()
        print({key: val for key, val in data.items() if key != "image"})
        print('includes image' if 'image' in data else 'does not include image')
    
        # Extract base64 image data
        image_data = None
        if 'image' in data:
            image_data = data['image']
            if "base64," in image_data:
                image_data = image_data.split("base64,")[1]
            # Decode the base64 string
            image_bytes = base64.b64decode(image_data)

            # Convert bytes data to a PIL Image
            image = Image.open(BytesIO(image_bytes))

            # You can now use this image object for your purposes, e.g., processing, saving, etc.
            # Example: Saving the image
            image.save("uploaded_image.png")
        
        # prompt = data['inputText']
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
        
        # prompt
        
        images = pipe(prompt=prompt, num_inference_steps=INITIAL_STEPS, denoising_end=HIGH_NOISE_FRAC, num_images_per_prompt=IMAGE_COUNT, output_type="latent", negative_prompt=NEGATIVE_PROMPT).images
        updated_images = []
        ret_info = dict()
        for i, image in enumerate(images):
                # image.save("result_image.jpg")
            image = refiner(prompt=prompt, num_inference_steps=INITIAL_STEPS, denoising_start=HIGH_NOISE_FRAC, image=image, negative_prompt=NEGATIVE_PROMPT).images[0]
            updated_images.append(image)
            image_name = f'result_image_{i + 1}'
            image.save(f"{image_name}.jpg")
            
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()) 
            
            ret_info[f"{image_name}"] = img_str.decode('utf-8')
        return jsonify({'result_images': ret_info, 'prompt': prompt_base})
        
        
        
        
        # # image_data = base64.b64decode(data['imageBase64'])
        # # image = Image.open(io.BytesIO(image_data))
        # controlnet_conditioning_scale = float(data.get('intensityLevel'))  
        # # Convert RGBA to RGB with a white background
        # if image.mode == 'RGBA':
        #     background = Image.new("RGB", image.size, (255, 255, 255))
        #     background.paste(image, mask=image.split()[3])  # 3 is the alpha channel
        #     image = background  # Now 'image' is an RGB image
        # image = image.resize((1024, 1024))

        # prompt = data['inputText']
        # print('PROMPT: ', prompt)
        # if len(prompt) == 0:
        #     prompt = avatar_generation.caption_image(cap_processor, cap_model, image, text=caption_text, device=DEFVICE)
        # print(prompt)
        # result_image = avatar_generation.generate_avatar(pipe, prompt, controlnet_conditioning_scale, image, negative_prompt=negative_prompt)
#         result_image.save("result_image.jpg")  # You mentioned you don't want to save the new image

#         buffered = io.BytesIO()
#         result_image.save(buffered, format="PNG")
#         img_str = base64.b64encode(buffered.getvalue())
        
#         return jsonify({'result_image': img_str.decode('utf-8')})

    except Exception as e:
        print(f'error: {str(e)}')
        return jsonify({'error': str(e)})
    
@app.route('/modify', methods=['POST'])
def modify_image():
    data = request.json
    num = data['num']
    
    

if __name__ == '__main__':
    port = 5003
    app.run(debug=True, port=port)