from flask import Flask, render_template, request, jsonify
import torch
import base64
import io
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, AutoencoderKL
# from generation import avatar_generation as avatar_generation
import avatar_generation
import random

# Initialize the Flask application and configurations
app = Flask(__name__)

caption_path = "Salesforce/blip-image-captioning-large"
caption_text = ""
control_net_path = "diffusers/controlnet-canny-sdxl-1.0"
vae_path = "madebyollin/sdxl-vae-fp16-fix"
diffuser_path = "stabilityai/stable-diffusion-xl-base-1.0"
device = "cuda"

images_to_generate = 4

def make_generators(count):
    seeds = [random.randint(0, 1000) for _ in range(count)]
    generators = [torch.Generator(device=device).manual_seed(i) for i in seeds]
    # for i in range(count):
    #     generators.append(random.randint(0, 100))
    return seeds, generators

seeds, generators = make_generators(images_to_generate)


# generators = 

# Initialize models outside routes
cap_processor = BlipProcessor.from_pretrained(caption_path)
cap_model = BlipForConditionalGeneration.from_pretrained(caption_path, torch_dtype=torch.float16).to(device)
controlnet = ControlNetModel.from_pretrained(control_net_path, torch_dtype=torch.float16)
vae = AutoencoderKL.from_pretrained(vae_path, torch_dtype=torch.float16)
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(diffuser_path, controlnet=controlnet, vae=vae, torch_dtype=torch.float16)
pipe.enable_model_cpu_offload()




@app.route('/')
def index():
    return render_template('index.html')

@app.route('/transform', methods=['POST'])
def transform_image():
    try:
        seeds, generators = make_generators(images_to_generate)
        
        
        data = request.json
        image_data = base64.b64decode(data['imageBase64'])
        image = Image.open(io.BytesIO(image_data))
        controlnet_conditioning_scale = float(data.get('intensityLevel'))  
        # Convert RGBA to RGB with a white background
        if image.mode == 'RGBA':
            background = Image.new("RGB", image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[3])  # 3 is the alpha channel
            image = background  # Now 'image' is an RGB image
        image = image.resize((1024, 1024))

        prompt = data['inputText']
        negative_prompt = "soft line, lowres, text, sketch, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, artist name, blurry, ugly, logo, pixelated, oversharpened, high contrast"
        print('PROMPT: ', prompt)
        if len(prompt) == 0:
            prompt = avatar_generation.caption_image(cap_processor, cap_model, image, text=caption_text, device=device)
        print(prompt)
        prompt = "8k, RAW photo, best quality, masterpiece, highly detailed, realistic style, photo-realistic, uhd, DSLR, soft lighting, film grain, high dynamic range, an avatar of a  " + prompt
        result_image = avatar_generation.generate_avatar(pipe, prompt, controlnet_conditioning_scale, image, negative_prompt=negative_prompt)
        result_image.save("result_image.jpg")  # You mentioned you don't want to save the new image

        buffered = io.BytesIO()
        result_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue())
        
        return jsonify({'result_image': img_str.decode('utf-8')})

    except Exception as e:
        print(f'error: {str(e)}')
        return jsonify({'error': str(e)})
    
@app.route('/modify', methods=['POST'])
def modify_image():
    data = request.json
    num = data['num']
    
    

if __name__ == '__main__':
    port = 5001
    app.run(debug=True, port=port)