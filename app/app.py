from flask import Flask, render_template, request, jsonify
import torch
import base64
import io
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, AutoencoderKL
# from generation import avatar_generation as avatar_generation
import avatar_generation

# Initialize the Flask application and configurations
app = Flask(__name__)

caption_path = "Salesforce/blip-image-captioning-large"
caption_text = ""
control_net_path = "diffusers/controlnet-canny-sdxl-1.0"
vae_path = "madebyollin/sdxl-vae-fp16-fix"
diffuser_path = "stabilityai/stable-diffusion-xl-base-1.0"
device = "cuda"

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

        prompt = avatar_generation.caption_image(cap_processor, cap_model, image, text=caption_text, device=device)
        print(prompt)
        prompt = "HD, 4k, Masterpiece, High Quality avatar inspired by a " + prompt
        result_image = avatar_generation.generate_avatar(pipe, prompt, controlnet_conditioning_scale, image)
        result_image.save("result_image.jpg")  # You mentioned you don't want to save the new image

        buffered = io.BytesIO()
        result_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue())
        
        return jsonify({'result_image': img_str.decode('utf-8')})

    except Exception as e:
        print(f'error: {str(e)}')
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    port = 5001
    app.run(debug=True, port=port)