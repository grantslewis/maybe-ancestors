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
from model import ImageProcessor


# Initialize the Flask application and configurations
app = Flask(__name__)


model = ImageProcessor()

@app.route('/')
def index():
    return render_template('index.html')



@app.route('/transform', methods=['POST'])
def transform_image():
    try:
        data = request.json

        # Extract and print form data
        name = data.get('name')
        culture = data.get('culture')
        occupation = data.get('occupation')
        date_or_description = data.get('dateOrDescription')

        print(f'Name: {name}, Culture: {culture}, Occupation: {occupation}, Year: {date_or_description}')

        # Extract base64 image data
        image_data = data.get('image')
        
        # Processing the image data
        image_data = image_data.split("base64,")[1] if "base64," in image_data else image_data
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))

        result_images = model.run(data, image)

        encoded_images = {}
        for i, img in enumerate(result_images):
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
            encoded_images[f'result_image_{i + 1}'] = f'data:image/jpeg;base64,{img_str}'

        return jsonify({'result_images': encoded_images})
        
    except Exception as e:
        print(f'error: {str(e)}')
        return jsonify({'error': str(e)})
    
if __name__ == '__main__':
    port = 5003
    app.run(debug=True, port=port)