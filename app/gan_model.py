import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
# os.chdir('../')
CODE_DIR  = 'StyleGANEX'
# device = 'cuda'

from models.psp import pSp
from models.bisenet.model import BiSeNet

import torch
import dlib
import cv2
import PIL
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
import torchvision
from torchvision import transforms, utils
from argparse import Namespace
from datasets import augmentations
from huggingface_hub import hf_hub_download
from scripts.align_all_parallel import align_face
from latent_optimization import latent_optimization
from utils.inference_utils import save_image, load_image, visualize, get_video_crop_parameter, tensor2cv2, tensor2label, labelcolormap

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

HIGH = 0.35 # .45
MEDIUM = 0.10 # .25
LOW = 0.005 # .01
DEFLT = 0.09
DEFAULT = None
NONE = 0.00001#0


dim_scaling = {0 : MEDIUM,
    1 : HIGH,
    2 : NONE,
    3 : MEDIUM,
    4 : HIGH,
    5 : LOW,
    6 : HIGH,
    7 : LOW,
    8 : NONE,
    9 : NONE,
    10 : NONE,
    11 : NONE,
    12 : NONE,
    13 : NONE,
    14 : DEFAULT,
    15 : DEFAULT,
    16 : DEFAULT,
    17 : DEFAULT,
}



def load_model(path, device):
    local_path = hf_hub_download('PKUWilliamYang/StyleGANEX', path)
    ckpt = torch.load(local_path, map_location='cpu')
    opts = ckpt['opts']
    opts['checkpoint_path'] = local_path
    opts['device'] = device
    opts = Namespace(**opts)
    pspex = pSp(opts).to(device).eval()
    pspex.latent_avg = pspex.latent_avg.to(device)
    if 'editing_w' in ckpt.keys():
        return pspex, ckpt['editing_w'].clone().to(device)
    return pspex


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5,0.5,0.5]),
    ])

landmarkpredictor = dlib.shape_predictor(hf_hub_download('PKUWilliamYang/VToonify',
                                                                         'models/shape_predictor_68_face_landmarks.dat'))
path = 'pretrained_models/styleganex_inversion.pt'
pspex = load_model(path, DEVICE)


def prep_image(frame):
    with torch.no_grad():
        

        # Check if the image is loaded properly
        if frame is None:
            raise Exception("Failed to load image")
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        paras = get_video_crop_parameter(frame, landmarkpredictor)

        h,w,top,bottom,left,right,scale = paras
        H, W = int(bottom-top), int(right-left)
        frame = cv2.resize(frame, (w, h))[top:bottom, left:right]

        return frame, (h, w), (H, W)

def load_image(image_path):
    
    frame = None
    
    if not os.path.exists(image_path):
        raise Exception("Image file does not exist:" + image_path)
    frame = cv2.imread(image_path)
    
    frame, (h, w), (H, W) = prep_image(frame)
    return frame, (h, w), (H, W)
    




# def save_images(image_tensor, base_image_name='new_image_{}.png'):
#     # of the form n x 3 x h x w, where n is the number of photos to be saved - with values ranging from -1 to 1
#     np_images = image_tensor.to('cpu').numpy()
#     np_images = ((np_images + 1) * 0.5 * 255).astype(np.uint8).transpose(0,2,3,1)
    
#     # n = image_tensor.shape[0]
#     # for i in range(n):
#     for i, image_array in enumerate(np_images):
#         # print(image_array.shape)
#         image = Image.fromarray(image_array)
#         img_name = base_image_name.format(i)

#         # Saving the image
#         image.save(img_name)
#         print("Image saved as {}".format(img_name))
        
# def get_images(image_tensor):
#     # Convert the tensor to numpy array and adjust the range from [-1, 1] to [0, 255]
#     np_images = image_tensor.to('cpu').numpy()
#     np_images = ((np_images + 1) * 0.5 * 255).astype(np.uint8).transpose(0, 2, 3, 1)

#     image_list = []
#     for image_array in np_images:
#         # Convert numpy array to PIL Image
#         image = Image.fromarray(image_array)
#         image_list.append(image)

#     return image_list

def get_images(image_tensor):
    # Convert the tensor to numpy array and adjust the range from [-1, 1] to [0, 255]
    np_images = image_tensor.to('cpu').numpy()
    np_images = ((np_images + 1) * 0.5 * 255).astype(np.uint8).transpose(0, 2, 3, 1)

    image_list = []
    for image_array in np_images:
        # Convert numpy array to PIL Image
        image = Image.fromarray(image_array)
        image_list.append(image)

    return image_list

def save_images(image_tensor, base_image_name='new_image_{}.png'):
    image_list = get_images(image_tensor)

    for i, image in enumerate(image_list):
        img_name = base_image_name.format(i)
        image.save(img_name)
        print("Image saved as {}".format(img_name))
        

def image_variations(frame, pspex, num_vars=10, default_var_scale=0.17, dimensions=dict(), device='cuda'):
    
    def prod(vals):
        tot = 1
        for val in vals:
            tot *= val
        return tot
    
    wplus_hat, f_hat, noises_hat, wplus, f = latent_optimization(frame, pspex, landmarkpredictor, step=1, device=device)
    dim_size = prod(list(wplus.shape))
    
    if dimensions == 'all':
        dimensions = list(range(wplus.shape[1]))
        # dimensions = {dim : default_var_scale for dim in range(wplus.shape[1])}
    
    # if is_instance(dimensions, list):
    if isinstance(dimensions, list):
        dimensions = {dim : default_var_scale for dim in dimensions}
    
    dims_updated = {}
    for dim, val in dimensions.items():
        if val is None:
            dims_updated[dim] = default_var_scale
        else:
            dims_updated[dim] = val
    
    dimensions = dims_updated 
        
    # if is_instance(var_scale, float):
    #     var_scale = [var_scale]
    # if len(var_scale) < len(dimensions):
    #     last_var =
    #     var_scale = var_scale + []
    
    with torch.no_grad():
        combined_results = []
        for i in range(num_vars):
            # Initialize random noise
            
            
                
            
            random_noise = np.zeros_like(wplus.cpu().numpy())
            # Apply noise only to specified dimensions
            consistent_noise = np.random.normal(0, 1, wplus.shape[2])
            for dim, scale in dimensions.items():
                # random_noise[:, dim, :] = np.random.normal(0, var_scale, wplus.shape[2])
                random_noise[:, dim, :] = consistent_noise * scale
                

            torch_noise = torch.from_numpy(random_noise).to(device)
            y_init, _ = pspex.decoder([torch.add(wplus, torch_noise)], input_is_latent=True, first_layer_feature=f)
            combined_results.append(y_init.to('cpu'))

        combined_results = torch.cat(combined_results, dim=0)
        return combined_results
    
def main(image):
    prepped_image, (h, w), (H, W) = prep_image(image)
    
    combined_results = image_variations(frame, pspex, 6, default_var_scale=deflt, device=DEVICE, dimensions=dim_scaling)
    
    images = get_images(combined_results)
    return images
