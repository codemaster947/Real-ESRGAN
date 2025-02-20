import os 
import torch
from PIL import Image
import numpy as np
from RealESRGAN import RealESRGAN

input_dir = "inputs"
output_dir = "results"

def main() -> int:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RealESRGAN(device, 'RealESRGAN_x4')
    model.load_weights(download=True)
    
    #image = Image.open(f"inputs/lr_image.png").convert('RGB')
    #sr_image = model.predict(image)
    #sr_image.save(f'results/sr_image.png')

    for filename in os.listdir(input_dir):
        image = Image.open(os.path.join(input_dir, filename)).convert('RGB')
        sr_image = model.predict(image)
        sr_image.save(os.path.join(output_dir, filename))

if __name__ == '__main__':
    main()