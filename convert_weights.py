import os 
import torch
from PIL import Image
import numpy as np
from RealESRGAN import RealESRGAN
import exportsd

def save_to_torchsharp_format(input_dir, output_dir):
    """
    Converts all weight files in the input directory to TorchSharp-compatible format.
    Args:
        input_dir (str): Directory containing weight files (.pth).
        output_dir (str): Directory to save the converted weight files (.dat).
    """
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    for file_name in os.listdir(input_dir):
        if file_name.endswith(".pth"):  # Process only .pth files
            input_path = os.path.join(input_dir, file_name)
            input_name = os.path.splitext(file_name)[0]
            output_file = input_name + ".dat"  # Change .pth to .dat
            output_path = os.path.join(output_dir, output_file)
            
            print(f"Processing {input_path}...")

            # Load weights
            model = RealESRGAN(device, model_name=input_name)
            loadnet = torch.load(input_path, map_location=device, weights_only=True)

            if 'params' in loadnet:
                model.model.load_state_dict(loadnet['params'], strict=True)
            elif 'params_ema' in loadnet:
                model.model.load_state_dict(loadnet['params_ema'], strict=True)
            else:
                model.model.load_state_dict(loadnet, strict=True)

            f = open(output_path, "wb")
            exportsd.save_state_dict(model.model.to("cpu").state_dict(), f)
            f.close()
            
            print(f"Saved converted weights to {output_path}")

# Input and output directory paths
input_directory = "weights"  # Replace with your directory containing .pth files
output_directory = "converted_weights"  # Directory to save .dat files

save_to_torchsharp_format(input_directory, output_directory)