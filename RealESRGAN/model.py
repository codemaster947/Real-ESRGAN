import os
import torch
from torch.nn import functional as F
from PIL import Image
import numpy as np
import urllib.request
from huggingface_hub import hf_hub_download

from .rrdbnet_arch import RRDBNet
from .srvgg_arch import SRVGGNetCompact
from .utils import pad_reflect, split_image_into_overlapping_patches, stich_together, \
                   unpad_image


MODELS = {
    'RealESRGAN_x2': dict(
        repo_id='ai-forever/Real-ESRGAN', filename='RealESRGAN_x2.pth',
        scale=2, num_block=23, model_type='RRDBNet'
    ),
    'RealESRGAN_x4': dict(
        repo_id='ai-forever/Real-ESRGAN', filename='RealESRGAN_x4.pth',
        scale=4, num_block=23, model_type='RRDBNet'
    ),
    'RealESRGAN_x8': dict(
        repo_id='ai-forever/Real-ESRGAN', filename='RealESRGAN_x8.pth',
        scale=8, num_block=23, model_type='RRDBNet'
    ),
    'RealESRGAN_x4plus': dict(
        url='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
        scale=4, num_block=23, model_type='RRDBNet'
    ),
    'RealESRGAN_x4plus_anime_6B': dict(
        url='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth',
        scale=4, num_block=6, model_type='RRDBNet'
    ),
    'realesr-general-x4v3': dict(
        url='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth',
        scale=4, num_conv=32, model_type='SRVGGNetCompact'
    ),
    'realesr-animevideov3': dict(
        url='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth',
        scale=4, num_conv=16, model_type='SRVGGNetCompact'
    ),
}


class RealESRGAN:
    def __init__(self, device, model_name='RealESRGAN_x4'):
        assert model_name in MODELS, f'Invalid model name: {model_name}'
        self.device = device
        self.model_name = model_name
        self.config = MODELS[model_name]
        self.scale = self.config['scale']

        if self.config['model_type'] == 'SRVGGNetCompact':
            self.model = SRVGGNetCompact(
                num_in_ch=3, num_out_ch=3, num_feat=64,
                num_conv=self.config['num_conv'], upscale=self.scale, act_type='prelu'
            )
        else:
            self.model = RRDBNet(
                num_in_ch=3, num_out_ch=3, num_feat=64,
                num_block=self.config['num_block'], num_grow_ch=32, scale=self.scale
            )
        
    def load_weights(self, download=True):
        weights_dir = 'weights'
        os.makedirs(weights_dir, exist_ok=True)
        model_path = os.path.join(weights_dir, f'{self.model_name}.pth')

        if not os.path.exists(model_path) and download:
            if 'repo_id' in self.config:
                print(f'Downloading weights for {self.model_name} from Hugging Face...')
                model_path = hf_hub_download(repo_id=self.config['repo_id'], filename=self.config['filename'], local_dir=weights_dir)
            else:
                print(f'Downloading weights for {self.model_name} using urllib...')
                urllib.request.urlretrieve(self.config['url'], model_path)
            print(f'Weights downloaded to: {model_path}')

        loadnet = torch.load(model_path, map_location=self.device, weights_only=True)
        if 'params' in loadnet:
            self.model.load_state_dict(loadnet['params'], strict=True)
        elif 'params_ema' in loadnet:
            self.model.load_state_dict(loadnet['params_ema'], strict=True)
        else:
            self.model.load_state_dict(loadnet, strict=True)

        self.model.eval()
        self.model.to(self.device)
        
    def predict(self, lr_image, batch_size=4, patches_size=192,
                padding=24, pad_size=15):
        torch.autocast(device_type=self.device.type)
        scale = self.scale
        device = self.device
        lr_image = np.array(lr_image)
        lr_image = pad_reflect(lr_image, pad_size)

        patches, p_shape = split_image_into_overlapping_patches(
            lr_image, patch_size=patches_size, padding_size=padding
        )
        img = torch.FloatTensor(patches/255).permute((0,3,1,2)).to(device).detach()

        with torch.no_grad():
            res = self.model(img[0:batch_size])
            for i in range(batch_size, img.shape[0], batch_size):
                res = torch.cat((res, self.model(img[i:i+batch_size])), 0)

        sr_image = res.permute((0,2,3,1)).cpu().clamp_(0, 1)
        np_sr_image = sr_image.numpy()

        padded_size_scaled = tuple(np.multiply(p_shape[0:2], scale)) + (3,)
        scaled_image_shape = tuple(np.multiply(lr_image.shape[0:2], scale)) + (3,)
        np_sr_image = stich_together(
            np_sr_image, padded_image_shape=padded_size_scaled, 
            target_shape=scaled_image_shape, padding_size=padding * scale
        )
        sr_img = (np_sr_image*255).astype(np.uint8)
        sr_img = unpad_image(sr_img, pad_size*scale)
        sr_img = Image.fromarray(sr_img)

        return sr_img