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
        
    def _upscale_ai(self, image_np, batch_size, patches_size, padding, pad_size):
        """
        Upscales a 3-channel image using the AI model.
        Assumes image_np is a numpy array with shape (H, W, 3).
        """
        # Apply reflection padding
        image_np = pad_reflect(image_np, pad_size)
        patches, p_shape = split_image_into_overlapping_patches(
            image_np, patch_size=patches_size, padding_size=padding
        )
        img = torch.FloatTensor(patches / 255).permute((0, 3, 1, 2)).to(self.device).detach()

        # Process the patches in batches through the model
        with torch.no_grad():
            res = self.model(img[0:batch_size])
            for i in range(batch_size, img.shape[0], batch_size):
                res = torch.cat((res, self.model(img[i:i+batch_size])), 0)

        # Reassemble the patches into a full image
        sr_image = res.permute((0, 2, 3, 1)).cpu().clamp_(0, 1)
        np_sr_image = sr_image.numpy()
        padded_size_scaled = tuple(np.multiply(p_shape[0:2], self.scale)) + (3,)
        scaled_image_shape = tuple(np.multiply(image_np.shape[0:2], self.scale)) + (3,)
        np_sr_image = stich_together(
            np_sr_image,
            padded_image_shape=padded_size_scaled,
            target_shape=scaled_image_shape,
            padding_size=padding * self.scale
        )
        final_img = (np_sr_image * 255).astype(np.uint8)
        final_img = unpad_image(final_img, pad_size * self.scale)
        return final_img

    def predict(self, lr_image, batch_size=4, patches_size=192,
                padding=24, pad_size=15, alpha_scale="Esrgan"):
        """
        Upscales an image. If an alpha channel is present, the RGB is processed through the AI upscaler,
        and the alpha channel is either upscaled via bicubic interpolation or passed through the same
        AI pipeline (if alpha_scale is 'Esrgan'). The upscaled alpha is then recombined.
        """
        torch.autocast(device_type=self.device.type)
        scale = self.scale
        lr_image_np = np.array(lr_image)

        # Check if there is an alpha channel and split it out
        if lr_image_np.ndim == 3 and lr_image_np.shape[2] == 4:
            has_alpha = True
            alpha_channel = lr_image_np[..., 3]
            lr_rgb = lr_image_np[..., :3]
        else:
            has_alpha = False
            lr_rgb = lr_image_np

        # Upscale the RGB image using the helper function
        sr_rgb = self._upscale_ai(lr_rgb, batch_size, patches_size, padding, pad_size)
        sr_rgb_img = Image.fromarray(sr_rgb)

        if has_alpha:
            if alpha_scale == "Bicubic":
                # Upscale alpha using bicubic interpolation
                alpha_img = Image.fromarray(alpha_channel).convert("L")
                new_size = (lr_rgb.shape[1] * scale, lr_rgb.shape[0] * scale)
                alpha_upscaled = alpha_img.resize(new_size, resample=Image.BICUBIC)
            elif alpha_scale == "Esrgan":
                # Replicate the single-channel alpha to three channels and upscale via the AI pipeline
                alpha_3ch = np.repeat(alpha_channel[..., np.newaxis], 3, axis=2)
                sr_alpha = self._upscale_ai(alpha_3ch, batch_size, patches_size, padding, pad_size)
                # Convert the upscaled alpha back to a single-channel grayscale image
                alpha_upscaled = Image.fromarray(sr_alpha).convert("L")
            else:
                raise ValueError("alpha_scale must be either 'Bicubic' or 'Esrgan'")
            # Combine the upscaled alpha with the upscaled RGB image
            sr_rgb_img.putalpha(alpha_upscaled)

        return sr_rgb_img