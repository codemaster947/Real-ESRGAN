# Real-ESRGAN
PyTorch implementation of a Real-ESRGAN model trained on custom dataset. This model shows better results on faces compared to the original version. It is also easier to integrate this model into your projects.

> This is not an official implementation. We partially use code from the [original repository](https://github.com/xinntao/Real-ESRGAN)

Real-ESRGAN is an upgraded [ESRGAN](https://arxiv.org/abs/1809.00219) trained with pure synthetic data is capable of enhancing details while removing annoying artifacts for common real-world images. 

You can try it in [google colab](https://colab.research.google.com/drive/1YlWt--P9w25JUs8bHBOuf8GcMkx-hocP?usp=sharing) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1YlWt--P9w25JUs8bHBOuf8GcMkx-hocP?usp=sharing)

- [Paper (Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data)](https://arxiv.org/abs/2107.10833)
- [Original implementation](https://github.com/xinntao/Real-ESRGAN)
- [Huggingface 🤗](https://huggingface.co/sberbank-ai/Real-ESRGAN)

### Installation

```bash
pip install git+https://github.com/sberbank-ai/Real-ESRGAN.git
```

### Usage

---

Basic usage:

```python
import torch
from PIL import Image
import numpy as np
from RealESRGAN import RealESRGAN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = RealESRGAN(device, 'realesr-general-x4v3')
model.load_weights(download=True)

path_to_image = 'inputs/image.png'
image = Image.open(path_to_image)

sr_image = model.predict(image)

sr_image.save('results/image.png')
```

### Examples

---

Low quality image:

![](inputs/image.png)

Real-ESRGAN result:

![](results/image.png)

---

Low quality image:

![](inputs/face.png)

Real-ESRGAN result:

![](results/face.png)

---

Low quality image:

![](inputs/lion.png)

Real-ESRGAN result:

![](results/lion.png)
---

Low quality image:

![](inputs/0014.jpg)

Real-ESRGAN result:

![](results/0014.jpg)

---

Low quality image:

![](inputs/children-alpha.png)

Real-ESRGAN result:

![](results/children-alpha.png)

---

Low quality image:

![](inputs/logo.jpg)

Real-ESRGAN result:

![](results/logo.jpg)