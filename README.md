
# DLCV Final Project: Multiple Concept Personalization

Our code structure and pipeline are primarily based on [Concept-Conductor](https://github.com/Nihukat/Concept-Conductor?tab=readme-ov-file).  

The pipeline first trains **ed-lora** for each concept and fuses them together during inference.


## Setup

### 1. Download Pretrained Models (will take for 20 minutes)
```bash
mkdir pretrained_models
cd pretrained_models
git lfs install
git clone https://huggingface.co/windwhinny/chilloutmix.git
git clone https://huggingface.co/xyn-ai/anything-v4.0.git
git clone https://huggingface.co/h94/IP-Adapter
git clone https://huggingface.co/Yntec/AbsoluteReality
git clone https://huggingface.co/lllyasviel/sd-controlnet-canny
```
*Note: Above repositories consume significant memory due to the presence of `.git` directories, duplicate checkpoints in various formats, and multiple versions of the checkpoints. If the download process uses excessive memory, consider removing unused files to free up space. Below is a list of files we used for reference. Thank you for your understanding!*
```
.
├── AbsoluteReality
│   ├── feature_extractor
│   │   └── preprocessor_config.json
│   ├── model_index.json
│   ├── README.md
│   ├── safety_checker
│   │   ├── config.json
│   │   └── model.safetensors
│   ├── scheduler
│   │   └── scheduler_config.json
│   ├── text_encoder
│   │   ├── config.json
│   │   └── model.safetensors
│   ├── tokenizer
│   │   ├── merges.txt
│   │   ├── special_tokens_map.json
│   │   ├── tokenizer_config.json
│   │   └── vocab.json
│   ├── unet
│   │   ├── config.json
│   │   └── diffusion_pytorch_model.safetensors
│   └── vae
│       ├── config.json
│       └── diffusion_pytorch_model.bin
├── anything-v4.0
│   ├── feature_extractor
│   │   └── preprocessor_config.json
│   ├── model_index.json
│   ├── safety_checker
│   │   ├── config.json
│   │   └── pytorch_model.bin
│   ├── scheduler
│   │   └── scheduler_config.json
│   ├── text_encoder
│   │   ├── config.json
│   │   └── pytorch_model.bin
│   ├── tokenizer
│   │   ├── merges.txt
│   │   ├── special_tokens_map.json
│   │   ├── tokenizer_config.json
│   │   └── vocab.json
│   ├── unet
│   │   ├── config.json
│   │   └── diffusion_pytorch_model.bin
│   └── vae
│       ├── config.json
│       └── diffusion_pytorch_model.bin
├── chilloutmix
│   ├── feature_extractor
│   │   └── preprocessor_config.json
│   ├── model_index.json
│   ├── safety_checker
│   │   ├── config.json
│   │   └── pytorch_model.bin
│   ├── scheduler
│   │   └── scheduler_config.json
│   ├── text_encoder
│   │   ├── config.json
│   │   └── pytorch_model.bin
│   ├── tokenizer
│   │   ├── merges.txt
│   │   ├── special_tokens_map.json
│   │   ├── tokenizer_config.json
│   │   └── vocab.json
│   ├── unet
│   │   ├── config.json
│   │   └── diffusion_pytorch_model.bin
│   └── vae
│       ├── config.json
│       └── diffusion_pytorch_model.bin
├── IP-Adapter
│   ├── models
│   │   ├── image_encoder
│   │   │   ├── config.json
│   │   │   └── pytorch_model.bin
│   │   └── ip-adapter_sd15.bin
└── sd-controlnet-canny
    ├── config.json
    └── diffusion_pytorch_model.bin
```

### 2. Training ed-lora
- Modify the configuration YAML files in `configs/train_edlora`.  
  For the "dogs" and "cats" concepts, we use pre-trained ed-lora models provided by [Concept-Conductor](https://github.com/Nihukat/Concept-Conductor?tab=readme-ov-file) on the same image set for convenience.  
- Start training by running:
  ```bash
  bash train.sh
  ```
- The trained ed-lora models will be saved in the `/experiments` folder.  

*Note: Training is optional. By default, the inference process uses the pre-trained ed-loras provided by us.*



### 3. Inference
- Modify the sample configuration YAML files in the `configs` folder.  
- Run the inference process:
  ```bash
  bash inference.sh
  ```
- The generated images will be saved in the `/outputs` folder.


## Troubleshooting

You may encounter the following compatibility error during execution:

```bash
Traceback (most recent call last):
  File "sample.py", line 18, in <module>
    from diffusers.utils import logging
  File "path_to_site_packages/diffusers/utils/dynamic_modules_utils.py", line 28, in <module>
    from huggingface_hub import cached_download, hf_hub_download, model_info
ImportError: cannot import name 'cached_download' from 'huggingface_hub'
```

This error occurs due to a version mismatch in the `huggingface_hub` package.  
To resolve this, remove `cached_download` from the import statement:  

Replace:
```python
from huggingface_hub import cached_download, hf_hub_download, model_info
```

With:
```python
from huggingface_hub import hf_hub_download, model_info
```
