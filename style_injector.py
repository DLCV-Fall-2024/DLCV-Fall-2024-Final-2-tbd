import os
import torch
from PIL import Image
import cv2
import numpy as np
import argparse

from controlnet_aux import CannyDetector
from diffusers.utils import load_image, make_image_grid
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel


class IPStyleInjector:
    def __init__(
            self,
            pipe,
            input_image_path="", 
            style_folder="dlcv_data/concept_image/watercolor", 
        ):
        super().__init__()
        self.images = None
        self.pipe = pipe
        self.style_images = [load_image(f"{style_folder}/image_01_0{i+1}.jpg") for i in range(3)]

        self.input_image = load_image(input_image_path).resize((768, 768))

        self.canny = CannyDetector()
        self.controlnet_condition_image = self.canny(self.input_image, detect_resolution=512, image_resolution=768, low_threshold=150, high_threshold=200)
        # self.controlnet_condition_image = self.canny(self.input_image, detect_resolution=512, image_resolution=768, low_threshold=180, high_threshold=240)
        self.input_image = self.input_image.resize((512, 512), Image.Resampling.LANCZOS)
        self.controlnet_condition_image = self.controlnet_condition_image.resize((512, 512), Image.Resampling.LANCZOS)
        # self.controlnet_condition_image.save("canny.png", format="PNG")


    def __call__(
        self,
        prompt: str,
        num_images_per_prompt: int,
        *args,
        **kwargs,
    ):
        images = self.pipe(prompt = prompt, 
                    negative_prompt = "low quality, photorealistic, sharp, deformed, oversaturated",
                    height = 512, 
                    width = 512,
                    init_image = self.input_image,
                    ip_adapter_image = [self.style_images],
                    # ip_adapter_image = ip_adap_img,
                    image = self.controlnet_condition_image,
                    guidance_scale = 3,
                    controlnet_conditioning_scale = 0.7,
                    control_guidance_end = 0.9, # 1.0
                    num_inference_steps = 20,
                    num_images_per_prompt = num_images_per_prompt).images
        return images

    def align_lab_color(self, gen_image, weight_l=0.5):
        """
        Match colors with additional weight for lightness (L channel).

        Args:
            generated (PIL.Image): Generated image.
            weight_l (float): Weight for lightness channel (0-1).

        Returns:
            PIL.Image: Adjusted image.
        """
        # Convert to LAB
        original_lab = cv2.cvtColor(np.array(self.input_image), cv2.COLOR_RGB2LAB)
        generated_lab = cv2.cvtColor(np.array(gen_image), cv2.COLOR_RGB2LAB)

        # Adjust A and B channels
        for i in range(1, 3):  # A and B channels
            generated_lab[:, :, i] = (1 - weight_l) * generated_lab[:, :, i] + weight_l * original_lab[:, :, i]

        # Convert back to RGB
        adjusted_rgb = cv2.cvtColor(generated_lab, cv2.COLOR_LAB2RGB)
        return Image.fromarray(adjusted_rgb)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--style_folder", type=str, default="datasets/watercolor")
    parser.add_argument("--img_folder", type=str, required=True)
    parser.add_argument("--controlnet", type=str, default="pretrained_models/sd-controlnet-canny")
    parser.add_argument("--pipe", type=str, default="pretrained_models/AbsoluteReality")
    parser.add_argument("--ip_adapter", type=str, default="pretrained_models/IP-Adapter")
    parser.add_argument("--img_per_prompt", type=str, default="pretrained_models/IP-Adapter")
    args = parser.parse_args()
        
    device = torch.device("cuda:0")
    save_dir = args.output_dir
    os.makedirs(save_dir, exist_ok=True)

    controlnet = ControlNetModel.from_pretrained(
        args.controlnet, 
        torch_dtype=torch.float16,
        varient="fp16").to(device)
    
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        args.pipe,
        controlnet=controlnet, 
        torch_dtype=torch.float16).to(device)

    pipe.load_ip_adapter(args.ip_adapter, subfolder="models", weight_name="ip-adapter_sd15.bin")
    pipe.set_ip_adapter_scale({"up": {"block_0": [0, 0.5, 0]}})

    prompt = """A dark grey cat wearing wearable glasses in a watercolor style."""
    num_images_per_prompt = 5
    ip_scale = 0.5

    # generation
    for img_file in os.listdir(args.img_folder):
        style_injector = IPStyleInjector(
            pipe=pipe,
            input_image_path=os.path.join(args.img_folder, img_file), 
            style_folder=args.style_folder, 
        )
        images = style_injector(prompt=prompt, num_images_per_prompt=num_images_per_prompt)

        for i, image in enumerate(images):
            image.save(os.path.join(save_dir, f"{i}_output_ip{ip_scale}.png"), format="PNG")

        grid = make_image_grid(images, cols=num_images_per_prompt, rows=1)
        grid.save(os.path.join(save_dir, f"grid_ip{ip_scale}.png"), format="PNG")


        aligned_images = [style_injector.align_lab_color(gen_img) for gen_img in images]
        for j, image in enumerate(aligned_images):
            image.save(os.path.join(save_dir, f"{j}_output_aligned_ip{ip_scale}.png"), format="PNG")

        grid = make_image_grid(aligned_images, cols=num_images_per_prompt, rows=1)
        grid.save(os.path.join(save_dir, f"grid_aligned_ip{ip_scale}.png"), format="PNG")

