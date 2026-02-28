from typing import List, Optional

from diffusers import StableDiffusionPipeline
import torch


class ImageGenerator:
    def __init__(self, device: str = "cuda", hf_token: Optional[str] = None):
        self.device = device
        try:
            self.sd = StableDiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                use_auth_token=hf_token,
                torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32,
            ).to(device)
        except Exception as e:
            print("failed to load SD pipeline:", e)
            self.sd = None

    def generate(self, prompts: List[str], num_inference_steps: int = 25) -> List:
        if self.sd is None:
            raise RuntimeError("stable diffusion pipeline not initialized")
        imgs = []
        for p in prompts:
            out = self.sd(p, num_inference_steps=num_inference_steps)
            imgs.append(out.images[0])
        return imgs

    def with_controlnet(self, *args, **kwargs):
        raise NotImplementedError("ControlNet support not yet implemented")

    def personalize_character(self, *args, **kwargs):
        raise NotImplementedError("Character personalization not yet implemented")
