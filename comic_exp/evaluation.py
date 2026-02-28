from typing import List

import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image


class Evaluator:
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def clip_score(self, texts: List[str], images: List[Image.Image]) -> List[float]:
        inputs = self.clip_proc(text=texts, images=images, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.clip_model(**inputs)
            logits = outputs.logits_per_image
        return logits.diag().cpu().tolist()

    def embedding_distance(self, images: List[Image.Image]) -> List[List[float]]:
        inputs = self.clip_proc(images=images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            emb = self.clip_model.get_image_features(**inputs)
        emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
        sim = emb @ emb.t()
        dist = 1 - sim
        return dist.cpu().tolist()
