from typing import List, Dict, Optional

from .nlp import SceneSegmenter, CharacterExtractor, EmotionExtractor, LanguageTranslator
from .generation import ImageGenerator
from .evaluation import Evaluator


class ComicPipeline:
    def __init__(self, device: str = "cuda", hf_token: Optional[str] = None):
        self.translator = LanguageTranslator()
        self.segmenter = SceneSegmenter()
        self.char_extractor = CharacterExtractor()
        self.emotion_extractor = EmotionExtractor()
        self.generator = ImageGenerator(device=device, hf_token=hf_token)
        self.evaluator = Evaluator(device=device)

    def process_text(self, text: str) -> Dict:
        text_en = self.translator.to_english(text)
        scenes = self.segmenter.split(text_en)
        chars = self.char_extractor.extract(scenes)
        emos = self.emotion_extractor.predict(scenes)
        return {"scenes": scenes, "characters": chars, "emotions": emos}

    def make_prompts(self, scenes: List[str], characters: List[List[str]], emotions: List[Dict[str, float]]) -> List[str]:
        prompts = []
        for s, chars, emo in zip(scenes, characters, emotions):
            top = max(emo, key=emo.get) if emo else "neutral"
            char_str = ", ".join(chars) if chars else ""
            prompts.append(f"{s}. depict {char_str} with {top} emotion, cinematic illustration")
        return prompts

    def generate(self, prompts: List[str], **kwargs) -> List:
        return self.generator.generate(prompts, **kwargs)

    def score(self, scenes: List[str], images: List):
        texts = [f"{s}" for s in scenes]
        return self.evaluator.clip_score(texts, images)

    def consistency(self, images: List):
        return self.evaluator.embedding_distance(images)
