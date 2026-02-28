import re
from typing import List, Dict
from transformers import pipeline
import spacy
from langdetect import detect


class LanguageTranslator:
    def __init__(self, model_name: str = "Helsinki-NLP/opus-mt-mul-en"):
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def to_english(self, text: str) -> str:
        try:
            lang = detect(text)
        except Exception:
            lang = "en"
        if lang.startswith("en"):
            return text
        inputs = self.tokenizer(text, return_tensors="pt", padding=True)
        outputs = self.model.generate(**inputs)
        decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return decoded[0] if decoded else text



class SceneSegmenter:
    def __init__(self):
        pass

    def split(self, text: str) -> List[str]:
        parts = re.split(r"\n\s*\n", text.strip())
        scenes = [p.strip() for p in parts if p.strip()]
        merged: List[str] = []
        for s in scenes:
            if merged and len(s.split()) < 10:
                merged[-1] = merged[-1] + " " + s
            else:
                merged.append(s)
        return merged


class CharacterExtractor:
    def __init__(self, model_name: str = "dbmdz/bert-large-cased-finetuned-conll03-english"):
        try:
            self.ner = pipeline("ner", model=model_name, aggregation_strategy="simple")
        except TypeError:
            self.ner = pipeline("ner", grouped_entities=True, model=model_name)

        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            self.nlp = None

    def extract(self, scenes: List[str]) -> List[List[str]]:
        all_chars = []
        for s in scenes:
            entities = self.ner(s)
            names = [e["word"] for e in entities if e["entity_group"] == "PER"]
            all_chars.append(names)
        return all_chars

    def resolve_coref(self, text: str) -> str:
        if not self.nlp:
            raise RuntimeError("spaCy model not loaded; run 'python -m spacy download en_core_web_sm'")
        doc = self.nlp(text)
        return text


class EmotionExtractor:
    def __init__(self, model_name: str = "mrm8488/t5-base-finetuned-emotion"):
        try:
            self.emotion = pipeline("text-classification", model=model_name, return_all_scores=True)
        except Exception as e:
            print("warning: emotion pipeline failed, using fallback (", e, ")")
            self.emotion = None

    def predict(self, texts: List[str]) -> List[Dict[str, float]]:
        if self.emotion is None:
            return [{"neutral": 1.0} for _ in texts]
        scores_list = []
        for s in texts:
            scores = self.emotion(s)
            if isinstance(scores, list) and scores:
                scores_list.append({r["label"]: r["score"] for r in scores[0]})
            else:
                scores_list.append({})
        return scores_list

    def predict(self, texts: List[str]) -> List[Dict[str, float]]:
        scores_list = []
        for s in texts:
            scores = self.emotion(s)
            if isinstance(scores, list) and scores:
                scores_list.append({r["label"]: r["score"] for r in scores[0]})
            else:
                scores_list.append({})
        return scores_list