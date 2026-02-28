# comic-exp

A storyboarding toolkit that converts mythological text into a sequence
of illustrated scenes.  The entire application is packaged in a single
Python module; there is no CLI, and running the package always launches
the GUI.

### Features implemented

* **Language detection & translation** – input may be Hindi,
  Sanskrit or other Indic text; the pipeline detects the language and
  translates it to English before further processing.
* **Scene segmentation** uses paragraph boundaries as the base unit.
  Very short fragments are merged with their neighbours so that each
  scene description is a longer, more complete chunk of narrative rather
  than an isolated sentence.
* **Character extraction** using HuggingFace NER; placeholder for
  coreference resolution.
* **Emotion classification** powered by a pretrained transformer.
* **Prompt engineering** routines that combine scene text, characters,
  and the dominant emotion to produce Stable Diffusion prompts.
* **Stable Diffusion wrapper** via Diffusers with hooks for
  ControlNet/DreamBooth personalization.
* **Evaluation utilities** – CLIP score and character embedding
  consistency metrics.
* **GUI-only entry point** via the module or console script.  The
  older CLI and Colab scripts have been removed to keep the codebase
  focused.
* **Gradio frontend** that runs locally; no remote/Colab helper
  scripts are needed.
* **Modular design** – every major capability resides in its own
  module (`nlp.py`, `generation.py`, `evaluation.py`), making it easy to
  extend or replace components.

### Getting started

```bash
pip install -r requirements.txt
```

```bash
python -m comic_exp    # or, after doing a `pip install .`, simply `comic-exp`
```
```python
!pip install -r requirements.txt
from colab import ComicPipeline
pipe = ComicPipeline(device="cuda")

text = "..."
data = pipe.process_text(text)
prompts = pipe.make_prompts(data['scenes'], data['characters'], data['emotions'])
images = pipe.generate(prompts)
```

### Extending the product

Later phases may add:

* **Coreference resolution** (see
  ``CharacterExtractor.resolve_coref``).  spaCy or AllenNLP can be
  plugged in here.
* **ControlNet / pose/layout conditioning** – add methods in
  ``generation.ImageGenerator.with_controlnet``.
* **DreamBooth / LoRA personalization** via
  ``ImageGenerator.personalize_character``.
* **Multilingual preprocessing** using IndicNLP models for Hindi/
  Sanskrit.
* **Evaluation harness** to run automated metrics or conduct small
  human studies (see :mod:`comic_exp.evaluation`).

---
