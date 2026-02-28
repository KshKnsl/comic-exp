import gradio as gr
from .pipeline import ComicPipeline


def launch_local_ui(device: str = "cpu"):
    pipe = ComicPipeline(device=device)

    def _gen(txt: str):
        data = pipe.process_text(txt)
        prompts = pipe.make_prompts(data["scenes"], data["characters"], data["emotions"])
        try:
            imgs = pipe.generate(prompts)
        except Exception as e:
            imgs = []
            print("generation error", e)
        captions = [f"Scene {i+1}: {s}" for i, s in enumerate(data["scenes"])]
        return imgs, "\n".join(captions)

    gr.Interface(
        fn=_gen,
        inputs=gr.Textbox(lines=10, placeholder="Paste mythological text here"),
        outputs=[gr.Gallery(label="Generated Images"), gr.Textbox(label="Scene captions")],
        title="Comic‑exp Storyboard Generator",
        description="Enter narrative text and get a sequence of images",
    ).launch()