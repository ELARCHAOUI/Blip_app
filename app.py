from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline
import gradio as gr
import os

# =========================================================
# 🔹 Chargement des modèles
# =========================================================

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Pipelines de traduction
translator_en_ar = pipeline("translation", model="Helsinki-NLP/opus-mt-en-ar")
translator_en_fr = pipeline("translation", model="Helsinki-NLP/opus-mt-en-fr")

# =========================================================
# ⚙️ Fonctions principales
# =========================================================
def generate_caption(img, use_detailed=False):
    """Caption en anglais via BLIP"""
    img_input = Image.fromarray(img)
    inputs = processor(img_input, return_tensors="pt")

    params = {}
    if use_detailed:
        params = {
            "max_length": 80,
            "min_length": 20,
            "num_beams": 5,
            "repetition_penalty": 1.2,
            "length_penalty": 1.5,
            "temperature": 0.7,
        }

    out = model.generate(**inputs, **params)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption


def translate_text(text, target_pipeline):
    """Traduit le texte anglais"""
    try:
        translated = target_pipeline(text, max_length=512)
        return translated[0]["translation_text"]
    except Exception as e:
        return f"Erreur de traduction : {e}"


def process_image(img, detail_level):
    """Retourne caption EN → FR → AR"""
    detailed = detail_level == "Détaillée"
    caption_en = generate_caption(img, detailed)
    caption_fr = translate_text(caption_en, translator_en_fr)
    caption_ar = translate_text(caption_en, translator_en_ar)
    return caption_en, caption_fr, caption_ar

# =========================================================
# 🖥️ Interface Gradio
# =========================================================
demo = gr.Interface(
    fn=process_image,
    inputs=[
        gr.Image(label="🖼️ Choisir une image"),
        gr.Radio(["Simple", "Détaillée"], value="Détaillée", label="🎚️ Niveau de détail")
    ],
    outputs=[
        gr.Text(label="🇬🇧 Description (anglais)"),
        gr.Text(label="🇫🇷 Description (français)"),
        gr.Text(label="🌙 الوصف (arabe)", rtl=True)
    ],
    title="🖼️ Image Captioning Multilingue (EN → FR → AR)",
    description="Téléversez une image pour générer une description en anglais, puis la traduire en français et en arabe."
)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860))
    )
