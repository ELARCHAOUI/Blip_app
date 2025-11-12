from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline
import gradio as gr


# 1️⃣ Modèle BLIP (Captioning en anglais)
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# 2️⃣ Pipelines de traduction
# Anglais → Arabe
translator_en_ar = pipeline("translation", model="Helsinki-NLP/opus-mt-en-ar")
# Anglais → Français
translator_en_fr = pipeline("translation", model="Helsinki-NLP/opus-mt-en-fr")




def generate_caption(img, use_detailed=False):
    """Génère une description (caption) en anglais via BLIP"""
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
    """Traduit le texte anglais dans une autre langue avec le pipeline spécifié"""
    try:
        translated = target_pipeline(text, max_length=512)
        return translated[0]["translation_text"]
    except Exception as e:
        return f"Erreur de traduction : {e}"


def process_image(img, detail_level):
    """Retourne la description en anglais, en français et en arabe"""
    use_detailed = detail_level == "Détaillée"

    # 1️⃣ Description en anglais
    caption_en = generate_caption(img, use_detailed=use_detailed)

    # 2️⃣ Traduction en français
    caption_fr = translate_text(caption_en, translator_en_fr)

    # 3️⃣ Traduction en arabe
    caption_ar = translate_text(caption_en, translator_en_ar)

    return caption_en, caption_fr, caption_ar




demo = gr.Interface(
    fn=process_image,
    inputs=[
        gr.Image(label="Importez une image à analyser"),
        gr.Radio(
            choices=["Simple", "Détaillée"],
            value="Détaillée",
            label="Niveau de détail de la description"
        ),
    ],
    outputs=[
        gr.Text(label="🇬🇧 1. Description (Anglais)"),
        gr.Text(label="🇫🇷 2. Description (Français)"),
        gr.Text(label="🌙 3. الوصف (Arabe)", rtl=True),
    ],
    title="Image Captioning Multilingue (EN → FR → AR)",
    description="Téléversez une image pour obtenir automatiquement une description en anglais, français et arabe.",
    allow_flagging="never"
)


if __name__ == "__main__":
    # host="0.0.0.0" pour Render / HuggingFace / Docker
    demo.launch(server_name="0.0.0.0", server_port=7860)
