
import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline
import torch
import os



@st.cache_resource
def load_models():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    translator_en_ar = pipeline("translation", model="Helsinki-NLP/opus-mt-en-ar")
    translator_en_fr = pipeline("translation", model="Helsinki-NLP/opus-mt-en-fr")
    return processor, model, translator_en_fr, translator_en_ar


processor, model, translator_en_fr, translator_en_ar = load_models()


def generate_caption(img, use_detailed=False):
    """Génère une description anglaise via BLIP"""
    img_input = img.convert("RGB")
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
    """Traduit un texte anglais avec le pipeline spécifié"""
    try:
        translated = target_pipeline(text, max_length=512)
        return translated[0]["translation_text"]
    except Exception as e:
        return f"Erreur de traduction : {e}"



st.set_page_config(page_title="Image Captioning Multilingue", layout="centered")

st.title("Image Captioning")
st.caption("Modèle utilisé : **BLIP (Salesforce)** + **Helsinki-NLP** pour traductions")

uploaded_file = st.file_uploader("📤 Téléverser une image :", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Aperçu de l'image", use_container_width=True)

    detail_level = st.radio(
        "Choisissez le niveau de détail de la description :",
        ["Simple", "Détaillée"],
        index=1,
        horizontal=True,
    )

    if st.button("Générer les descriptions"):
        with st.spinner("Génération avec BLIP..."):
            caption_en = generate_caption(image, use_detailed=(detail_level == "Détaillée"))

        st.success("Description générée avec succès !")
        st.subheader("🇬🇧 Description (Anglais)")
        st.write(caption_en)

        with st.spinner("Traduction en Français..."):
            caption_fr = translate_text(caption_en, translator_en_fr)

        with st.spinner("Traduction en Arabe..."):
            caption_ar = translate_text(caption_en, translator_en_ar)

        st.subheader("🇫🇷 Description (Français)")
        st.write(caption_fr)

        st.subheader("🌙 الوصف (Arabe)")
        st.write(caption_ar)

    st.divider()

st.caption("🔧 Architecture : BLIP (Vision) + Helsinki-NLP (Traduction)")
