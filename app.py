import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import os

# --- CONFIGURATION GEMINI ---
# --- Configuration de l'API Gemini ---
try:
    from google import genai
    from google.genai.errors import APIError

    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

    if GEMINI_API_KEY:
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        GEMINI_MODEL_NAME = "gemini-2.5-flash"
    else:
        st.error("⚠️ La clé API Gemini n'est pas configurée dans les variables d'environnement.")
        gemini_client = None

except ImportError:
    st.error("Veuillez installer la bibliothèque Google GenAI : `pip install google-genai`")
    gemini_client = None
except Exception as e:
    st.error(f"Erreur d'initialisation de l'API Gemini : {e}")
    gemini_client = None


# --- CHARGEMENT DES MODÈLES BLIP ---
@st.cache_resource
def load_blip_model():
    with st.spinner("Chargement du modèle BLIP..."):
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model


processor, blip_model = load_blip_model()


# --- FONCTIONS ---
def generate_caption(img, use_detailed=False):
    """Génère une description de l'image (anglais)."""
    img_input = img.convert("RGB")
    inputs = processor(img_input, return_tensors="pt")

    if use_detailed:
        out = blip_model.generate(
            **inputs,
            max_length=100,
            min_length=20,
            num_beams=3,
            repetition_penalty=1.1,
            length_penalty=1.0,
            temperature=0.7,
        )
    else:
        out = blip_model.generate(**inputs)

    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption


def translate_to_french_with_gemini(text):
    if not gemini_client:
        return "⚠️ Traduction non disponible (clé API manquante)."

    prompt = (
        "Traduisez ce texte de l'anglais au français de manière naturelle et fluide, "
        "sans ajouter de commentaires ni d’explications :\n\n" + text
    )

    try:
        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL_NAME,
            contents=prompt,
        )
        return response.text.strip()
    except APIError as e:
        return f"Erreur API Gemini : {e}"
    except Exception as e:
        return f"Erreur inattendue : {e}"


# --- INTERFACE STREAMLIT ---

st.set_page_config(page_title="🖼️ Image Captioning & Traduction", layout="centered")

st.title("🖼️ Image Captioning")
st.write("**Générez des descriptions détaillées avec BLIP.**")

st.markdown("---")

uploaded_file = st.file_uploader("📤 Choisissez une image…", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Image Téléversée", use_container_width=True)
    st.markdown("---")

    col_detail = st.radio(
        "Niveau de détail de la description :",
        ["Simple", "Détaillée"],
        index=1,
        horizontal=True,
    )

    use_detailed = col_detail == "Détaillée"

    if st.button("Générer et Traduire"):
        with st.spinner(f"Génération de la description ({col_detail})..."):
            caption_en = generate_caption(image, use_detailed)
        st.success("✅ Description générée avec succès")

        st.markdown("### ✏️ Description (anglais)")
        st.write(f"**{caption_en}**")

        st.markdown("### 🇫🇷 Traduction (français)")
        if gemini_client:
            with st.spinner("Traduction en cours..."):
                caption_fr = translate_to_french_with_gemini(caption_en)
            st.write(f"**{caption_fr}**")
        else:
            st.warning("Veuillez configurer une clé API Gemini pour activer la traduction.")

        st.markdown("---")


st.caption("🧠 Architecture : BLIP (Vision) + Gemini (Traduction)")
