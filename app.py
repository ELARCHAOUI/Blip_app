import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import os

# =========================================================
# 🧠 CONFIGURATION DE L'API GEMINI
# =========================================================
try:
    from google import genai
    from google.genai.errors import APIError

    # On lit la clé depuis les variables d'environnement (compat Render)
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

    if GEMINI_API_KEY:
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        GEMINI_MODEL_NAME = "gemini-2.5-flash"
    else:
        st.warning(
            "⚠️ Clé API Gemini non trouvée. "
            "Veuillez la configurer dans les variables d'environnement (Render > Environment Variables)."
        )
        gemini_client = None

except ImportError:
    st.error("❌ Veuillez installer la bibliothèque `google-genai` : pip install google-genai")
    gemini_client = None
except Exception as e:
    st.error(f"❌ Erreur d'initialisation de l'API Gemini : {e}")
    gemini_client = None


# =========================================================
# ⚙️ CHARGEMENT DU MODÈLE BLIP (mise en cache)
# =========================================================
@st.cache_resource
def load_blip_model():
    """Charge et met en cache le modèle BLIP pour la génération de description."""
    with st.spinner("Chargement du modèle BLIP..."):
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model


processor, blip_model = load_blip_model()


# =========================================================
# 🖼️ FONCTIONS DE TRAITEMENT
# =========================================================
def generate_caption(img, use_detailed=False):
    """Génère une description anglaise de l'image à l'aide de BLIP."""
    img_input = img.convert("RGB")
    inputs = processor(img_input, return_tensors="pt")

    if use_detailed:
        out = blip_model.generate(
            **inputs,
            max_length=80,
            min_length=20,
            num_beams=2,
            repetition_penalty=1.1,
            temperature=0.7,
        )
    else:
        out = blip_model.generate(**inputs, max_length=50, min_length=10)

    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption


def translate_to_french_with_gemini(text):
    """Traduit le texte anglais en français à l'aide de l'API Gemini."""
    if not gemini_client:
        return "⚠️ Traduction non disponible (clé API manquante)."

    prompt = (
        "Traduisez ce texte de l'anglais au français de manière fluide et naturelle, "
        "sans ajouter de commentaires ni d'explications :\n\n" + text
    )

    try:
        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL_NAME, contents=prompt
        )
        return response.text.strip()
    except APIError as e:
        return f"Erreur API Gemini : {e}"
    except Exception as e:
        return f"Erreur inattendue de traduction : {e}"


# =========================================================
# 🌐 INTERFACE STREAMLIT
# =========================================================
st.set_page_config(page_title="🖼️ BLIP + Gemini Translator", layout="centered")

st.title("🖼️ Image Captioning & Traduction 🇫🇷")
st.write("Générez des descriptions précises avec **BLIP** et traduisez-les automatiquement avec **Gemini**.")
st.markdown("---")

# Zone d'import d'image
uploaded_file = st.file_uploader("📸 Choisissez une image (JPG, JPEG, PNG)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Image téléversée", use_container_width=True)
    st.markdown("---")

    detail_level = st.radio(
        "🔎 Niveau de détail de la description :",
        ["Simple", "Détaillée"],
        index=1,
        horizontal=True,
    )

    use_detailed = detail_level == "Détaillée"

    if st.button("🚀 Générer et Traduire"):
        with st.spinner(f"Génération de la description ({detail_level})..."):
            caption_en = generate_caption(image, use_detailed)

        st.success("✅ Description générée avec succès !")
        st.markdown("#### 📝 Description (anglais)")
        st.write(caption_en)

        st.markdown("#### 🇫🇷 Traduction (français)")
        if gemini_client:
            with st.spinner("Traduction via Gemini..."):
                caption_fr = translate_to_french_with_gemini(caption_en)
            st.write(caption_fr)
        else:
            st.warning("Gemini n'est pas initialisé – assurez-vous d'avoir configuré la clé API.")

    st.markdown("---")

st.caption("🔧 Architecture : BLIP (Vision) → Gemini (Traduction)")
