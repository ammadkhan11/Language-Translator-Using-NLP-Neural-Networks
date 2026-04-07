import streamlit as st
import pickle
import numpy as np
import torch

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import MarianTokenizer, MarianMTModel


# =========================================================
# Paths
# =========================================================

# French -> English resources
F2E_MODEL_PATH = "model_1.h5"
F2E_ENGLISH_TOKENIZER_PATH = "english_tokenizer.pkl"
F2E_FRENCH_TOKENIZER_PATH = "french_tokenizer.pkl"
F2E_MAX_SEQUENCE_LENGTH = 40

# English -> French resources
# This should be a folder in your GitHub repo containing the saved Marian model files
E2F_MODEL_FOLDER = "my_fast_translator"


# =========================================================
# French -> English helper functions
# =========================================================

def pad(x, length=None):
    if length is None:
        return pad_sequences(x, maxlen=F2E_MAX_SEQUENCE_LENGTH, padding="post")
    return pad_sequences(x, maxlen=length, padding="post")


def logits_to_text(logits, tokenizer):
    index_to_words = {idx: word for word, idx in tokenizer.word_index.items()}
    index_to_words[0] = ""
    predicted_tokens = np.argmax(logits, axis=1)
    words = [index_to_words.get(token, "") for token in predicted_tokens]
    return " ".join(words).strip()


def translate_french_to_english(french_sentence, loaded_model, english_tokenizer, french_tokenizer, max_sequence_length=40):
    tokenized_sentence = french_tokenizer.texts_to_sequences([french_sentence])
    padded_sentence = pad(tokenized_sentence, length=max_sequence_length)
    prediction_logits = loaded_model.predict(padded_sentence, verbose=0)[0]
    translated_text = logits_to_text(prediction_logits, english_tokenizer)
    return translated_text


@st.cache_resource
def load_french_to_english_resources():
    try:
        loaded_model = load_model(F2E_MODEL_PATH)

        with open(F2E_ENGLISH_TOKENIZER_PATH, "rb") as f:
            loaded_english_tokenizer = pickle.load(f)

        with open(F2E_FRENCH_TOKENIZER_PATH, "rb") as f:
            loaded_french_tokenizer = pickle.load(f)

        return loaded_model, loaded_english_tokenizer, loaded_french_tokenizer

    except Exception as e:
        st.error(f"Error loading French→English resources: {e}")
        return None, None, None


# =========================================================
# English -> French helper functions
# =========================================================

@st.cache_resource
def load_english_to_french_resources(path):
    try:
        tokenizer = MarianTokenizer.from_pretrained(path)
        model = MarianMTModel.from_pretrained(path)

        device = torch.device("cpu")
        model.to(device)
        model.eval()

        return tokenizer, model, device

    except Exception as e:
        st.error(f"Error loading English→French model/tokenizer from '{path}': {e}")
        return None, None, None


def translate_english_to_french(text, tokenizer, model, device, max_length=64, num_beams=2):
    if tokenizer is None or model is None or device is None:
        return "Error: English→French translator not loaded."

    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# =========================================================
# Streamlit UI
# =========================================================

st.set_page_config(page_title="Language Translator", page_icon="🌍")
st.title("🌍 English ↔ French Translator")

translation_direction = st.selectbox(
    "Choose translation direction:",
    ["French to English", "English to French"]
)

if translation_direction == "French to English":
    st.subheader("🇫🇷 French → English 🇬🇧")
    st.write("Translate French text into English using your trained Keras model.")

    f2e_model, f2e_english_tokenizer, f2e_french_tokenizer = load_french_to_english_resources()

    french_input = st.text_area(
        "Enter French text:",
        "Je suis un étudiant."
    )

    if st.button("Translate to English"):
        if french_input.strip() == "":
            st.warning("Please enter some French text.")
        elif f2e_model is not None and f2e_english_tokenizer is not None and f2e_french_tokenizer is not None:
            with st.spinner("Translating..."):
                translation = translate_french_to_english(
                    french_input,
                    f2e_model,
                    f2e_english_tokenizer,
                    f2e_french_tokenizer,
                    F2E_MAX_SEQUENCE_LENGTH
                )
                st.success("Translation complete!")
                st.write("**English Translation:**")
                st.info(translation)
        else:
            st.error("French→English model or tokenizers could not be loaded.")

else:
    st.subheader("🇬🇧 English → French 🇫🇷")
    st.write("Translate English text into French using your MarianMT model.")

    e2f_tokenizer, e2f_model, e2f_device = load_english_to_french_resources(E2F_MODEL_FOLDER)

    english_input = st.text_area(
        "Enter English text:",
        "Hello, how are you today?"
    )

    if st.button("Translate to French"):
        if english_input.strip() == "":
            st.warning("Please enter some English text.")
        elif e2f_tokenizer is not None and e2f_model is not None:
            with st.spinner("Translating..."):
                french_translation = translate_english_to_french(
                    english_input,
                    e2f_tokenizer,
                    e2f_model,
                    e2f_device
                )
                st.success("Translation complete!")
                st.write("**French Translation:**")
                st.info(french_translation)
        else:
            st.error("English→French model could not be loaded.")


st.markdown("---")
st.caption("Built with Streamlit, TensorFlow, PyTorch, and Hugging Face Transformers.")
