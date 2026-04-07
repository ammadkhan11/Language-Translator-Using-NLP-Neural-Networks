import os
import re
import pickle
import numpy as np
import torch
import streamlit as st

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import MarianTokenizer, MarianMTModel


# =========================================================
# Paths
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# French -> English resources
F2E_MODEL_PATH = os.path.join(BASE_DIR, "model_1.h5")
F2E_ENGLISH_TOKENIZER_PATH = os.path.join(BASE_DIR, "english_tokenizer.pkl")
F2E_FRENCH_TOKENIZER_PATH = os.path.join(BASE_DIR, "french_tokenizer.pkl")
F2E_MAX_SEQUENCE_LENGTH = 40

# English -> French resources
# Folder containing saved Marian model files
E2F_MODEL_FOLDER = os.path.join(BASE_DIR, "my_fast_translator")

# English -> Spanish resources
E2S_MODEL_PATH = os.path.join(BASE_DIR, "English_to_Spanish", "model_2.keras")
E2S_ENG_TOKENIZER_PATH = os.path.join(BASE_DIR, "English_to_Spanish", "eng_tokenizer.pkl")
E2S_SP_TOKENIZER_PATH = os.path.join(BASE_DIR, "English_to_Spanish", "sp_tokenizer.pkl")
E2S_MAX_LEN = 15


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


def translate_french_to_english(
    french_sentence,
    loaded_model,
    english_tokenizer,
    french_tokenizer,
    max_sequence_length=40
):
    tokenized_sentence = french_tokenizer.texts_to_sequences([french_sentence])
    padded_sentence = pad(tokenized_sentence, length=max_sequence_length)
    prediction_logits = loaded_model.predict(padded_sentence, verbose=0)[0]
    translated_text = logits_to_text(prediction_logits, english_tokenizer)
    return translated_text


@st.cache_resource
def load_french_to_english_resources():
    try:
        loaded_model = load_model(F2E_MODEL_PATH, compile=False)

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
# English -> Spanish helper functions
# =========================================================

def clean_english_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Zñáéíóúü ]", "", text)
    return text


@st.cache_resource
def load_english_to_spanish_resources():
    try:
        model = load_model(E2S_MODEL_PATH, compile=False)

        with open(E2S_ENG_TOKENIZER_PATH, "rb") as f:
            eng_tokenizer = pickle.load(f)

        with open(E2S_SP_TOKENIZER_PATH, "rb") as f:
            sp_tokenizer = pickle.load(f)

        index_to_word = {i: w for w, i in sp_tokenizer.word_index.items()}

        return model, eng_tokenizer, sp_tokenizer, index_to_word

    except Exception as e:
        st.error(f"Error loading English→Spanish resources: {e}")
        return None, None, None, None


def translate_english_to_spanish(sentence, model, eng_tokenizer, index_to_word, max_len=15):
    if model is None or eng_tokenizer is None or index_to_word is None:
        return "Error: English→Spanish translator not loaded."

    sentence = clean_english_text(sentence)

    seq = eng_tokenizer.texts_to_sequences([sentence])
    seq = pad_sequences(seq, maxlen=max_len, padding="post")

    pred = model.predict(seq, verbose=0)[0]

    words = []
    for p in pred:
        word = index_to_word.get(np.argmax(p), "")
        if word:
            words.append(word)

    return " ".join(words).strip()


# =========================================================
# Streamlit UI
# =========================================================

st.set_page_config(page_title="Language Translator", page_icon="🌍")
st.title("🌍 Language Translator")

translation_direction = st.selectbox(
    "Choose translation direction:",
    [
        "French to English",
        "English to French",
        "English to Spanish"
    ]
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
        elif (
            f2e_model is not None
            and f2e_english_tokenizer is not None
            and f2e_french_tokenizer is not None
        ):
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

elif translation_direction == "English to French":
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
        elif e2f_tokenizer is not None and e2f_model is not None and e2f_device is not None:
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

else:
    st.subheader("🇬🇧 English → Spanish 🇪🇸")
    st.write("Translate English text into Spanish using your trained Keras model.")

    e2s_model, e2s_eng_tokenizer, e2s_sp_tokenizer, e2s_index_to_word = load_english_to_spanish_resources()

    english_input = st.text_area(
        "Enter English text:",
        "Hello, how are you today?"
    )

    if st.button("Translate to Spanish"):
        if english_input.strip() == "":
            st.warning("Please enter some English text.")
        elif e2s_model is not None and e2s_eng_tokenizer is not None and e2s_index_to_word is not None:
            with st.spinner("Translating..."):
                spanish_translation = translate_english_to_spanish(
                    english_input,
                    e2s_model,
                    e2s_eng_tokenizer,
                    e2s_index_to_word,
                    E2S_MAX_LEN
                )
                st.success("Translation complete!")
                st.write("**Spanish Translation:**")
                st.info(spanish_translation)
        else:
            st.error("English→Spanish model or tokenizers could not be loaded.")

