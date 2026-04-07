import streamlit as st
import os
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np


def pad(x, length=None):
  if length is None:
      return pad_sequences(x, maxlen=40, padding = "post")
  return pad_sequences(x, maxlen=length, padding = "post")

def logits_to_text(logits, tokenizer):
    index_to_words = {id: word for word, id in tokenizer.word_index.items()}
    index_to_words[0] = ''
    return ' '.join([index_to_words[prediction] for prediction in np.argmax(logits, 1)])

def optimized_translate_sentence(french_sentence, loaded_model, english_tokenizer, french_tokenizer, max_sequence_length=40):
    tokenized_sentence = french_tokenizer.texts_to_sequences([french_sentence])
    padded_sentence = pad(tokenized_sentence, length=max_sequence_length)
    prediction_logits = loaded_model.predict(padded_sentence.reshape(1, -1))[0]
    translated_text = logits_to_text(prediction_logits, english_tokenizer)
    return translated_text

MODEL_PATH = "model_1.h5"
ENGLISH_TOKENIZER_PATH = "english_tokenizer.pkl"
FRENCH_TOKENIZER_PATH = "french_tokenizer.pkl"
MAX_SEQUENCE_LENGTH = 40 

@st.cache_resource 
def load_translation_resources():
    st.write("Streamlit app: Loading model and tokenizers...") 
    try:
        loaded_model = load_model(MODEL_PATH)
        with open(ENGLISH_TOKENIZER_PATH, 'rb') as f:
            loaded_english_tokenizer = pickle.load(f)
        with open(FRENCH_TOKENIZER_PATH, 'rb') as f:
            loaded_french_tokenizer = pickle.load(f)
        st.success("Streamlit app: Model and tokenizers loaded successfully.")
        return loaded_model, loaded_english_tokenizer, loaded_french_tokenizer
    except Exception as e:
        st.error(f"Error loading translation resources: {e}")
        return None, None, None

streamlit_model, streamlit_english_tokenizer, streamlit_french_tokenizer = load_translation_resources()

st.title("French to English Translator")

french_input = st.text_area("Enter French Text:", "Je suis un étudiant.")

if st.button("Translate"):
    if streamlit_model and streamlit_english_tokenizer and streamlit_french_tokenizer:
        with st.spinner('Translating...') :
            translation = optimized_translate_sentence(
                french_input,
                streamlit_model,
                streamlit_english_tokenizer,
                streamlit_french_tokenizer,
                MAX_SEQUENCE_LENGTH
            )
            st.success("Translation complete!")
            st.write("**Translated English:**")
            st.write(translation)
    else:
        st.error("Translation service not available. Please check model and tokenizer paths.")
