import streamlit as st
import pandas as pd
import numpy as np
import re
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- Paths to your saved files (adjust if necessary) ---
MODEL_PATH = "model_2.keras"
ENG_TOKENIZER_PATH = "eng_tokenizer.pkl"
SP_TOKENIZER_PATH = "sp_tokenizer.pkl"

# --- Load the model and tokenizers ---
@st.cache_resource
import os
import streamlit as st
from tensorflow.keras.models import load_model

@st.cache_resource
def load_artifacts():
    st.write("Current working dir:", os.getcwd())
    st.write("MODEL_PATH:", MODEL_PATH)
    st.write("Exists:", os.path.exists(MODEL_PATH))
    
    model = load_model(MODEL_PATH, compile=False)
    return model, eng_tokenizer, sp_tokenizer, index_to_word

model, eng_tokenizer, sp_tokenizer, index_to_word = load_artifacts()

# --- Configuration (should match training) ---
MAX_LEN = 15 # Make sure this matches the max_len used during training

# --- Helper functions (copied from your notebook) ---
def clean(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Zñáéíóúü ]", "", text)
    return text

def translate(sentence):
    sentence = clean(sentence)

    seq = eng_tokenizer.texts_to_sequences([sentence])
    seq = pad_sequences(seq, maxlen=MAX_LEN, padding="post")

    pred = model.predict(seq, verbose=0)[0] # verbose=0 to suppress output

    words = []
    for p in pred:
        word = index_to_word.get(np.argmax(p), "")
        words.append(word)

    return " ".join(words).strip()

# --- Streamlit UI ---
st.title("English to Spanish Translator")
st.write("Enter an English sentence below to get its Spanish translation.")

user_input = st.text_input("English Sentence:", "")

if st.button("Translate"):
    if user_input:
        translation = translate(user_input)
        st.success(f"Spanish Translation: {translation}")
    else:
        st.warning("Please enter an English sentence to translate.")
