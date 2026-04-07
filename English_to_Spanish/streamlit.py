import streamlit as st
import pandas as pd
import numpy as np
import re
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- Paths to your saved files (adjust if necessary) ---
MODEL_PATH = "English_to_Spanish/model_2.keras"
ENG_TOKENIZER_PATH = "English_to_Spanish/eng_tokenizer.pkl"
SP_TOKENIZER_PATH = "English_to_Spanish/sp_tokenizer.pkl"

# --- Load the model and tokenizers ---
@st.cache_resource
import os
import streamlit as st
from tensorflow.keras.models import load_model

@st.cache_resource
def load_artifacts():
    model = load_model(MODEL_PATH)
    with open(ENG_TOKENIZER_PATH, 'rb') as f:
        eng_tokenizer = pickle.load(f)
    with open(SP_TOKENIZER_PATH, 'rb') as f:
        sp_tokenizer = pickle.load(f)
    
    # Create a reverse dictionary for Spanish tokenizer
    index_to_word = {i: w for w, i in sp_tokenizer.word_index.items()}
    
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
