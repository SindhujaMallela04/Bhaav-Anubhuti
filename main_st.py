import streamlit as st
import base64

# Function to load and display background image
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{encoded_string});
            background-size: cover;
            background-position: center;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# function call to set the background
add_bg_from_local('final.png') 

import speech_recognition as sr
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from langdetect import detect
from pydub import AudioSegment
from pydub.utils import which

AudioSegment.converter = which("ffmpeg")
AudioSegment.ffprobe = which("ffprobe")

# Loading models for sentiment analysis
models = {
    "hi": ("ai4bharat/indic-bert", "positive", "negative"),  # Hindi
    "bn": ("ai4bharat/indic-bert", "positive", "negative"),  # Bengali
    "te": ("ai4bharat/indic-bert", "positive", "negative"),  # Telugu
    "mr": ("ai4bharat/indic-bert", "positive", "negative"),  # Marathi
    "ta": ("google/muril-base-cased", "positive", "negative"),  # Tamil
    "ur": ("google/muril-base-cased", "positive", "negative"),  # Urdu
    "gu": ("ai4bharat/indic-bert", "positive", "negative"),  # Gujarati
    "kn": ("ai4bharat/indic-bert", "positive", "negative"),  # Kannada
    "ml": ("ai4bharat/indic-bert", "positive", "negative"),  # Malayalam
    "pa": ("ai4bharat/indic-bert", "positive", "negative"),  # Punjabi
    "or": ("ai4bharat/indic-bert", "positive", "negative"),  # Odia
    "as": ("ai4bharat/indic-bert", "positive", "negative")   # Assamese
}

def load_model(language_code):
    model_name, pos_label, neg_label = models.get(language_code, models["hi"])  # Default to Hindi
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def transcribe_audio(file_path, language_code='en-US'):
    recognizer = sr.Recognizer()
    audio = AudioSegment.from_file(file_path)
    audio = audio.set_frame_rate(16000)
    # Exporting as a WAV file
    audio.export("temp.wav", format="wav")
    with sr.AudioFile("temp.wav") as source:
        audio_data = recognizer.record(source)
        # Specifying the language code for transcription
        text = recognizer.recognize_google(audio_data, language=language_code)
    return text

def identify_language(text):
    lang = detect(text)
    return lang

def analyze_sentiment(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    sentiment = torch.argmax(probabilities, dim=-1).item()
    
    # Debugging output
    st.write(f"Text: {text}")
    #st.write(f"Logits: {logits}")
    #st.write(f"Probabilities: {probabilities}")
    st.write(f"Sentiment: {sentiment}")    
    return sentiment

def main():
    st.title("Bhaav Anubhuti – A Multilingual Audio Sentiment Analysis")
    
    audio_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "m4a"])
    
    if audio_file is not None:
        with open("uploaded_audio.wav", "wb") as f:
            f.write(audio_file.read())
        
        st.audio("uploaded_audio.wav")
        
        initial_text = transcribe_audio("uploaded_audio.wav", 'en-US')
        #st.write(f"Initial text: {initial_text}")
        
        language = identify_language(initial_text)

        language_code_map = {
            "hi": "hi-IN", "bn": "bn-IN", "te": "te-IN", "mr": "mr-IN",
            "ta": "ta-IN", "ur": "ur-IN", "gu": "gu-IN", "kn": "kn-IN",
            "ml": "ml-IN", "pa": "pa-IN", "or": "or-IN", "as": "as-IN"
        }
        language_code = language_code_map.get(language, 'kn-IN') 

        text = transcribe_audio("uploaded_audio.wav", language_code)
        st.markdown('<div class="content-box">', unsafe_allow_html=True)
        st.write(f"Transcribed text: {text}")

        # Loading appropriate model for sentiment analysis
        model, tokenizer = load_model(language)
        sentiment = analyze_sentiment(text, model, tokenizer)
        sentiment_label = "positive" if sentiment == 1 else "negative"
        st.write(f"Detected language: {language}")
        st.write(f"Text: {text}")
        #st.write(f"Sentiment: {sentiment_label}")
        
        if sentiment_label == "positive":
            st.success(f"Sentiment: {sentiment_label}")
            st.balloons()
        else:
            st.error(f"Sentiment: {sentiment_label}")
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
