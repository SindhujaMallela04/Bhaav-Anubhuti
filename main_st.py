import streamlit as st
import base64
import speech_recognition as sr
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from pydub import AudioSegment
from pydub.utils import which
from googletrans import Translator

# Set up audio segment
AudioSegment.converter = which("ffmpeg")
AudioSegment.ffprobe = which("ffprobe")

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

add_bg_from_local('final.png')

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
    model_name, pos_label, neg_label = models.get(language_code, models["hi"])  #default to Hindi
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)  # Use slow tokenizer
    return model, tokenizer

def transcribe_audio(file_path, language_code='en-US'):
    recognizer = sr.Recognizer()
    audio = AudioSegment.from_file(file_path)
    audio = audio.set_frame_rate(16000)
    audio.export("temp.wav", format="wav")
    with sr.AudioFile("temp.wav") as source:
        audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data, language=language_code)
    return text

def analyze_sentiment(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    sentiment = torch.argmax(probabilities, dim=-1).item()    
    st.write(f"Text: {text}")
    st.write(f"Sentiment: {sentiment}")    
    return sentiment

def translate_to_english(text):
    translator = Translator()
    translated = translator.translate(text, dest='en')
    return translated.text

def main():
    st.title("Bhaav Anubhuti – A Multilingual Audio Sentiment Analysis")
    
    lan = st.text_input("Enter the language code (e.g., 'hi-IN' for Hindi, 'te-IN' for Telugu, 'ta-IN' for Tamil, 'kn-IN' for Kannada, 'ml-IN' for Malayalam):", value='en')

    audio_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "m4a"])
    
    if audio_file is not None:
        with open("uploaded_audio.wav", "wb") as f:
            f.write(audio_file.read())
        
        st.audio("uploaded_audio.wav")
        
       
        text = transcribe_audio("uploaded_audio.wav", lan)
        st.markdown('<div class="content-box">', unsafe_allow_html=True)
        st.write(f"Transcribed text: {text}")
            
        model_language_code = lan.split('-')[0]

        if lan != "en":
            translated_text = translate_to_english(text)
            st.write(f"Translated Text: {translated_text}")
        else:
            translated_text = text

        model, tokenizer = load_model(lan)
        sentiment = analyze_sentiment(translated_text, model, tokenizer)
        sentiment_label = "positive" if sentiment == 1 else "negative"
        
        if sentiment_label == "positive":
            st.success(f"Sentiment: {sentiment_label}")            
            st.balloons()
        else:
            st.error(f"Sentiment: {sentiment_label}")
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
