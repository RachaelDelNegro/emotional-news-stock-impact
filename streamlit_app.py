# Rachael DelNegro (sep8vb)

import streamlit as st
import requests
from bs4 import BeautifulSoup
from transformers import pipeline
import joblib
import numpy as np

# Load RF model and scalar
def load_model_and_scaler():
    model = joblib.load('final_rf_model.pkl')
    scaler = joblib.load('final_scaler.pkl')
    return model, scaler

feature_columns = joblib.load("model_features.pkl")
# Load Hugging Face pipelines
@st.cache_resource
def load_pipelines():
    emotion_pipe = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)
    sentiment_pipe = pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")
    return emotion_pipe, sentiment_pipe

model, scaler = load_model_and_scaler()
emotion_classifier, sentiment_classifier = load_pipelines()



# Streamlit UI

st.title("Headline Emotion & Market Return Predictor")

st.markdown("Enter a news **headline** below and get instant emotion + sentiment analysis")

headline_input = st.text_input("Enter headline:")

if headline_input:
    with st.spinner("Analyzing headline..."):
        emotion_scores = emotion_classifier(headline_input)[0]
        sentiment_result = sentiment_classifier(headline_input)[0]

        top_emotion = max(emotion_scores, key=lambda x: x['score'])
        top_emotion_label = top_emotion['label']
        emotion_dict = {e['label'].lower(): e['score'] for e in emotion_scores}

        input_dict = {
            "anger": emotion_dict.get("anger", 0.0),
            "disgust": emotion_dict.get("disgust", 0.0),
            "fear": emotion_dict.get("fear", 0.0),
            "joy": emotion_dict.get("joy", 0.0),
            "sadness": emotion_dict.get("sadness", 0.0),
            "surprise": emotion_dict.get("surprise", 0.0),
            "neutral": emotion_dict.get("neutral", 0.0),
            "sentiment_neg": 1 if sentiment_result == "NEGATIVE" else 0,
            "sentiment_pos": 1 if sentiment_result == "POSITIVE" else 0,
            "emotion_max": max(emotion_dict.values()),
        }

        feature_vector = [input_dict.get(col, 0.0) for col in feature_columns]

        scaled_vector = scaler.transform([feature_vector])
        predicted_return = model.predict(scaled_vector)[0]
        
    st.markdown("### Results")
    st.write(f"**Emotion:** {top_emotion_label} ({round(top_emotion['score'], 3)})")
    st.write(f"**Sentiment:** {sentiment_result['label']} ({round(sentiment_result['score'], 3)})")
    st.markdown(f"Predicted 3-Day Return: <span style='font-size:32px; color:green;'>{predicted_return:.4f}</span>", unsafe_allow_html=True)


    st.caption("Note: Prediction is based on historical emotional and sentiment signals.")



