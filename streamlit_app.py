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

        feature_vector = [
            emotion_dict.get("anger", 0),
            emotion_dict.get("disgust", 0),
            emotion_dict.get("fear", 0),
            emotion_dict.get("joy", 0),
            emotion_dict.get("sadness", 0),
            emotion_dict.get("surprise", 0),
            sentiment_result["score"] if sentiment_result["label"] == "POSITIVE" else 1 - sentiment_result["score"]
        ]

        scaled_vector = scaler.transform([feature_vector])
        predicted_return = model.predict(scaled_vector)[0]


st.markdown("### Results")
st.write(f"**Emotion:** {top_emotion_label} ({round(top_emotion['score'], 3)})")
st.write(f"**Sentiment:** {sentiment_result['label']} ({round(sentiment_result['score'], 3)})")
st.write(f"**Predicted 3-Day Return:** `{predicted_return:.4f}`")

st.caption("Note: Prediction is based on historical emotional and sentiment signals.")
