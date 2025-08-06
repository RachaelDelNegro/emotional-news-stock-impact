import streamlit as st
import requests
from bs4 import BeautifulSoup
from transformers import pipeline
import joblib
import numpy as np
import pandas as pd

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

# Load historical predicted returns
historical_df = pd.read_csv("historical_returns.csv")
historical_returns = historical_df["predicted_return"]
historical_mean = historical_returns.mean()
historical_std = historical_returns.std()

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
    st.write(f"**Predicted 3-Day Return:** `{predicted_return:.4f}`")

    # Historical Comparison
    z_score = (predicted_return - historical_mean) / historical_std

    if z_score > 1.5:
        comparison_text = "This predicted return is **significantly higher** than typical past values."
    elif z_score > 0.5:
        comparison_text = "This predicted return is **somewhat higher** than the historical average."
    elif z_score > -0.5:
        comparison_text = "This predicted return is **close to the historical average**."
    elif z_score > -1.5:
        comparison_text = "This predicted return is **somewhat lower** than the historical average."
    else:
        comparison_text = "This predicted return is **significantly lower** than typical past values."

    st.markdown("### Historical Context")
    st.write(comparison_text)
    st.caption(
        f"Historical Mean: `{historical_mean:.4f}` | Std Dev: `{historical_std:.4f}` | Z-Score: `{z_score:.2f}`"
    )

    # Interpretation logic
    if predicted_return > 0.004:
        analysis_text = (
            "The predicted return suggests **a strong positive movement** in the stock market "
            "over the next 3 days. "
            "This may reflect positive sentiment or investor optimism in response to the headline."
        )
    elif predicted_return > 0.002:
        analysis_text = (
            "The predicted return suggests a **slight positive movement** in the stock market "
            "over the next 3 days. This may indicate mild optimism in the market."
        )
    elif predicted_return > -0.002:
        analysis_text = (
            "The predicted return is **close to neutral**, suggesting little to no expected movement "
            "in the market. Headlines like this may not strongly influence investor behavior in the short term."
        )
    elif predicted_return > -0.01:
        analysis_text = (
            "The predicted return suggests a **slight negative movement** in the market."
            "This could indicate mild investor caution or hesitation in response to similar emotional signals."
        )
    else:
        analysis_text = (
            "The predicted return suggests a **strong negative movement** in the market. This could reflect "
            "fear, uncertainty, or pessimism in response to recent news."
        )

    st.markdown("### Interpretation")
    st.write(analysis_text)

    st.caption("Note: Prediction is based on historical emotional and sentiment signals. "
               "It does not imply that the content of the headline is inherently positive or negative.")






