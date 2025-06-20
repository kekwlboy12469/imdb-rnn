import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.datasets import imdb
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import streamlit as st

# Load model and word index
model = load_model('imdb_rnn_model.h5')
word_index = imdb.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

# --- Custom CSS Styling ---
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: white;
        font-family: 'Segoe UI', sans-serif;
    }
    body {
        background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
        color: white;
    }
    h1 {
        color: #ff4b1f;
        background: linear-gradient(to right, #ff416c, #ff4b2b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
    }
    .stTextArea textarea {
        background-color: #262730;
        color: white;
        border: 1px solid #ff4b2b;
        border-radius: 8px;
    }
    .stButton>button {
        background-color: transparent;
        border: 2px solid #ff4b2b;
        color: #ff4b2b;
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: bold;
        transition: all 0.3s ease-in-out;
    }
    .stButton>button:hover {
        background-color: #ff4b2b;
        color: black;
    }
    </style>
""", unsafe_allow_html=True)

# --- Helper Functions ---
def preprocess_text(text):
    text = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in text]  # +3 due to reserved indices
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

def predict_sentiment(review):
    processed = preprocess_text(review)
    prediction = model.predict(processed, verbose=0)
    sentiment = 'positive' if prediction[0][0] > 0.5 else 'negative'
    return sentiment, prediction[0][0]

def generate_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='black', colormap='plasma').generate(text)
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    return fig

# --- Streamlit UI ---
st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to predict its sentiment.')

user_input = st.text_area("Movie Review", "Type your review here...")

if st.button('Predict Sentiment'):
    if user_input.strip():
        with st.spinner("Analyzing sentiment..."):
            sentiment, score = predict_sentiment(user_input)
            st.success("Prediction complete!")

            st.markdown(f"### <span style='color:#39FF14;'>Sentiment:</span> {sentiment}", unsafe_allow_html=True)
            st.markdown(f"### <span style='color:#FFDD00;'>Score:</span> {score:.2f}", unsafe_allow_html=True)

            # Add emoji
            emoji = "😊" if sentiment == "positive" else "😠"
            st.markdown(f"### Sentiment Emoji: {emoji}")

            # Show Word Cloud
            st.markdown("### Review Word Cloud")
            fig = generate_wordcloud(user_input)
            st.pyplot(fig)
    else:
        st.warning("Please enter a review to analyze.")


