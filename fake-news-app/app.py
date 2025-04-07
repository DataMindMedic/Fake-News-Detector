import streamlit as st
import joblib
from utils import clean_text

# Load model + vectorizer
model = joblib.load("xgboost_fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# App layout
st.title("📰 Fake News Detector")
st.subheader("Paste a news article below and find out if it's FAKE or REAL")

# Text input
user_input = st.text_area("Enter the news article content here:", height=200)

# Button
if st.button("Detect"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned = clean_text(user_input)
        vect_text = vectorizer.transform([cleaned])
        prediction = model.predict(vect_text)[0]
        
        if prediction == 1:
            st.success("✅ This article is likely REAL.")
        else:
            st.error("🚨 This article is likely FAKE.")
