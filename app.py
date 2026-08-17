import streamlit as st
import joblib

# Load the saved pipeline
model = joblib.load('fake_news_pipeline.pkl')

# Paste your clean_text function definition here if you used one during training
def clean_text(text):
    return text.lower().strip()

st.title("Fake News Detector 📰")
text_input = st.text_area("Paste News Headline or Article Text:")

if st.button("Analyze"):
    words = text_input.strip().split()
    
    if not text_input.strip():
        st.warning("⚠️ Please enter some text to analyze.")
    elif len(words) < 6:
        st.warning("⚠️ Short inputs lack enough context for accurate classification. Please enter a longer headline or full sentence.")
    else:
        cleaned = clean_text(text_input)
        prob = model.predict_proba([cleaned])[0][1]
        
        if prob >= 0.65:
            st.error(f"🚨 FAKE NEWS DETECTED ({prob*100:.1f}% confidence)")
        else:
            st.success(f"✅ REAL NEWS ({(1-prob)*100:.1f}% confidence)")
