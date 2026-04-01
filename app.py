import streamlit as st
import joblib
from src.utils import clean_text,stop_words
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

st.title("🔍 Fake News Detector")
st.subheader("Paste a news article or headline below")

news_text = st.text_area("Enter or paste a news to verify fake or real",height = 200)

# clean_text function 


if st.button("Verify"):
    if news_text.strip() == "":
        st.warning("Please enter some text first!")
    else:
       st.write("Checking...")

       model = joblib.load("models/model.pkl")
       vectorizer = joblib.load("models/vectorizer.pkl")
   
       cleaned = clean_text(news_text) # for clean text
       # transform using vectorizer
       text_tfidf = vectorizer.transform([cleaned])
       prediction = model.predict(text_tfidf)[0]
       confidence = model.predict_proba(text_tfidf)
       
    #    show results 
       if prediction == 1:
            st.success(f"✅ REAL NEWS (Confidence: {confidence[0][1]*100:.2f}%)")  
       else:
            st.error(f"❌ FAKE NEWS (Confidence: {confidence[0][0]*100:.2f}%)")  
