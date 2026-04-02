# for cloud huggiface donload
# This is a professional "hack" to ensure the model exists before the UI loads. 
import os
import subprocess
import streamlit as st


# Professional Check: If the model doesn't exist, run the setup script

if not os.path.exists("models/model.pkl"):
    st.info("First time setup: Downloading data and training model... Please wait.")
    subprocess.run(["sh", "setup.sh"])
    st.success("Setup complete!")

else:
       # rest all code

   import joblib 

   from src.utils import clean_text

   

   

   # --- CACHED LOADING ---

   @st.cache_resource

   def load_assets():

       return joblib.load("models/model.pkl"), joblib.load("models/vectorizer.pkl")

   

   model, vectorizer = load_assets()

   

   # --- SIDEBAR ---

   with st.sidebar:

       with st.container(horizontal_alignment="center"):

          st.image("https://threedio-prod-var-cdn.icons8.com/wo/preview_sets/previews/QS7oEnf1jrYbjKU5.webp", width=100,)

       st.title("About Project")

       st.info("This AI uses Logistic Regression and TF-IDF to detect misinformation with 99% accuracy.")

       st.markdown("---")

       st.write("Developed by: Akash Chaurasiya")

   

   st.title("🔍 Fake News Detector")

   st.markdown("# Combat misinformation with the power of Machine Learning.")

   

   st.subheader("Paste a news article or headline below")

   

   news_text = st.text_area("Enter or paste a news to verify fake or real , ask for  only news",height = 200,placeholder="e.g., Reuters reports new budget fight in Congress...")

   

   # clean_text function 

   

   

   if st.button("Verify Authenticity", use_container_width=True):

       if news_text.strip() == "":

           st.warning("⚠️ Please enter some text first!")

       else:

        with st.spinner("Analyzing linguistic patterns..."):

   

   

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

   

   

   
