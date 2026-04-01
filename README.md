# 🔍 Fake News Detector

An end-to-end Machine Learning solution designed to combat misinformation by classifying news articles as **Real** or **Fake** using NLP techniques.

## 🚀 Features
- **High Accuracy**: Trained on 44,000+ articles with a 99.03% success rate.
- **Modular Architecture**: Clean, engineered folder structure (Data/Models/Src).
-**Real-time Verification**: Instant prediction via a Streamlit web interface.
- **Explainable AI**: Provides a confidence score for every prediction.

## 🛠️ Tech Stack
- **Language**: Python
- **Libraries**: Pandas, Scikit-Learn, NLTK, Joblib 
- **Algorithm**: Logistic Regression with TF-IDF Vectorization 
- **UI**: Streamlit 

## 📁 Project Structure
```text
fake-news-detector/
├── data/       # Cleaned and Raw CSVs
├── models/     # Saved .pkl files (Model & Vectorizer)
├── src/        # Backend logic (Cleaning & Training)
└── app.py      # Streamlit Entry Point


📊 Performance
Metric	          Score
Accuracy	-->     99.03%
Precision	-->     0.99
Recall	    -->     0.99
F1-Score	-->     0.99
