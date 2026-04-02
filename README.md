# 📰 Fake News Detector

A machine learning web app that checks whether a news article is **real or fake** — built and deployed by a student learning Python and ML from scratch.

🔗 **Live App:** [fake-news-detector-akash.streamlit.app](https://fake-news-detector-akash.streamlit.app)

---

## 🤔 What Does It Do?

You paste any news text into the app, click a button, and it tells you:
- 🚨 **FAKE NEWS** — with a confidence score
- ✅ **REAL NEWS** — with a confidence score

Simple, fast, and it works!

---

## 🛠️ What I Used

| Tool | Why I Used It |
|---|---|
| Python 3.11+ | Main programming language |
| pandas | To load, merge and clean the dataset |
| scikit-learn | ML model — Logistic Regression + TF-IDF |
| joblib | To save and load the trained model (3x faster than pickle) |
| Streamlit | To build and deploy the web app |
| GitHub | To store and version the code |
| Hugging Face | To host the large dataset files (too big for GitHub) |

---

## 🧠 How It Works (Simple Version)

1. Loaded ~45,000 news articles (real + fake) from two CSV files
2. Renamed the broken `ttitle` column in `True.csv` to `title`
3. Combined `title` + `text` into one `content` column
4. Used **TF-IDF** to convert text into numbers the model can understand
5. Trained a **Logistic Regression** model to learn the difference
6. Saved the model as `.pkl` files using `joblib`
7. Built a Streamlit app that loads the model and predicts on new text

---

## 📊 Final Results

| Metric | Score |
|---|---|
| Accuracy | **99.03%** |
| Precision | 0.99 |
| Recall | 0.99 |
| Total Samples | 44,898 |
| Fake News | 23,481 |
| Real News | 21,417 |

---

## 📁 Project Structure

```
fake-news-detector/
│
├── app.py                  # Streamlit web app (main UI)
│
├── src/
│   ├── __init__.py
│   ├── train.py            # Model training script
│   ├── explore.py          # Data exploration
│   └── utils.py            # Helper functions
│
├── data/
│   ├── Fake.csv            # Fake news dataset
│   ├── True.csv            # Real news dataset
│   ├── cleaned_news.csv    # Cleaned/merged data
│   └── news_data.csv       # Final processed data
│
├── models/
│   ├── model.pkl           # Saved trained model
│   └── vectorizer.pkl      # Saved TF-IDF vectorizer
│
├── setup.sh                # Auto-downloads data from Hugging Face
├── requirements.txt        # Python dependencies
├── .gitignore
├── README.md
└── DEVLOG.md               # Development log / notes
```

---

## 🌋 Problems I Faced (And How I Fixed Them)

This project didn't go smoothly at all. Here are the 4 big errors I ran into:

---

### 1. 🗂️ GitHub Rejected My Dataset (600MB+ files)

**Problem:** GitHub has a 100MB file size limit. My CSV files were way too big to push.

**Fix:** I uploaded the data to **Hugging Face** instead and wrote a `setup.sh` script that downloads the files automatically when the app starts on the server.

```bash
# setup.sh downloads data from Hugging Face on cold start
curl -L "https://huggingface.co/datasets/Akai1014/fake-news-data/resolve/main/Fake.csv?download=true" -o data/Fake.csv
curl -L "https://huggingface.co/datasets/Akai1014/fake-news-data/resolve/main/True.csv?download=true" -o data/True.csv

```

**What I learned:** Always separate your data from your code. Code lives on GitHub, big files live somewhere else.

---

### 2. 🪤 Model Crashed — "Only One Class Found"

**Problem:** `ValueError: This solver needs samples of at least 2 classes`

The model was only seeing Fake news (label 0) and no Real news (label 1). The reason was NaN values in `True.csv` were causing all real news rows to get silently deleted during cleaning.

**Fix:** Fill NaN values **before** merging datasets, and create the `content` column **before** concat — not after.

```python
# ✅ Fill BEFORE merge
df_true['title'] = df_true['title'].fillna("")
df_true['text']  = df_true['text'].fillna("")

# ✅ Create content BEFORE concat
df_true['content'] = df_true['title'] + " " + df_true['text']
```

**What I learned:** Data cleaning order matters a lot. Wrong order = silent data loss.

---

### 3. 🔑 KeyError: 'content' (Column Didn't Exist Yet)

**Problem:** My script was trying to filter on a `content` column that hadn't been created yet.

Also discovered that `True.csv` had a typo — the column was named `ttitle` (double t) instead of `title`!

**Fix:** Rename the broken column first, then create `content` early in the script before any filtering.

```python
# Fix the typo in True.csv
df_true = df_true.rename(columns={"ttitle": "title"})
```

**What I learned:** Always print `df.columns.tolist()` when debugging column errors.

---

### 4. ❄️ App Crashed on First Visit (Cold Start)

**Problem:** When the app starts fresh on Streamlit Cloud, the `model.pkl` file doesn't exist yet — so the app crashed immediately on load.

**Fix:** Added a "Gatekeeper" check in `app.py`. If the model is missing, it runs `setup.sh` and `train.py` automatically, then reloads the app.

```python
if not os.path.exists("models/model.pkl"):
    os.system("bash setup.sh")
    os.system("python src/train.py")
    st.rerun()
```

**What I learned:** Always think about what happens when your app runs for the very first time on a fresh server.

---

## ▶️ Run It Locally

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/fake-news-detector.git
cd fake-news-detector

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download data and train the model
bash setup.sh
python src/train.py

# 4. Run the app
streamlit run app.py
```

---

## 👨‍💻 About

I'm a student learning Machine Learning and Python. This is one of my first complete end-to-end ML projects — from raw messy data all the way to a live deployed app.

I ran into a lot of errors, debugged them one by one, and learned something from each one. That's what `DEVLOG.md` is for — I kept notes of every problem and fix along the way.

If you have any feedback, feel free to open an issue! 🙂
