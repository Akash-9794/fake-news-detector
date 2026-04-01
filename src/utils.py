import pandas as pd
import re
import nltk
from nltk.corpus import stopwords





# text processing
nltk.download('stopwords')
nltk.download('punkt') # split sentense into words tokenes

# Load Stop Words: It imports stopwords from nltk.corpus and creates a set of English stop words
# from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
# print(stop_words)



def clean_text(text):

    # NaN check karo
    if pd.isna(text):
        return ""
    text = text.lower()
    # re.sub() for Cleaning: Agar aapko unwanted characters hatane hain, toh re.sub(pattern, replacement, text) sabse best function hai.

    text = re.sub(r'http\S+','',text)
    text = re.sub(r'[^a-z\s]','',text) # ^ = not means remove all except a-z
    # remove stopwords
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)