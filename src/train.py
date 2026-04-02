import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

import os

os.makedirs("models", exist_ok=True) # This creates the folder if it's missing

df_fake = pd.read_csv("data/Fake.csv")
df_true = pd.read_csv("data/True.csv")

df_fake['label'] = 0
df_true['label'] = 1
df = pd.concat([df_fake, df_true], axis=0).reset_index(drop=True)

# print(df.shape)
# print(df["label"].value_counts())

X = df["title"] + " " + df["text"] 
y = df["label"]

# print("X shape",X.shape)
# print("y shape",y.shape)


# NaN fill karo empty string se
X = X.fillna("")

# Create a mask for non-empty rows
mask = X != ""
X = X[mask]
y = y[mask]

# Reset index
X = X.reset_index(drop=True)
y = y.reset_index(drop=True)

# print("After cleaning:")
# print(y.value_counts())


X_train , X_test , y_train, y_test = train_test_split(
    X,y,
    test_size=0.2,
    random_state=42, # reproducible results
) 

# print("Training size:", X_train.shape)
# print("Testing size:", X_test.shape)

# tf-idf
vectorizer = TfidfVectorizer(max_features=5000)
# to fit vectorizer on training data
X_train_tfidf = vectorizer.fit_transform(X_train)
# to text ony transform data
X_test_tfidf = vectorizer.transform(X_test)

# print("X_train_tfidf shape",X_train_tfidf.shape)
# print("X_test_tfidf shape",X_test_tfidf.shape)

# MODEL TRAINING

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train_tfidf,y_train)
# print("Model trained!")

#  Check accuracy
from sklearn.metrics import accuracy_score

y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test,y_pred)
# print(accuracy)

# classification_report -->>> precision , recall, F1 score
from sklearn.metrics import classification_report

# print(classification_report(y_test,y_pred))


# Now — save the model!
# Concept — why save model?
# Abhi model sirf RAM mein hai — program band karo toh model gone! 😄
# Real world mein model ek baar train karte hain — phir save karke bar bar use karte hain.
# Library: pickle — Python objects ko file mein save karta hai! and "Use joblib for sklearn models — it's more efficient"

import joblib

joblib.dump(model, "models/model.pkl") # dump use for model save 
joblib.dump(vectorizer, "models/vectorizer.pkl")

print("Model saved!")
