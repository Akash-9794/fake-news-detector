import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

import os

os.makedirs("models", exist_ok=True) # This creates the folder if it's missing

df_fake = pd.read_csv("data/Fake.csv")
df_true = pd.read_csv("data/True.csv")
# rename 
df_true = df_true.rename(columns={"ttitle": "title"})


print("Fake columns:", df_fake.columns.tolist())
print("True columns:", df_true.columns.tolist())


df_fake['label'] = 0
df_true['label'] = 1
# If title or text is empty, fill it with an empty string so the + doesn't break
# If title or text is empty, fill it with an empty string so the + doesn't break
df_fake['title'] = df_fake['title'].fillna("")
df_fake['text'] = df_fake['text'].fillna("")
df_true['title'] = df_true['title'].fillna("")
df_true['text'] = df_true['text'].fillna("")

# content column banao BEFORE concat so NaN doesn't sneak in
df_fake['content'] = df_fake['title'] + " " + df_fake['text']
df_true['content'] = df_true['title'] + " " + df_true['text']

df = pd.concat([df_fake, df_true], axis=0).reset_index(drop=True)

print(df.columns.tolist())



# Drop any remaining NaN rows in content
df = df.dropna(subset=["content"])

# Filter out empty rows from the dataframe itself
df = df[df["content"].str.strip() != ""]

# print(df.shape)
# print(df["label"].value_counts())

# print("X shape",X.shape)
# print("y shape",y.shape)

# 6. Final check before splitting

X = df["content"]
y = df["label"]

print(f"Total samples: {len(X)}")
print(f"Class distribution:\n{y.value_counts()}")
print(f"Final Class distribution:\n{y.value_counts()}")


# Reset index
X = X.reset_index(drop=True)
y = y.reset_index(drop=True)

# print("After cleaning:")
# print(y.value_counts())


X_train , X_test , y_train, y_test = train_test_split(
    X,y,
    test_size=0.1,
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

model = LogisticRegression()
model.fit(X_train_tfidf,y_train)
print("Model trained!")

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
