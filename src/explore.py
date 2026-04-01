import pandas as pd

from utils import clean_text,stop_words

fake_df = pd.read_csv("Fake.csv")
true_df = pd.read_csv("True.csv")

# print("Fake News Shape:", fake_df.shape)
# print("True News Shape:", true_df.shape)
# print("\nFake News Columns:", fake_df.columns)
# print("\nFirst row of Fake:\n",true_df.head(1) )

fake_df["label"] = 0
true_df["label"] = 1
df = pd.concat([fake_df, true_df], ignore_index=True)
# print(df.shape)
# print(df["label"].value_counts())


df["title"] = df["title"].fillna(df["ttitle"])
df["content"] = df["title"] + " " + df["text"]
# print(df["content"][0])

df.to_csv("news_data.csv",index=False)
# print("Data saved")





# sample = "Donald Trump is the or Sends Out Embarrassing New Year’s Eve Message"
# print(clean_text(sample))

df = pd.read_csv("news_data.csv")
print("cleaning start")
df["clean_content"] = df["content"].apply(clean_text)
print("Cleaning done!")

# print(df["clean_content"][0])
print(df["label"].value_counts())
print(df.columns)

df.to_csv("cleaned_news.csv",index=False)
print("Cleaned Data saved")

# print(df[df["label"]==1]["clean_content"].isna().sum())
# print(df[df["label"]==0]["clean_content"].isna().sum())
# print(df[df["label"]==1]["content"].head())
# print(df[df["label"]==1]["clean_content"].value_counts().head())
# print(df["content"].isna().sum())
# print(df[df["label"]==1]["title"].head())
# print(df[df["label"]==1]["text"].head())
