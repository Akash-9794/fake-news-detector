
#!/bin/bash

# force to donload 
pip install pandas scikit-learn joblib nitk 


# 1. Create the data and models directories on the cloud server
mkdir -p data
mkdir -p models

# 2. Download the data from Hugging Face (Direct Download)
curl -L "https://huggingface.co/datasets/Akai1014/fake-news-data/resolve/main/Fake.csv -o data/Fake.csv?download=true"
curl -L "https://huggingface.co/datasets/Akai1014/fake-news-data/resolve/main/True.csv -o data/True.csv?download=true"

# 3. Train the model on the server to create model.pkl and vectorizer.pkl
python3 src/train.py

