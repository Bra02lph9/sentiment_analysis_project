# model_training.py

import os
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)          # Remove HTML tags
    text = re.sub(r'[^a-z\s]', '', text)       # Remove non-letter characters
    text = re.sub(r'\s+', ' ', text).strip()   # Remove extra whitespace
    return text

def load_imdb_data(base_dir):
    texts, sentiments = [], []
    for label in ['pos', 'neg']:
        folder = os.path.join(base_dir, label)
        for fname in os.listdir(folder):
            if fname.endswith('.txt'):
                with open(os.path.join(folder, fname), encoding='utf-8') as f:
                    raw_text = f.read()
                    cleaned_text = clean_text(raw_text)
                    texts.append(cleaned_text)
                sentiments.append(label)
    return pd.DataFrame({'text': texts, 'sentiment': sentiments})

if __name__ == '__main__':
    imdb_train_dir = 'data/aclImdb/train'
    df = load_imdb_data(imdb_train_dir)

    print(df['sentiment'].value_counts())  # Check class balance

    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['sentiment'], test_size=0.2, random_state=42)

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', max_features=5000, ngram_range=(1, 2))),
        ('clf', LogisticRegression(max_iter=1000))
    ])

    print("Training model...")
    pipeline.fit(X_train, y_train)

    acc = pipeline.score(X_test, y_test)
    print(f"Model Accuracy: {acc:.4f}")

    os.makedirs('model', exist_ok=True)
    joblib.dump(pipeline, 'model/sentiment_model.pkl')
    print("✅ Model saved at model/sentiment_model.pkl")
