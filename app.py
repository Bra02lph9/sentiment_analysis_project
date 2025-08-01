from flask import Flask, render_template, request
import joblib
import os
import re
import string

app = Flask(__name__)

MODEL_PATH = 'model/sentiment_model.pkl'
VECTORIZER_PATH = 'model/vectorizer.pkl'

model = None
vectorizer = None

if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
    except Exception as e:
        print(f"⚠️ Failed to load model or vectorizer: {e}")
else:
    print("⚠️ Model or vectorizer file not found.")

def preprocess_text(text):
    text = text.lower()
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
    text = text.strip()
    return text

@app.route('/', methods=['GET', 'POST'])
def index():
    sentiment = None
    text = ""

    if request.method == 'POST':
        text = request.form.get('text', '').strip()

        if not text:
            sentiment = "No input provided"
        elif model and vectorizer:
            try:
                cleaned_text = preprocess_text(text)
                vect_text = vectorizer.transform([cleaned_text])
                prediction = model.predict(vect_text)
                sentiment = prediction[0]
            except Exception as e:
                print(f"⚠️ Prediction error: {e}")
                sentiment = "Prediction error"
        else:
            sentiment = "Model or vectorizer not available"

    return render_template('index.html', sentiment=sentiment, text=text)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
