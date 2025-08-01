# Sentiment Analysis Project

This project implements a sentiment analysis model trained on the IMDb movie reviews dataset. It uses a machine learning pipeline with TF-IDF vectorization and logistic regression to classify movie reviews as positive or negative. A Flask web app allows users to input their own reviews and get sentiment predictions.

---

## Project Structure

sentiment_project/  
├── model_training.py         # Script to train and save the sentiment model  
├── app.py                    # Flask web application  
├── data/                     # Raw IMDb dataset (not included in repo)  
├── model/                    # Trained model and vectorizer (not included in repo)  
├── templates/  
│   └── index.html            # Flask HTML template for user interface  
├── static/  
│   └── style.css             # CSS styles for the web app  
├── tests/  
│   └── test_app.py           # Unit tests for Flask app  
├── requirements.txt          # Python dependencies  
└── README.md                 # This file  

---
