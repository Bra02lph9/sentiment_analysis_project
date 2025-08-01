# tests/test_app.py

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_home_page(client):
    res = client.get('/')
    assert res.status_code == 200
    assert b"IMDb Sentiment Analysis" in res.data

def test_prediction(client):
    res = client.post('/', data={'text': 'I loved this movie, it was fantastic!'})
    assert b"Sentiment:" in res.data
