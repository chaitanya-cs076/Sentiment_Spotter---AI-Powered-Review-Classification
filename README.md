# Sentiment_Spotter - AI-Powered-Review-Classification

Overview
Sentiment Spotter is an AI-powered mini project that classifies IMDb movie reviews as positive or negative.
It combines RoBERTa transformer embeddings with a lightweight GRU classifier and includes an interactive Streamlit web app where users can enter their own reviews and instantly see predictions.

Key Features
Hybrid model: RoBERTa for contextual embeddings and GRU for sequence learning

Trained on the IMDB Large Movie Review Dataset (approximately 50,000 labeled reviews)

User-friendly Streamlit interface to test reviews in real time

Displays sentiment (Positive/Negative) with confidence score

Modular structure with separate training script (train_model.py) and app (app.py)

Pre-trained model saved as roberta_gru_imdb.pth for quick loading without retraining

Technologies Used
Python

PyTorch

Hugging Face Transformers (roberta-base)

Streamlit

pandas
