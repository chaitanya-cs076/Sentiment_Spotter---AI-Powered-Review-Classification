# **Sentiment Spotter : AI Powered Review Classification**

#### **Overview**
Sentiment Spotter is web application built using Streamlit that classifies IMDb movie reviews as positive or negative.It combines RoBERTa transformer embeddings with a lightweight GRU classifier and includes an interactive Streamlit web app where users can enter their own reviews and instantly see predictions.

---

#### **Key Features** 
- Hybrid model RoBERTa for contextual embeddings and GRU for sequence learning.
- Trained on the IMDB Large Movie Review Dataset (approximately 50,000 labeled reviews).
- User-friendly Streamlit interface to test reviews in real time.
- Displays sentiment (Positive/Negative) with confidence score
- Modular structure with separate training script (train_model.py) and app (app.py)
- Pre-trained model saved as roberta_gru_imdb.pth for quick loading without retraining
     
---

#### **Technologies Used**
- Python
- PyTorch
- Hugging Face Transformers (roberta-base)
- Streamlit
- pandas
  
---

#### **How It Works**
The project uses a pre-trained RoBERTa model to extract contextual embeddings from movie reviews.
These embeddings are passed to a GRU (Gated Recurrent Unit) layer that learns sequential patterns relevant to sentiment classification.
The model is trained on the IMDb dataset and saved as roberta_gru_imdb.pth.
A Streamlit web app loads this trained model and allows users to input their own reviews.
It then predicts the sentiment (positive or negative) and displays a confidence score.

---

#### **Installation and Usage**
1. Clone the repository:
   ```bash
   git clone https://github.com/chaitanya-cs076/Sentiment_Spotter-AI-Powered-Review-Classification.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Sentiment_Spotter-AI-Powered-Review-Classification
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the application:
   ```bash
   streamlit run app.py
   ```
5. Access the application at `http://localhost:8501`.

---

#### **Future Enhancements**
- Integrate real-time traffic data for more accurate travel time estimation.
- Expand the disease prediction system using machine learning models.
- Implement GPS-based live ambulance tracking.
- Enhance scalability to include multiple cities.
