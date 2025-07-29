import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn

# Define GRU Model
class RoBERTaGRU(nn.Module):
    def __init__(self, roberta_model, hidden_size=256, output_size=2):
        super(RoBERTaGRU, self).__init__()
        self.roberta = roberta_model
        self.gru = nn.GRU(input_size=768, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        _, h_n = self.gru(hidden_states)
        logits = self.fc(h_n[-1])
        return logits


# Load Model and Tokenizer
@st.cache_resource
def load_model_and_tokenizer():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    roberta_model = AutoModel.from_pretrained("roberta-base", add_pooling_layer=False)
    model = RoBERTaGRU(roberta_model).to(device)
    model.load_state_dict(torch.load("roberta_gru_imdb.pth", map_location=device))
    model.eval()
    return model, tokenizer, device


# Predict Sentiment
def predict_sentiment(model, tokenizer, device, review):
    max_length = 128
    tokens = tokenizer(review, truncation=True, padding="max_length", max_length=max_length, return_tensors="pt")
    input_ids = tokens['input_ids'].to(device)
    attention_mask = tokens['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, predicted_class].item()

    return "Positive" if predicted_class == 1 else "Negative", confidence


# Streamlit App with Enhanced UI
def main():
    # Set Streamlit configuration and title
    st.set_page_config(page_title="Sentiment Spotter", page_icon="🎭", layout="wide")
    st.title("🎭 Sentiment Spotter: AI-powered Review Classification")
    st.write("Analyze movie reviews and determine their sentiment (Positive or Negative). Powered by **RoBERTa + GRU**.")

    # Sidebar for instructions, dataset details, and credits
    with st.sidebar:
        st.header("About")
        st.write("""
        This app uses a **RoBERTa** transformer model combined with a GRU layer for sentiment classification of movie reviews.
        - **Input**: Any text-based movie review.
        - **Output**: Predicted sentiment (Positive/Negative) with confidence.
        """)
        
        st.write("---")
        st.write("### 📌 **How to Use:**")
        st.write("""
        1. Enter a movie review in the text box.
        2. Click the **Analyze Sentiment** button.
        3. View the sentiment and confidence score.
        """)
        
        st.write("---")
        st.write("### 🗂️ Dataset Used")
        st.write("""
        - **Dataset**: IMDB Large Movie Review Dataset
        - **Description**: The IMDB dataset contains 50,000 labeled movie reviews (25,000 for training and 25,000 for testing).
        - **Sentiment Labels**:
          - Positive = 1
          - Negative = 0
        - **Source**: [Kaggle IMDB Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
        """)
        
        st.write("---")
        st.write("🔗 Developed as part of a 5th-semester mini-project.")
        st.write("🎓 **Developed By:**")
        st.write("""
        - Chaitanya N  
        - Deepthi B E  
        - Chandrakala K M  
        """)

    # Model and tokenizer loading
    model, tokenizer, device = load_model_and_tokenizer()

    # Input and layout
    st.write("## 📝 Input Review")
    review = st.text_area("Enter the movie review text here:", height=150)

    # Single Analyze Sentiment button
    if st.button("🔍 Analyze Sentiment"):
        if review.strip() == "":
            st.warning("⚠️ Please enter a review before clicking Analyze.")
        else:
            # Perform sentiment prediction
            sentiment, confidence = predict_sentiment(model, tokenizer, device, review)

            # Display Results
            st.write("## 🎯 Prediction Result")
            if sentiment == "Positive":
                st.success(f"✅ **Sentiment: Positive**")
            else:
                st.error(f"❌ **Sentiment: Negative**")

            st.info(f"🔍 **Confidence Score:** {confidence:.2%}")

    # Footer
    st.write("---")


if __name__ == "__main__":
    main()
