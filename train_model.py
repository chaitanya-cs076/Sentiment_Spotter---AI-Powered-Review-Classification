import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
import torch.optim as optim

# Suppress TensorFlow oneDNN warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Debug Logging
print("Step 1: Loading IMDb dataset...")

# Load IMDb dataset
data = pd.read_csv("IMDB Dataset.csv")  # Ensure this file exists in the directory
data['sentiment'] = data['sentiment'].map({'positive': 1, 'negative': 0})

print("Dataset loaded. Preparing dataset class...")

# Define Dataset Class
class IMDbDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.reviews = data['review'].tolist()
        self.labels = data['sentiment'].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, idx):
        review = self.reviews[idx]
        label = self.labels[idx]
        tokens = self.tokenizer(
            review, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt"
        )
        return tokens['input_ids'].squeeze(), tokens['attention_mask'].squeeze(), torch.tensor(label, dtype=torch.long)

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

# Initialize Tokenizer and Dataset
print("Step 2: Initializing tokenizer and dataset...")
tokenizer = AutoTokenizer.from_pretrained("roberta-base")
dataset = IMDbDataset(data, tokenizer, max_length=128)

# Split Dataset
print("Splitting dataset into training and validation sets...")
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)  # Reduced batch size
val_loader = DataLoader(val_dataset, batch_size=8)

# Initialize Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print("Loading RoBERTa model...")
roberta_model = AutoModel.from_pretrained("roberta-base", add_pooling_layer=False)
model = RoBERTaGRU(roberta_model).to(device)
print("Model initialized successfully.")

# Training Setup
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-5)

# Training Loop
print("Starting training...")
epochs = 3
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}...")
    model.train()
    total_loss, total_correct = 0, 0

    for step, (input_ids, attention_mask, labels) in enumerate(train_loader):
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

        # Forward pass
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculate metrics
        total_loss += loss.item()
        total_correct += (outputs.argmax(dim=1) == labels).sum().item()

        if step % 10 == 0:  # Log every 10 steps
            print(f"Step {step}/{len(train_loader)}, Loss: {loss.item():.4f}")

    print(f"Epoch {epoch + 1}: Loss = {total_loss:.4f}, Accuracy = {total_correct / len(train_dataset):.4f}")

# Save the Trained Model
print("Training complete. Saving the model...")
torch.save(model.state_dict(), "roberta_gru_imdb.pth")
print("Model saved as roberta_gru_imdb.pth.")
