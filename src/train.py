import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import os
import argparse
from config import *
from utils import load_pickle

# Define the LSTM Model
class MusicLSTM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(MusicLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=3, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)
        # We only want the output from the final time step
        out = self.fc(out[:, -1, :])
        return out

def train(mode):
    print(f"--- Training on {DEVICE} for mode: {mode} ---")
    
    # Load Data
    try:
        X = torch.load(os.path.join(PROCESSED_DIR, f"{mode}_X.pt"))
        y = torch.load(os.path.join(PROCESSED_DIR, f"{mode}_y.pt"))
        vocab_size = load_pickle(os.path.join(PROCESSED_DIR, f"{mode}_vocab.pkl"))
    except FileNotFoundError:
        print("Processed data not found. Please run preprocess.py first.")
        return

    # Setup DataLoader
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Initialize Model, Loss, Optimizer
    model = MusicLSTM(vocab_size, EMBED_SIZE, HIDDEN_SIZE).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training Loop
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch_X, batch_y in dataloader:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(dataloader):.4f}")

    # Save Model
    model_path = os.path.join(MODELS_DIR, f"lstm_{mode}.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True, choices=['hiphop', 'retro', 'mixed'])
    args = parser.parse_args()
    train(args.mode)