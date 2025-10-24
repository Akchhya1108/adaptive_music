import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from conditioned_lstm import ConditionedLSTM  # your model

# 1. Load processed dataset
X = np.load("../data/processed_midi.npy", allow_pickle=True)      # shape: (num_samples, seq_len, 3)
y = np.load("../data/processed_midi_y.npy", allow_pickle=True)    # shape: (num_samples, 3)
print("Loaded X:", X.shape, "y:", y.shape)

# 2. Convert to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# 3. DataLoader
dataset = TensorDataset(X_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
print(f"DataLoader ready with {len(dataloader)} batches")

# 4. Model, loss, optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConditionedLSTM(input_size=3, hidden_size=128, num_layers=2).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 5. Training loop
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch_X, batch_y in dataloader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        optimizer.zero_grad()
        output = model(batch_X)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

# 6. Save model
import os
os.makedirs("../models", exist_ok=True)
torch.save(model.state_dict(), "../models/conditioned_lstm_phase1.pth")
print("Model saved at '../models/conditioned_lstm_phase1.pth'")
