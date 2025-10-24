import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from conditioned_lstm import ConditionedLSTM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Load your preprocessed dataset
# -----------------------------
data = np.load("../data/processed_midi.npy", allow_pickle=True)  # (num_samples, seq_len, 3)
X = data[:, :-1, :]   # input sequence
y = data[:, 1:, :]    # next-note prediction

# Simulate 3 conditioning variables: tempo, mood, intensity
cond = np.random.rand(len(X), 3).astype(np.float32)

# Convert to tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)
cond_tensor = torch.tensor(cond, dtype=torch.float32)

dataset = TensorDataset(X_tensor, cond_tensor, y_tensor)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# -----------------------------
# Initialize model, loss, optimizer
# -----------------------------
model = ConditionedLSTM(input_size=3, cond_size=3, hidden_size=256, num_layers=2, output_size=3).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# -----------------------------
# Training loop
# -----------------------------
epochs = 25
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for X_batch, cond_batch, y_batch in loader:
        X_batch, cond_batch, y_batch = X_batch.to(device), cond_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        output = model(X_batch, cond_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}] | Loss: {total_loss/len(loader):.4f}")

# -----------------------------
# Save trained model
# -----------------------------
torch.save(model.state_dict(), "../models/conditioned_lstm.pth")
print("✅ Model trained and saved at '../models/conditioned_lstm.pth'")
