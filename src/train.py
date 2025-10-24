import os
import torch
os.makedirs("../models", exist_ok=True)  
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim

from midi_loader import load_midi_files
from midi_to_tokens import midi_to_sequence
from create_dataset import create_sequences
from model import MusicRNN

# 1. Load MIDI files
midi_data_list = load_midi_files("./data")
print(f"Total MIDI files loaded: {len(midi_data_list)}")

# 2. Convert MIDI files to sequences
sequences = midi_to_sequence(midi_data_list)
print(f"Total MIDI sequences: {len(sequences)}")

# Quick debug: show first few events of first sequence
if sequences:
    print("First 10 events of first sequence:", sequences[0][:10])
else:
    raise ValueError("No sequences were extracted from MIDI files. Check your MIDI files or sequence_length.")

# 3. Create training sequences
X, y = create_sequences(sequences, sequence_length=32)
print(f"Number of training sequences created: {len(X)}")

# Stop if no sequences were created
if len(X) == 0:
    raise ValueError("No sequences were created. Your MIDI files may be too short or sequence_length is too large.")

# 4. Convert to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# 5. DataLoader
dataset = TensorDataset(X_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
print(f"DataLoader ready with {len(dataloader)} batches")

# 6. Model, loss, optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MusicRNN(input_size=3, hidden_size=128, num_layers=2).to(device)
criterion = nn.MSELoss()  # predicting continuous values
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 7. Training loop
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

# 8. Save model
os.makedirs("models", exist_ok=True)  # ensures 'models' folder exists in current dir
torch.save(model.state_dict(), "models/music_rnn.pth")
print("Model saved successfully at 'models/music_rnn.pth'")

