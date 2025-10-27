import torch.nn as nn

class MusicLSTM(nn.Module):
    def __init__(self, vocab_size, emb=64, hidden=128):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb)
        self.lstm = nn.LSTM(emb, hidden, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden, vocab_size)

    def forward(self, x, hidden=None):
        x = self.emb(x)
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out[:, -1, :])  # only last time step
        return out, hidden


