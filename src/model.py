import torch
import torch.nn as nn

class MusicRNN(nn.Module):
    def __init__(self, input_size=3, hidden_size=128, num_layers=2):
        """
        Args:
            input_size: number of features per time step (pitch, start, duration)
            hidden_size: number of hidden units in LSTM
            num_layers: number of LSTM layers
        """
        super(MusicRNN, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_size)  # predict next token (3 features)

    def forward(self, x):
        """
        x: shape (batch_size, sequence_length, input_size)
        """
        out, _ = self.lstm(x)  # out shape: (batch, seq_len, hidden_size)
        out = out[:, -1, :]     # take last time step
        out = self.fc(out)      # shape: (batch, input_size)
        return out


# Quick test
if __name__ == "__main__":
    model = MusicRNN()
    dummy_input = torch.randn(4, 32, 3)  # batch_size=4, sequence_length=32, 3 features
    output = model(dummy_input)
    print("Output shape:", output.shape)  # should be (4, 3)
