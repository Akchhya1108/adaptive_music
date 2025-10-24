import torch
import torch.nn as nn

class ConditionedLSTM(nn.Module):
    def __init__(self, input_size=3, cond_size=3, hidden_size=256, num_layers=2, output_size=3):
        super().__init__()
        self.lstm = nn.LSTM(input_size + cond_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, cond):
        cond_repeated = cond.unsqueeze(1).repeat(1, x.size(1), 1)  # repeat conditioning for each timestep
        x_cond = torch.cat((x, cond_repeated), dim=-1)
        out, _ = self.lstm(x_cond)
        out = self.fc(out)
        return out
