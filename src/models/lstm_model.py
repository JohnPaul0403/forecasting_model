import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_dim * 2, 1)  # For BiLSTM

    def forward(self, lstm_out):
        weights = F.softmax(self.attn(lstm_out), dim=1)
        context = torch.sum(weights * lstm_out, dim=1)
        return context, weights
    
class CNNBiLSTMAttention(nn.Module):
    def __init__(self, input_size=4, cnn_filters=32, lstm_hidden=64):
        super(CNNBiLSTMAttention, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=cnn_filters, kernel_size=3, padding=1)
        
        self.bilstm = nn.LSTM(
            input_size=cnn_filters,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.attn = Attention(lstm_hidden)
        self.regressor = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # x: (batch, seq_len, features)
        x = x.permute(0, 2, 1)  # (batch, features, seq_len)
        x = F.relu(self.conv1(x))  # (batch, cnn_filters, seq_len)
        x = x.permute(0, 2, 1)  # (batch, seq_len, cnn_filters)
        
        lstm_out, _ = self.bilstm(x)  # (batch, seq_len, hidden*2)
        context, _ = self.attn(lstm_out)  # (batch, hidden*2)
        
        return self.regressor(context)