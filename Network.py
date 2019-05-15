import torch.nn as nn
import torch.nn.functional as F


class LSTM(nn.Module):
    def __init__(self, in_size, hidden_size, out_size):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(in_size, hidden_size)
        self.fc = nn.Linear(hidden_size, out_size)

    def forward(self, input):
        lstm_out, _ = self.lstm(input)
        fc_out = self.fc(lstm_out[-1, :, :]) # use only last output vector.
        tag_scores = F.log_softmax(fc_out, dim=1)
        return tag_scores
