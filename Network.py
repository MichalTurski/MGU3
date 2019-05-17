import torch.nn as nn
import torch.nn.functional as F


class LSTM(nn.Module):
    def __init__(self, in_size, hidden_size, out_size):
        self.conv_channels = 8
        self.in_size = in_size

        super(LSTM, self).__init__()
        self.convolution = nn.Conv2d(1, self.conv_channels, 3, padding=1, groups=1)
        self.lstm = nn.LSTM(in_size * self.conv_channels, hidden_size)
        self.fc = nn.Linear(hidden_size, out_size)

    def forward(self, input):
        input = input.unsqueeze(3).permute(0, 3, 1, 2)
        conv_out = F.relu(self.convolution(input))
        lstm_in = conv_out.view(-1, self.conv_channels * self.in_size, 44).permute(2, 0, 1)
        lstm_out, _ = self.lstm(lstm_in)
        fc_out = self.fc(lstm_out[-1, :, :]) # use only last output vector.
        tag_scores = F.log_softmax(fc_out, dim=1)
        return tag_scores
