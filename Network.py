import torch.nn as nn
import torch.nn.functional as F


class LSTM(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, time_len):
        self.conv_channels = 8
        self.in_size = in_size
        self.time_len = time_len

        super(LSTM, self).__init__()
        self.convolution = nn.Conv2d(1, self.conv_channels, 3, padding=1, groups=1)
        self.lstm = nn.LSTM(in_size * self.conv_channels, hidden_size)
        self.fc = nn.Linear(hidden_size, out_size)

    def forward(self, input):
        input = input.unsqueeze(3).permute(0, 3, 1, 2)
        conv_out = F.relu(self.convolution(input))
        lstm_in = conv_out.view(-1, self.conv_channels * self.in_size, self.time_len).permute(2, 0, 1)
        lstm_out, _ = self.lstm(lstm_in)
        fc_out = self.fc(lstm_out[-1, :, :]) # use only last output vector.
        tag_scores = F.log_softmax(fc_out, dim=1)
        return tag_scores


class LSTM_att(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, time_len):
        self.conv_channels_1 = 8
        self.conv_channels_2 = 12

        self.in_size = in_size
        self.time_len = time_len
        self.hidden_size = hidden_size

        super(LSTM_att, self).__init__()
        self.convolution_1 = nn.Conv2d(1, self.conv_channels_1, 3, padding=1, groups=1)
        self.convolution_2 = nn.Conv2d(self.conv_channels_1, self.conv_channels_2, 3, padding=1, groups=1)
        self.lstm = nn.LSTM(in_size * self.conv_channels_2, hidden_size, bidirectional=True)
        # self.fc = nn.Linear(2 * hidden_size * time_len, out_size) # 2 is for bidirectional
        self.fc = nn.Linear(2 * hidden_size, out_size) # 2 is for bidirectional

    def forward(self, input):
        input = input.unsqueeze(3).permute(0, 3, 1, 2)

        conv_mid = F.relu(self.convolution_1(input))
        conv_out = F.relu(self.convolution_2(conv_mid))
        lstm_in = conv_out.view(-1, self.conv_channels_2 * self.in_size, self.time_len).permute(2, 0, 1)
        lstm_out, _ = self.lstm(lstm_in)
        #fc_in = lstm_out.view(-1, self.time_len * 2 * self.hidden_size)
        fc_in = lstm_out[-1, :, :].view(-1, 2 * self.hidden_size) # Get first and last out (bidirectional LSTM)
        # print(fc_in.shape)
        fc_out = self.fc(fc_in)
        tag_scores = F.log_softmax(fc_out, dim=1)
        return tag_scores
