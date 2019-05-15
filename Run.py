import DataLoading
import PreprocessAudio
import Network
import torch.nn as nn
import torch.optim as optim
from Utils import *

# spect_dataset = DataLoading.SpectrogramDataset("spectrogram_dataset/", "All_files.csv")
# PreprocessAudio.plot_spectrogram(spect_dataset[4000]["spectrogram"])

learning_rate = 0.001
momentum = 0.9
classes_num = 30

train_set, test_set, cls_dict = DataLoading.load_data(4, 4, 0.2, True, 123, classes_num,
                                                      "spectrogram_dataset/", "All_files.csv")
net = Network.LSTM(44, 44, classes_num)
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)
loss_function = nn.CrossEntropyLoss()

for idx, (inputs, labels) in enumerate(train_set):
    print(idx)
    outputs = net(inputs.type('torch.FloatTensor'))
    loss = loss_function(outputs, labels)
    loss.backward()
    optimizer.step()

