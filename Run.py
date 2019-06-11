import DataLoading
# import PreprocessAudio
import Network
import Checks
import torch.nn as nn
import torch.optim as optim
import torch
import copy


# spect_dataset = DataLoading.SpectrogramDataset("spectrogram_dataset/", "All_files.csv")
# PreprocessAudio.plot_spectrogram(spect_dataset[4000]["spectrogram"])

learning_rate = 0.001
momentum = 0.9

epochs = 5
verbose = True
batch_size = 16
num_workers = 8
val_split = 0.2
shuffle = True
random_seed = 123
classes_num = 30
time_len = 44 # Each spectrogram contains 44 time-elements

train_set, test_set, cls_dict, unknown_idx = DataLoading.load_data(batch_size, num_workers,
                                                                   val_split, shuffle, random_seed,
                                                                   classes_num,
                                                      "spectrogram_dataset/", "All_files.csv")
net = Network.LSTM_att(128, 70, classes_num, time_len)
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)
loss_function = nn.CrossEntropyLoss()
lowest_loss = float("inf")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if verbose: print(f"I'm training on {device}")
net.to(device)
best_net = copy.deepcopy(net)

accuracy = Checks.accuracy(test_set, net, device, unknown_idx)
accuracy_list = [accuracy]
train_loss = Checks.test_loss(test_set, net, device, loss_function)
train_loss_list = [train_loss]
test_loss = Checks.test_loss(test_set, net, device, loss_function)
test_loss_list = [test_loss]


if verbose:
    print(f'[epoch {0}] train loss = {train_loss:.3f}, '
          f'test loss = {test_loss:.3f}, accuracy = {accuracy * 100:.2f}%')

# train_loss_list = []
# test_loss_list = []
# accuracy_list = []

for epoch in range(epochs):
    train_loss = 0.0
    for i, (inputs, labels) in enumerate(train_set, 0):
        inputs = inputs.type('torch.FloatTensor')
        inputs = inputs.to(device)
        labels = labels.to(device)
        # print(i)

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    train_loss = train_loss / i
    train_loss_list.append(train_loss)
    test_loss = Checks.test_loss(test_set, net, device, loss_function)
    test_loss_list.append(test_loss)
    accuracy = Checks.accuracy(test_set, net, device, unknown_idx)
    accuracy_list.append(accuracy)
    if verbose:
        print(f'[epoch {epoch + 1}] train loss = {train_loss:.3f}, '
              f'test loss = {test_loss:.3f}, accuracy = {accuracy * 100:.2f}%')
    # if test_loss > prev_loss:
    #     loss_rise_count += 1
    #     if loss_rise_count >= loss_rise_threshold:
    #         break
    # else:
    #     loss_rise_count = 0
    #
    if test_loss < lowest_loss:
        print("There is lower loss")
        best_net = copy.deepcopy(net)
        lowest_loss = test_loss

    prev_loss = test_loss
Checks.plot(train_loss_list, test_loss_list, accuracy_list)
Checks.roc_curves(best_net, test_set, cls_dict, device)


