import DataLoading
# import PreprocessAudio
import Network
import Checks
import torch.nn as nn
import torch.optim as optim
import torch


# spect_dataset = DataLoading.SpectrogramDataset("spectrogram_dataset/", "All_files.csv")
# PreprocessAudio.plot_spectrogram(spect_dataset[4000]["spectrogram"])

learning_rate = 0.001
momentum = 0.9
classes_num = 30
epochs = 4
verbose = True

train_set, test_set, cls_dict = DataLoading.load_data(1, 4, 0.2, True, 123, classes_num,
                                                      "spectrogram_dataset/", "All_files.csv")
net = Network.LSTM(128, 128, classes_num)
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)
loss_function = nn.CrossEntropyLoss()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if verbose: print(f"I'm training on {device}")
net.to(device)


# for idx, (inputs, labels) in enumerate(train_set):
#     print(idx)
#     outputs = net(inputs.type('torch.FloatTensor').permute(2, 0, 1))
#     loss = loss_function(outputs, labels)
#     loss.backward()
#     optimizer.step()

# train_loss = Checks.test_loss(test_set, net, device, loss_function)
# train_loss_list = [train_loss]
# test_loss = Checks.test_loss(test_set, net, device, loss_function)
# test_loss_list = [test_loss]
# accuracy = Checks.accuracy(test_set, net, device)
# accuracy_list = [accuracy]
train_loss_list = []
test_loss_list = []
accuracy_list = []

# if verbose:
#     print(f'[epoch {0}] train loss = {train_loss:.3f}, '
#           f'test loss = {test_loss:.3f}, accuracy = {accuracy * 100:.2f}%')

for epoch in range(epochs):
    train_loss = 0.0
    for i, (inputs, labels) in enumerate(train_set, 0):
        inputs = inputs.permute(2, 0, 1).type('torch.FloatTensor')
        inputs = inputs.to(device)
        labels = labels.to(device)
        print(i)

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
    accuracy = Checks.accuracy(test_set, net, device)
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
    # if test_loss < lowest_loss:
    #     best_net = copy.deepcopy(net)
    #     lowest_loss = test_loss

    prev_loss = test_loss
Checks.plot(train_loss_list, test_loss_list, accuracy_list)
# Checks.roc_curves(best_net, test_loader, classes, device)

