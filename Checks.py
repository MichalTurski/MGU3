import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc


def test_loss(test_loader, net, device, loss_function):
    with torch.no_grad():
        loss = 0.0
        for i, (inputs, labels) in enumerate(test_loader, 0):
            inputs = inputs.type('torch.FloatTensor').to(device)
            # labels.map_(labels, (lambda x, y: x in unknown_idx))
            labels = labels.to(device)

            outputs = net(inputs)
            loss += loss_function(outputs, labels)
        return loss/i


def accuracy(test_loader, net, device, unknown_idx):
    with torch.no_grad():
        correct = 0
        total = 0
        for (inputs, labels) in test_loader:
            inputs = inputs.type('torch.FloatTensor').to(device)
            labels.map_(labels, (lambda x, y: -1 if (x in unknown_idx) else x))
            # labels = labels.to(device)

            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            predicted = predicted.to("cpu")
            predicted.map_(predicted, (lambda x, y: -1 if (x in unknown_idx) else x))
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        return correct/total


def plot(train_loss, test_loss, accuracy):
    plt.rcParams["figure.figsize"] = [16,9]
    plt.subplot(211)
    plt.plot(train_loss, linestyle='-.', label='training')
    plt.plot(test_loss, linestyle='-', label='test')
    plt.legend()
    plt.ylabel('loss')
    plt.xlabel('epochs')
    axes = plt.gca()
    axes.set_ylim(bottom=0)
    plt.subplot(212)
    plt.plot(accuracy, linestyle='-', label='training')
    axes = plt.gca()
    axes.set_ylim(bottom=0)
    plt.show()


def roc_curves(net, test_loader, cls_dict, device):
    with torch.no_grad():
        outputs = []
        labels_vec = []
        for (inputs, labels) in test_loader:
            inputs = inputs.permute(2, 0, 1).type('torch.FloatTensor').to(device)
            outputs_cpu = net(inputs).cpu()
            for output in outputs_cpu:
                outputs.append(output.detach().numpy())
            for label in labels:
                labels_vec.append(label.detach().numpy())
    outputs_array = np.array(outputs)
    labels_array = np.array(labels_vec)
    for i in range(len(cls_dict) - 1): # -1 is for unknown class
        digit_pred = outputs_array[:, i]
        y_expected = labels_array == i
        fpr, tpr, thresholds = roc_curve(y_expected, digit_pred)
        roc_auc = auc(fpr, tpr)
        plt.rcParams["figure.figsize"] = [16, 9]
        plt.plot(fpr, tpr, lw=2, alpha=0.9, color='r', label='ROC curve')
        plt.plot([0, 1], [0, 1], linestyle='--', lw=1, color='g', label='Random classifier', alpha=0.4)
        plt.title(f"ROC curve for class {cls_dict[i]}, AUC = {roc_auc}")
        plt.legend(loc='lower right')
        plt.show()

