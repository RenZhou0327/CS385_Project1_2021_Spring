import pickle
import time
import numpy as np
import torch
import torch.nn as nn
from Classify.Utils import load_data
from tqdm import tqdm
from Classify.LossFunction import MSELoss
from copy import deepcopy


class KernelRegression(nn.Module):

    def __init__(self, x_train, x_test, lr, device):
        super(KernelRegression, self).__init__()
        self.num_train_samples = x_train.shape[0]
        self.num_test_samples = x_test.shape[0]
        self.input_dim = x_train.shape[1]
        self.lr = lr
        self.device = device
        with torch.no_grad():
            self.W = torch.randn((self.num_train_samples, 1)).to(self.device)
        self.k = self.get_rbf_kernel(x_train, x_train)
        self.k_test = self.get_rbf_kernel(x_test, x_train)

    @torch.no_grad()
    def forward(self, low_idx, high_idx):
        output = torch.matmul(self.k[low_idx: high_idx], self.W)
        # print(self.k[low_idx: high_idx])
        output = self.sigmoid(output)
        return output

    @torch.no_grad()
    def validate(self, low_idx, high_idx):
        output = torch.matmul(self.k_test[low_idx: high_idx], self.W)
        # print("#####")
        # print(self.k_test[low_idx: high_idx])
        output = self.sigmoid(output)
        # exit(0)
        return output

    # @torch.no_grad()
    # def backward(self, dW):
    #     print("here")
    #     print(self.W[:10])
    #     self.W -= self.lr * dW
    #     print(self.W[:10])

    @torch.no_grad()
    def backward(self, output, y, low_idx, high_idx):
        dW = torch.matmul(self.k[low_idx: high_idx].T, output - y)
        self.W -= self.lr * dW

    def update_lr(self, lr):
        self.lr = lr

    def sigmoid(self, x):
        return 1 / (1 + torch.exp(-x))

    @torch.no_grad()
    def get_rbf_kernel(self, x1, x2, sigma=0.01):
        x1_normal = torch.sum(x1 ** 2, dim=-1)
        x2_normal = torch.sum(x2 ** 2, dim=-1)
        k = torch.exp(-sigma * (x1_normal[:, None] + x2_normal[None, :] - 2 * torch.matmul(x1, x2.t())))
        return k

    def get_W_X(self, low_idx, high_idx):
        return deepcopy(self.W), self.k[low_idx: high_idx]


lr = 0.01
epochs = 50
device = torch.device("cpu")
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
criterion = MSELoss().to(device)

train_loader, test_loader, input_dim = load_data(batch_size=5000)
X_train = torch.Tensor()
y_train = torch.Tensor()
for batch_idx, (images, labels) in enumerate(train_loader):
    labels = labels % 10
    X_train = torch.cat((X_train, images))
    y_train = torch.cat((y_train, labels))


X_test = torch.Tensor()
y_test = torch.Tensor()
for batch_idx, (images, labels) in enumerate(test_loader):
    labels = labels % 10
    X_test = torch.cat((X_test, images))
    y_test = torch.cat((y_test, labels))

X_train = torch.cat((X_train, torch.zeros((X_train.shape[0], 1))), dim=1)
X_train = X_train.to(device)
X_test = torch.cat((X_test, torch.zeros((X_test.shape[0], 1))), dim=1)
X_test = X_test.to(device)


batch_size = 128
batches = (y_train.shape[0] - 1) // batch_size + 1
test_batches = (y_test.shape[0] - 1) // batch_size + 1

train_ones_list = [0] * epochs
test_ones_list = [0] * epochs

# models = []
for model_idx in range(0, 10):

    torch.cuda.empty_cache()
    lr = 0.01
    bin_labels = (y_train == model_idx).type(torch.uint8)
    bin_labels = bin_labels.to(device)
    test_bin_labels = (y_test == model_idx).type(torch.uint8)
    test_bin_labels = test_bin_labels.to(device)

    model = KernelRegression(X_train, X_test, lr, device)

    for epoch in range(epochs):
        train_loss = 0
        train_correct = 0
        train_num = 0
        correct_list = []
        one_cnt = 0
        res_cnt = 0
        for batch_idx in range(batches):
            low_idx = batch_idx * batch_size
            high_idx = min(low_idx + batch_size, bin_labels.shape[0])
            output = model(low_idx, high_idx)
            train_num += output.shape[0]
            loss = criterion.forward(output, bin_labels[low_idx: high_idx])
            model.backward(output, bin_labels[low_idx: high_idx], low_idx, high_idx)
            train_loss += loss.item()

            predict = (output >= 0.5).type(torch.uint8)
            correct = torch.sum(predict.eq(bin_labels[low_idx: high_idx])).item()
            train_correct += correct
            correct_list.append(correct)
            one_cnt += torch.sum(bin_labels[low_idx: high_idx])
            res_cnt += torch.sum(predict)

        print("Train Acc and Loss:")
        print(train_correct / train_num)
        print(train_loss)
        # print(correct_list)
        print(one_cnt.item(), res_cnt.item())
        train_ones_list[epoch] += res_cnt.item()

        test_loss = 0
        test_correct = 0
        test_num = 0
        test_correct_list = []
        test_one_cnt = 0
        test_res_cnt = 0
        for batch_idx in range(test_batches):
            low_idx = batch_idx * batch_size
            high_idx = min(low_idx + batch_size, test_bin_labels.shape[0])
            output = model.validate(low_idx, high_idx)
            test_num += output.shape[0]
            loss = criterion.forward(output, test_bin_labels[low_idx: high_idx])
            test_loss += loss.item()

            predict = (output >= 0.5).type(torch.uint8)
            correct = torch.sum(predict.eq(test_bin_labels[low_idx: high_idx])).item()
            test_correct += correct
            test_correct_list.append(correct)
            test_one_cnt += torch.sum(test_bin_labels[low_idx: high_idx])
            test_res_cnt += torch.sum(predict)

        print("Test Acc and Loss:")
        print(test_correct / test_num)
        print(test_loss)
        # print(test_correct_list)
        print(test_one_cnt.item(), test_res_cnt.item())
        test_ones_list[epoch] += test_res_cnt.item()
    # exit(0)

train_ones_list = np.array(train_ones_list) / 10
test_ones_list = np.array(test_ones_list) / 10
print(train_ones_list)
print(test_ones_list)

f = open("ResultData/Kernel_ones001.bin", "wb")
pickle.dump((train_ones_list, test_ones_list), f)
f.close()