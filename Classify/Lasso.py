import pickle
import time
import numpy as np
import torch
import torch.nn as nn
from Classify.Utils import load_data
from tqdm import tqdm
from Classify.LossFunction import LassoLoss
from copy import deepcopy


class LassoRegression(nn.Module):

    def __init__(self, input_dim, lr, device):
        super(LassoRegression, self).__init__()
        self.input_dim = input_dim
        self.lr = lr
        self.device = device
        with torch.no_grad():
            self.W = torch.randn((input_dim, 1)).to(self.device)

    @torch.no_grad()
    def forward(self, x):
        x = torch.matmul(x, self.W)
        x = self.sigmoid(x)
        return x

    @torch.no_grad()
    def backward(self, dW):
        self.W -= self.lr * dW

    def update_lr(self, lr):
        self.lr = lr

    def sigmoid(self, x):
        return 1 / (1 + torch.exp(-x))

    def get_W(self):
        return deepcopy(self.W)


lr = 0.01
epochs = 50
lam = 0.01
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
criterion = LassoLoss().to(device)

train_loader, test_loader, input_dim = load_data()

models = [LassoRegression(input_dim + 1, lr, device) for i in range(0, 10)]

train_loss_list = []
train_acc_list = []
test_loss_list = []
test_acc_list = []
train_ones_list = []
test_ones_list = []

for epoch in range(0, epochs):
    train_loss = np.zeros((10, ))
    train_correct = np.zeros((10, ))
    one_count = np.zeros((10,))
    res_count = np.zeros((10,))
    train_num = 0
    for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc=f"Train epoch{epoch}")):
        train_num += images.shape[0]
        images = torch.cat((images, torch.zeros((images.shape[0], 1))), dim=1).to(device)
        labels = labels.to(device)
        labels = labels % 10
        for model_idx in range(0, 10):

            output = models[model_idx](images)
            w = models[model_idx].get_W()
            bin_labels = (labels == model_idx).type(torch.uint8)
            loss = criterion.forward(output, bin_labels, w, lam)
            dW = criterion.backward(images, output, bin_labels, w, lam)
            models[model_idx].backward(dW)
            train_loss[model_idx] += loss / images.shape[0]

            predict = (output >= 0.5).type(torch.uint8)
            train_correct[model_idx] += torch.sum(predict.eq(bin_labels))
            one_count[model_idx] += torch.sum(bin_labels)
            res_count[model_idx] += torch.sum(predict)

    if (epoch + 1) % 10 == 0:
        for model in models:
            model.update_lr(lr / 2)
    print("Train Acc and Loss:")
    train_acc = train_correct / train_num
    print(train_acc)
    print(train_loss)
    train_loss_list.append(train_loss)
    train_acc_list.append(train_acc)
    print(one_count)
    print(res_count)
    train_ones_list.append(res_count.mean())
    time.sleep(0.1)

    test_loss = np.zeros((10,))
    test_correct = np.zeros((10,))
    test_one_count = np.zeros((10,))
    test_res_count = np.zeros((10,))
    test_num = 0
    for batch_idx, (images, labels) in enumerate(tqdm(test_loader, desc=f"Test epoch{epoch}")):
        test_num += images.shape[0]
        images = torch.cat((images, torch.zeros((images.shape[0], 1))), dim=1).to(device)
        labels = labels.to(device)
        labels = labels % 10
        for model_idx in range(0, 10):
            output = models[model_idx](images)
            w = models[model_idx].get_W()
            bin_labels = (labels == model_idx).type(torch.uint8)
            loss = criterion.forward(output, bin_labels, w, lam)
            test_loss[model_idx] += loss / images.shape[0]

            predict = (output >= 0.5).type(torch.uint8)
            test_correct[model_idx] += torch.sum(predict.eq(bin_labels))
            test_one_count[model_idx] += torch.sum(bin_labels)
            test_res_count[model_idx] += torch.sum(predict)


    print("Test Acc and Loss:")
    test_acc = test_correct / test_num
    print(test_acc)
    print(test_loss)
    test_acc_list.append(test_acc)
    test_loss_list.append(test_loss)
    print(test_one_count)
    print(test_res_count)
    test_ones_list.append(test_res_count.mean())
    time.sleep(0.1)


f = open("./ResultData/Lasso_ones001.bin", "wb")
pickle.dump((train_ones_list, test_ones_list), f)
f.close()

# f = open("./ResultData/Lasso_lam0001.bin", "wb")
# pickle.dump((train_acc_list, train_loss_list, test_acc_list, test_loss_list), f)
# f.close()
#
# f = open("./ResultData/Lasso_w_0001.bin", "wb")
# w = models[5].get_W()
# pickle.dump(w, f)
# f.close()