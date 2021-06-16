import torch
import torch.nn as nn


class CrossEntropyLoss(nn.Module):

    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        pass

    def forward(self, predict, y):
        delta = 1e-9
        return torch.sum(- y * torch.log(predict + delta) - (1 - y) * torch.log(1 - predict + delta)) / y.shape[0]

    def backward(self, predict, y):
        return (predict - y) / y.shape[0]


class MSELoss(nn.Module):

    def __init__(self):
        super(MSELoss, self).__init__()
        pass

    def forward(self, predict, y):
        return torch.sum((predict - y) ** 2) / (2 * y.shape[0])

    def backward(self, X, predict, y, w):
        sigXw = self.sigmoid(torch.matmul(X, w))
        dXw = sigXw * (1 - self.sigmoid(sigXw))
        return torch.matmul(X.T, ((predict - y) * dXw / y.shape[0]))

    def sigmoid(self, x):
        return 1 / (1 + torch.exp(-x))


class RidgeLoss(nn.Module):

    def __init__(self):
        super(RidgeLoss, self).__init__()
        pass

    def forward(self, predict, y, w, lam):
        return torch.sum((predict - y) ** 2) / (2 * y.shape[0]) + lam * torch.matmul(w.T, w).squeeze()

    def backward(self, X, predict, y, w, lam):
        sigXw = self.sigmoid(torch.matmul(X, w))
        dXw = sigXw * (1 - self.sigmoid(sigXw))
        return torch.matmul(X.T, ((predict - y) * dXw / y.shape[0])) + 2 * lam * w

    def partial_w(self, w):
        w[w < 0] = -1
        w[w > 0] = 1
        w[w == 0] = 0
        return w

    def sigmoid(self, x):
        return 1 / (1 + torch.exp(-x))


class LassoLoss(nn.Module):

    def __init__(self):
        super(LassoLoss, self).__init__()
        pass

    def forward(self, predict, y, w, lam):
        # print(w.shape)
        return torch.sum((predict - y) ** 2) / (2 * y.shape[0]) + torch.sum(lam * torch.abs(w))

    def backward(self, X, predict, y, w, lam):
        dw = self.partial_w(w)
        sigXw = self.sigmoid(torch.matmul(X, w))
        dXw = sigXw * (1 - self.sigmoid(sigXw))
        return torch.matmul(X.T, ((predict - y) * dXw / y.shape[0])) + lam * dw

    def partial_w(self, w):
        w[w < 0] = -1
        w[w > 0] = 1
        w[w == 0] = 0
        return w

    def sigmoid(self, x):
        return 1 / (1 + torch.exp(-x))