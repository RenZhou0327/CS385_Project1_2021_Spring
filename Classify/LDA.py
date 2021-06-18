import numpy as np
import pickle
from scipy import stats


class LDA:
    def __init__(self, x_pos, x_neg):
        self.x_pos = x_pos
        self.x_neg = x_neg
        self.num_pos = x_pos.shape[0]
        self.num_neg = x_neg.shape[0]
        self.ratio_pos = self.num_pos / (self.num_pos + self.num_neg)
        self.ratio_neg = 1 - self.ratio_pos

    def train(self):
        mu_pos = np.mean(self.x_pos, axis=0)
        mu_neg = np.mean(self.x_neg, axis=0)
        var_pos = np.cov(self.x_pos.T)
        var_neg = np.cov(self.x_neg.T)
        sw = self.ratio_pos * var_pos + self.ratio_neg * var_neg
        sw_inv = np.linalg.inv(sw)
        beta = np.matmul(sw_inv, (mu_pos - mu_neg))
        new_mu_pos = np.matmul(mu_pos, beta)
        new_mu_neg = np.matmul(mu_neg, beta)
        new_var_pos = np.matmul(np.matmul(beta, var_pos), beta)
        new_var_neg = np.matmul(np.matmul(beta, var_neg), beta)
        self.beta = beta
        self.mu_pos = new_mu_pos
        self.mu_neg = new_mu_neg
        self.std_pos = np.sqrt(new_var_pos)
        self.std_neg = np.sqrt(new_var_neg)

        return self.save_features()

    def save_features(self):
        x_pos_proj = np.matmul(x_pos, self.beta)
        x_neg_proj = np.matmul(x_neg, self.beta)
        return x_pos_proj, x_neg_proj

    def validate(self, x_test):
        new_x_test = np.matmul(x_test, self.beta)
        probs_pos = stats.norm(loc=self.mu_pos, scale=self.std_pos).pdf(new_x_test)
        probs_neg = stats.norm(loc=self.mu_neg, scale=self.std_neg).pdf(new_x_test)
        return probs_pos >= probs_neg


f = open("../DataSet/TrainData", "rb")
X_train, y_train = pickle.load(f)
y_train = y_train % 10
f.close()
print(X_train.dtype, y_train.dtype)
print(X_train.shape, y_train.shape)

f = open("../DataSet/TestData", "rb")
X_test, y_test = pickle.load(f)
y_test = y_test % 10
f.close()
print(X_test.dtype, y_test.dtype)
print(X_test.shape, y_test.shape)

X_pos_dataset = [[] for _ in range(0, 10)]
X_neg_dataset = [[] for _ in range(0, 10)]

for features, label in zip(X_train, y_train):
    for i in range(0, 10):
        if i == label:
            X_pos_dataset[i].append(features)
        else:
            X_neg_dataset[i].append(features)

y_test = y_test.reshape(-1)
models = []
x_pos_list = []
x_neg_list = []
for idx, (x_pos, x_neg) in enumerate(zip(X_pos_dataset, X_neg_dataset)):
    print("model:", idx)
    # print(idx, len(x_pos), len(x_neg), len(x_pos) + len(x_neg))
    x_pos = np.array(x_pos)
    x_neg = np.array(x_neg)
    labels = (y_test == idx)
    model = LDA(x_pos, x_neg)
    models.append(model)
    x_pos_proj, x_neg_proj = model.train()
    x_pos_list.append(x_pos_proj)
    x_neg_list.append(x_neg_proj)
    output = model.validate(X_test)
    correct = np.count_nonzero((output == labels))
    acc = correct / labels.shape[0]
    print(correct, acc)

f = open("ResultData/LDA_Proj.bin", "wb")
pickle.dump((x_pos_list, x_neg_list), f)
f.close()
