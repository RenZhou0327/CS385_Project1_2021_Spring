import numpy as np
import pickle
import random
from scipy import stats


class EM:
    def __init__(self, X, k, sigma):
        self.X = X
        self.num_samples = X.shape[0]
        self.num_feats = X.shape[1]
        self.k = k
        self.alpha = np.repeat(1.0 / self.k, self.k, axis=0)    # 高斯模型权重
        self.mu = np.mat(random.sample(X.tolist(), k))  # 5 * 32 初始均值
        self.gamma = np.zeros((features.shape[0], k))

        self.sigma = sigma * np.eye(self.num_feats).reshape((-1, self.num_feats, self.num_feats))
        self.sigma = np.repeat(self.sigma, self.k, axis=0)

    def train(self, epochs):
        print(epochs)
        for i in range(epochs):
            print("epochs:", i + 1)
            self.expectation_step()
            self.maximization_step()

    def validate(self, feature):
        prob_sum = 0
        for k in range(self.k):
            prob = self.alpha[k] * self.get_gauss_prob(feature, self.mu[k], self.sigma[k])
            prob_sum += prob
        return prob_sum

    def expectation_step(self):
        gamma = np.empty((self.num_samples, self.k))
        for i in range(self.num_samples):
            for k in range(self.k):
                gamma[:, k] = self.alpha[k] * self.get_gauss_prob(self.X[i], self.mu[k], self.sigma[k])
        self.gamma = gamma / np.sum(gamma, axis=1)[:, None]

    def maximization_step(self):
        n = np.sum(self.gamma, axis=0)
        mu = np.matmul(self.gamma.T, self.X) / n[:, None]
        self.alpha = n / self.num_samples
        sigma = np.empty((self.k, self.num_feats, self.num_feats))
        for k in range(self.k):
            mu_k = self.X - mu[k, :]
            sigma[k, :, :] = np.matmul(mu_k.T, np.multiply(mu_k, self.gamma[:, [k]])) / n[k]
        self.mu = mu
        self.sigma = sigma

    def get_gauss_prob(self, x, mu, sigma):
        mu = np.array(mu).ravel()
        return stats.multivariate_normal(mean=mu, cov=sigma, allow_singular=True).pdf(x)


f = open("../DataSet/EMTrainFeatures", "rb")
X_train, y_train = pickle.load(f)
X_train = X_train.detach().numpy()
y_train = y_train.detach().numpy().astype(np.int32)
f.close()
print(X_train.shape, y_train.shape)

f = open("../DataSet/EMTestFeatures", "rb")
X_test, y_test = pickle.load(f)
X_test = X_test.detach().numpy()
y_test = y_test.detach().numpy().astype(np.int32)
f.close()

X_dataset = [[] for _ in range(10)]
for features, label in zip(X_train, y_train):
    X_dataset[label.item()].append(features)

models = []
for idx, cate_features in enumerate(X_dataset):
    print("model:", idx)
    cate_features = np.array(cate_features)
    model = EM(cate_features, 5, 0)
    models.append(model)
    models[idx].train(1)


correct = 0
tot_num = 0
for idx, (feature, label) in enumerate(zip(X_test, y_test)):
    tot_num += 1
    probs = []
    for model in models:
        # prob_sum = 0
        # for k in range(model.k):
        #     prob = model.alpha[k] * model.get_gauss_prob(feature, model.mu[k], model.sigma[k])
        #     prob_sum += prob
        probs.append(model.validate(feature))
        # probs.append(prob_sum)
    if label == np.argmax(probs):
        correct += 1

print("res", correct, tot_num, correct / tot_num)