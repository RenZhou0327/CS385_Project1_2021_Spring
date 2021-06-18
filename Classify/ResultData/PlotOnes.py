from matplotlib import pyplot as plt
import pickle
import numpy as np


def load_ones(path):
    f = open(path, "rb")
    train_ones, test_ones = pickle.load(f)
    f.close()
    return train_ones, test_ones


train_ones1, test_ones1 = load_ones("LR_ones001.bin")
train_ones2, test_ones2 = load_ones("Ridge_ones001.bin")
train_ones3, test_ones3 = load_ones("Lasso_ones001.bin")
train_ones4, test_ones4 = load_ones("Kernel_ones001.bin")


# fig = plt.figure(figsize=(8, 6))
# plt.plot(list(range(1, 51)), train_ones1, linewidth=2, linestyle='-.', label="Logistic Train")
# plt.plot(list(range(1, 51)), train_ones2, linewidth=2, linestyle='-.', label="Ridge Train")
# plt.plot(list(range(1, 51)), train_ones3, linewidth=2, linestyle='-.', label="Lasso Train")
# plt.plot(list(range(1, 51)), train_ones4, linewidth=2, linestyle='-.', label="Kernel Train")
# plt.legend(fontsize=15, loc="upper right")
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
# plt.ylabel('Number of Ones', fontsize=15)
# plt.xlabel('Epoch', fontsize=15)
# plt.tight_layout()
# plt.savefig("Ones_Train.png")
# plt.show()
# exit(0)

fig = plt.figure(figsize=(8, 6))
plt.plot(list(range(1, 51)), test_ones1, linewidth=2, label="Logistic Test")
plt.plot(list(range(1, 51)), test_ones2, linewidth=2, label="Ridge Test")
plt.plot(list(range(1, 51)), test_ones3, linewidth=2, label="Lasso Test")
plt.plot(list(range(1, 51)), test_ones4, linewidth=2, label="Kernel Test")
plt.legend(fontsize=15, loc="upper right")
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.ylabel('Number of Ones', fontsize=15)
plt.xlabel('Epoch', fontsize=15)
plt.tight_layout()
plt.savefig("Ones_Test.png")
plt.show()