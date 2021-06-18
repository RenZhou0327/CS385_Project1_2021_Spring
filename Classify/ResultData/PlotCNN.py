from matplotlib import pyplot as plt
import pickle
import numpy as np
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D

# f = open("cnn_bin.pkl", "rb")
# train_acc, train_loss, test_acc, test_loss = pickle.load(f)
# f.close()
#
# train_loss = np.array(train_loss)
# train_acc = np.array(train_acc) * 100
# test_loss = np.array(test_loss)
# test_acc = np.array(test_acc) * 100
#
# fig = plt.figure(figsize=(8, 6))
# plt.plot(list(range(1, 101)), train_acc, linewidth=3, linestyle='-.', label="Train Acc")
# plt.plot(list(range(1, 101)), test_acc, linewidth=3, label="Test Acc")
# plt.legend(fontsize=15, loc="lower right")
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
# plt.ylabel('Acc(%)', fontsize=15)
# plt.xlabel('Epoch', fontsize=15)
# plt.tight_layout()
# plt.savefig("CNN_Acc.png")
# plt.show()
# exit(0)
#
#
# fig = plt.figure(figsize=(8, 6))
# plt.plot(list(range(1, 101)), train_loss, linewidth=3, linestyle='-.', label="Train Loss")
# plt.plot(list(range(1, 101)), test_loss, linewidth=3, label="Test Loss")
# plt.legend(fontsize=15, loc="upper right")
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
# plt.ylabel('Loss', fontsize=15)
# plt.xlabel('Epoch', fontsize=15)
# plt.tight_layout()
# plt.savefig("CNN_Loss.png")
# plt.show()


f = open("../../DataSet/EMTrainFeatures", "rb")
X_train, y_train = pickle.load(f)
X_train = X_train.detach().numpy()
y_train = y_train.detach().numpy().astype(np.int32)
f.close()
print(X_train.shape, y_train.shape)

f = open("../../DataSet/EMTestFeatures", "rb")
X_test, y_test = pickle.load(f)
X_test = X_test.detach().numpy()
y_test = y_test.detach().numpy().astype(np.int32)
f.close()

y_train %= 10
y_train = y_train.ravel()
y_test %= 10
y_test = y_test.ravel()

tsne = TSNE(n_components=2, verbose=1)
X_train = tsne.fit_transform(X_train, y_train)
# X_test = tsne.fit_transform(X_test, y_test)

print(X_train.shape)
print(X_test.shape)

plt.figure(figsize=(8, 6))
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='rainbow')
# plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='rainbow')
plt.tight_layout()
plt.savefig("TSNE_2D_train.png")
plt.show()
exit(0)

# plt.figure(figsize=(8, 6))
# ax = plt.subplot(projection="3d")
# # ax.scatter3D(X_train[:, 0], X_train[:, 1], X_train[:, 2], c=y_train, cmap='rainbow')
# ax.scatter3D(X_test[:, 0], X_test[:, 1], X_test[:, 2], c=y_test, cmap='rainbow')
# plt.tight_layout()
# plt.savefig("TSNE_3D_test.png")
# plt.show()

