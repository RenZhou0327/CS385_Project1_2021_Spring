import numpy as np
from sklearn.svm import SVC
from Classify.DataSet import HogDataSet
from sklearn.decomposition import KernelPCA
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

train_dataset = HogDataSet("../DataSet/TrainData")
test_dataset = HogDataSet("../DataSet/TestData")
X_train, y_train = zip(*train_dataset)
X_test, y_test = zip(*test_dataset)
X_train, y_train = np.array(X_train), np.array(y_train).ravel()
X_test, y_test = np.array(X_test), np.array(y_test).ravel()

print("Start PCA")
kpca = KernelPCA(n_components=2, kernel="rbf")
X_train = kpca.fit_transform(X_train, y_train)
X_test = kpca.transform(X_test)
print("End PCA")

c = 1.0
svm = SVC(C=c, kernel='sigmoid', random_state=37, gamma='scale')
svm.fit(X_train, y_train)
acc = svm.score(X_test, y_test)
print(acc)

# 2 3 10 50 100 500 1000
# C=0.5 0.3574 0.4064 0.697 0.817 0.8202 0.8054 0.8058
# C=0.1 0.357 0.407 0.679 0.7784 0.776 0.7484 0.7454
# C=1.0 0.3568 0.4082 0.701 0.8256 0.8324 0.822 0.8232

# Linear 0.3326 0.3548 0.6014 0.7196 0.7258 0.7312 0.7314
# Cosine 0.253 0.2846 0.5234 0.7484 0.7728 0.7874 0.791

# plt.figure(figsize=(8, 6))
# ax = plt.subplot(projection="3d")
# # ax.scatter3D(X_train[:, 0], X_train[:, 1], X_train[:, 2], c=y_train, cmap='rainbow')
# ax.scatter3D(X_test[:, 0], X_test[:, 1], X_test[:, 2], c=y_test, cmap='rainbow')
# plt.tight_layout()
# plt.savefig("./ResultData/PCA_3D_test.png")
# exit(0)

# plt.figure(figsize=(8, 6))
# # plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='rainbow')
# plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='rainbow')
# plt.tight_layout()
# plt.savefig("./ResultData/PCA_2D_test.png")
# exit(0)

