from matplotlib import pyplot as plt
import pickle
import numpy as np
from Classify.DataSet import HogDataSet

train_dataset = HogDataSet("../../DataSet/TrainData")
test_dataset = HogDataSet("../../DataSet/TestData")
_, y_train = zip(*train_dataset)
_, y_test = zip(*test_dataset)
y_train = np.array(y_train).ravel()
y_test = np.array(y_test).ravel()
y_train = y_train % 10
y_test = y_test % 10

train_sta = np.bincount(y_train)
test_sta = np.bincount(y_test)
x = np.array(list(range(0, 10)))

fig = plt.figure(figsize=(8, 6))
plt.bar(x - 0.2, train_sta / y_train.shape[0] * 100, width=0.4, label="Train", alpha=0.3)
plt.bar(x + 0.2, test_sta / y_test.shape[0] * 100, width=0.4, label="Test", alpha=0.3)
plt.legend(fontsize=15, loc="upper right")
plt.xticks(list(range(10)), list(range(10)), fontsize=12)
plt.yticks(fontsize=12)
plt.ylabel('Ratio(%)', fontsize=15)
plt.xlabel('Class', fontsize=15)
plt.tight_layout()
plt.savefig("Class_Distribute.png")
plt.show()

