import numpy as np
from sklearn.svm import SVC
from Classify.DataSet import HogDataSet

train_dataset = HogDataSet("../DataSet/TrainData")
test_dataset = HogDataSet("../DataSet/TestData")
X_train, y_train = zip(*train_dataset)
X_test, y_test = zip(*test_dataset)
X_train, y_train = np.array(X_train), np.array(y_train).ravel()
X_test, y_test = np.array(X_test), np.array(y_test).ravel()

c = 0.5
svm = SVC(C=c, kernel='rbf', random_state=42, gamma='scale')
svm.fit(X_train, y_train)
acc = svm.score(X_test, y_test)
print(acc)
