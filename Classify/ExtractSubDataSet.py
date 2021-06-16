import scipy.io as scio
import random
import pickle
import time
from Classify.Utils import flatten_imgs

train_data = scio.loadmat("../DataSet/train_32x32.mat")
test_data = scio.loadmat("../DataSet/test_32x32.mat")
X_train = train_data['X'].transpose((3, 0, 1, 2))
y_train = train_data['y']
X_test = test_data['X'].transpose((3, 0, 1, 2))
y_test = test_data['y']

train_range = list(range(0, X_train.shape[0]))
test_range = list(range(0, X_test.shape[0]))
train_idx = random.sample(train_range, 10000)
test_idx = random.sample(test_range, 5000)

X_train = X_train[train_idx]
y_train = y_train[train_idx]

X_test = X_test[test_idx]
y_test = y_test[test_idx]


print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)

f = open("../DataSet/SampleTrainData", "wb")
pickle.dump((X_train, y_train), f)
f.close()

f = open("../DataSet/SampleTestData", "wb")
pickle.dump((X_test, y_test), f)
f.close()

exit(0)
time.sleep(1)


X_train = flatten_imgs(X_train)
X_test = flatten_imgs(X_test)
print(X_train.shape, X_test.shape)


f = open("../DataSet/TrainData", "wb")
pickle.dump((X_train, y_train), f)
f.close()

f = open("../DataSet/TestData", "wb")
pickle.dump((X_test, y_test), f)
f.close()