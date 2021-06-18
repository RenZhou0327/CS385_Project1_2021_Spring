import torch
from torch import nn, optim
from torch.nn import CrossEntropyLoss
import pickle
from tqdm import tqdm
import time
from Classify.Utils import load_sample_data


class MyModel(nn.Module):

    def __init__(self, num_class=10):
        super(MyModel, self).__init__()

        self.features = torch.Tensor().to(device)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #
            nn.Conv2d(in_channels=16, out_channels=128, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(512, 32),
            nn.ReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(32, num_class),
            nn.ReLU()
        )

    def forward(self, imgs):
        imgs = self.conv(imgs)
        imgs = imgs.view(imgs.shape[0], -1)
        feature = self.fc1(imgs)
        self.cat_features(feature)
        output = self.fc2(feature)
        return output

    def cat_features(self, feature):
        self.features = torch.cat((self.features, feature))

    def reset_features(self):
        self.features = torch.Tensor().to(device)

    def get_features(self):
        return self.features


train_loader, test_loader = load_sample_data()

lr = 0.001
epochs = 100
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
criterion = CrossEntropyLoss().to(device)
model = MyModel(num_class=10).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
# lr_schedular = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1 / (epoch + 1))

train_loss_list = []
train_acc_list = []
test_loss_list = []
test_acc_list = []


for epoch in range(0, epochs):
    train_loss = 0
    train_correct = 0
    train_num = 0
    model.reset_features()
    y_train_labels = torch.Tensor().to(device)
    model.train()
    for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc=f"Train epoch{epoch}")):
        optimizer.zero_grad()
        train_num += images.shape[0]
        images = images.to(device)
        labels = labels.to(device)
        labels = labels % 10
        y_train_labels = torch.cat((y_train_labels, labels))
        labels = labels.to(torch.int64).squeeze()
        output = model(images)
        # print(labels.shape, output.shape)
        # exit(0)
        loss = criterion.forward(output, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss
        train_correct += (output.argmax(dim=1) == labels).sum().cpu().item()
    # lr_schedular.step()

    x_train_features = model.get_features().cpu()
    y_train_labels = y_train_labels.cpu()
    print("Train Acc and Loss:")
    # print(model.get_features().cpu().shape, y_train_labels.cpu().shape)
    train_acc = train_correct / train_num
    print(train_acc)
    print(train_loss.cpu().item())
    train_acc_list.append(train_acc)
    train_loss_list.append(train_loss.cpu().item())
    time.sleep(0.1)
    # exit(0)

    test_loss = 0
    test_correct = 0
    test_num = 0
    y_test_labels = torch.Tensor().to(device)
    model.reset_features()
    model.eval()
    for batch_idx, (images, labels) in enumerate(tqdm(test_loader, desc=f"Test epoch{epoch}")):
        test_num += images.shape[0]
        images = images.to(device)
        labels = labels.to(device)
        labels = labels % 10
        y_test_labels = torch.cat((y_test_labels, labels))
        labels = labels.to(torch.int64).squeeze()

        output = model(images)
        loss = criterion.forward(output, labels)
        test_loss += loss
        test_correct += (output.argmax(dim=1) == labels).sum().cpu().item()

    x_test_features = model.get_features().cpu()
    y_test_labels = y_test_labels.cpu()
    print("Test Acc and Loss:")
    # print(model.get_features().cpu().shape, y_test_labels.cpu().shape)
    test_acc = test_correct / test_num
    print(test_acc)
    print(test_loss.cpu().item())
    test_acc_list.append(test_acc)
    test_loss_list.append(test_loss.cpu().item())

    print(x_train_features.shape, y_train_labels.shape)
    print(x_test_features.shape, y_test_labels.shape)

    time.sleep(0.1)

    if (epoch + 1) % 10 == 0:
        torch.save(model.state_dict(), f"./ResultData/cnn_model_epoch{epoch + 1}.pkl")

    # f = open("../DataSet/EMTrainFeatures", "wb")
    # pickle.dump((x_train_features, y_train_labels), f)
    # f.close()
    #
    # f = open("../DataSet/EMTestFeatures", "wb")
    # pickle.dump((x_test_features, y_test_labels), f)
    # f.close()
f = open("./ResultData/cnn_bin.pkl", "wb")
pickle.dump((train_acc_list, train_loss_list, test_acc_list, test_loss_list), f)
f.close()



