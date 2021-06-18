import torch
from torch import nn, optim
from torch.nn import CrossEntropyLoss
import pickle
from tqdm import tqdm
import time
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
import cv2
import sys
sys.path.append("../..")
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchvision.transforms import transforms
from Classify.DataSet import ImageDataSet

def load_tmp_data(batch_size=128):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    train_dataset = ImageDataSet("../../DataSet/SampleTrainData", transform)
    test_dataset = ImageDataSet("../../DataSet/SampleTestData", transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=False
    )
    return train_loader, test_loader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

model = MyModel(num_class=10).to(device)
model.load_state_dict(torch.load("cnn_model_epoch100.pkl"))
print(model)
layer = model.conv[3]
print(layer)

cam = GradCAM(model=model, target_layer=layer, use_cuda=True)
train_loader, test_loader = load_tmp_data()
input_images = None
input_labels = None
for idx, (images, labels) in enumerate(test_loader):
    labels = labels % 10
    input_images = images[:64]
    # input_images.requires_grad = True
    input_labels = labels[:64].squeeze().type(torch.int64)
    break

# print(input_images.shape)
# print(input_labels.detach())
# exit(0)
save_image(input_images, "origin.png", nrow=8, normalize=True)
# exit(0)

#
grayscale_cam = cam(input_tensor=input_images, target_category=input_labels, aug_smooth=True)
print(grayscale_cam.shape)
# exit(0)

def save_gradcam_figure(grayscale_cam_img, origin_img, idx):
    visualization = show_cam_on_image(origin_img, grayscale_cam_img, use_rgb=False)
    # print(visualization.shape)
    return visualization
    # exit(0)
    # cv2.imwrite("./GradCamFigs/" + f"Layer8-{idx}.png", visualization)

figs = torch.Tensor()
for i in range(0, 64):
    grayscale_cam_img = grayscale_cam[i, :]
    origin_img = input_images[i, :].numpy().transpose((1, 2, 0))
    # print(grayscale_cam_img.shape)
    # print(origin_img.shape)
    # exit(0)
    fig = save_gradcam_figure(grayscale_cam_img, origin_img, i)
    # fig = torch.Tensor(fig).unsqueeze(0)
    fig = transforms.ToTensor()(fig).unsqueeze(0)
    # print(fig.shape)
    # exit(0)
    figs = torch.cat((figs, fig))
save_image(figs, "result_layer3.png", nrow=8, normalize=True)