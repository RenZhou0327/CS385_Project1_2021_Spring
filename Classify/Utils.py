import pickle
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from Classify.DataSet import HogDataSet, ImageDataSet
from skimage import feature as ft


def load_data(batch_size=128):
    train_dataset = HogDataSet("../DataSet/TrainData")
    test_dataset = HogDataSet("../DataSet/TestData")
    input_dim = train_dataset[0][0].shape[0]
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    return train_loader, test_loader, input_dim


def load_sample_data():
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    train_dataset = ImageDataSet("../DataSet/SampleTrainData", transform)
    test_dataset = ImageDataSet("../DataSet/SampleTestData", transform)
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


def load_color_data():
    transform = transforms.Compose([
        transforms.ToPILImage(),
        # transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        # transforms.Normalize(mean=(0.2), std=(0.5))
    ])
    train_dataset = ImageDataSet("../DataSet/SampleTrainData", transform)
    test_dataset = ImageDataSet("../DataSet/SampleTestData", transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=False
    )
    return train_loader, test_loader


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def flatten_imgs(data):
    images = []
    for img in tqdm(data, desc="extract hog features"):
        features = ft.hog(img, orientations=9, pixels_per_cell=(4, 4), cells_per_block=(2, 2))
        images.append(features)
    images = np.array(images)
    return images


if __name__ == '__main__':
    train_loader, test_loader = load_sample_data()
    for i, (images, labels) in enumerate(train_loader):
        print(i, images.shape, labels.shape)
        exit(0)
