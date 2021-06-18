from torch.utils.data import Dataset
import pickle


class HogDataSet(Dataset):

    def __init__(self, data_path, transform=None):
        f = open(data_path, "rb")
        X_data, y_data = pickle.load(f)
        f.close()
        self.data = list(zip(X_data, y_data))
        self.transform = transform

    def __getitem__(self, item):
        img, label = self.data[item]
        img = img.astype("float32")
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.data)


class ImageDataSet(Dataset):

    def __init__(self, data_path, transform=None):
        f = open(data_path, "rb")
        X_data, y_data = pickle.load(f)
        f.close()
        self.data = list(zip(X_data, y_data))
        self.transform = transform

    def __getitem__(self, item):
        img, label = self.data[item]
        img = img.astype("uint8")
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    svnh = HogDataSet("../DataSet/TrainData")
