import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
from torchvision import datasets, transforms
from Classify.Utils import load_sample_data


class Generator(nn.Module):

    def __init__(self, d=128):
        super(Generator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d, 4 * 4 * 512),
        )
        self.bn = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
        )
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU()
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU()
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(128, 3, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 512, 4, 4)
        x = self.bn(x)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        return x


class Discriminator(nn.Module):
    
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 5, 2, 2),
            nn.LeakyReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 5, 2, 2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 5, 2, 2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU()
        )
        self.fc = nn.Linear(4 * 4 * 256, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(-1, 4 * 4 * 256)
        x = self.fc(x)
        x = x.view(-1)
        return x


train_loader, test_loader = load_sample_data()
epochs = 100
batch_size = 128
lr_D = 1e-3
lr_G = 1e-3
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
g_dim = 128
G = Generator(g_dim).to(device)
D = Discriminator().to(device)
# G = G.to(device)
# D = D.to(device)
optim_G = Adam(G.parameters(), lr_G)
optim_D = Adam(D.parameters(), lr_D)
criterion = nn.BCELoss()
# z = torch.randn((2, 128))
# x = G(z)
# output = D(x)
# print(output.shape)
# exit(0)

for epoch in range(epochs):
    torch.cuda.empty_cache()
    optim_D.zero_grad()
    optim_G.zero_grad()
    for batch_idx, (images, labels) in enumerate(tqdm(train_loader)):
        real_input = images.to(device)
        z = torch.randn((images.shape[0], g_dim)).to(device)
        fake_images = G(z)
        fake_input = fake_images.detach()

        real_labels = torch.ones((labels.shape[0], )).to(device)
        fake_labels = torch.zeros((labels.shape[0], )).to(device)

        real_output = D(real_input)
        d_real_loss = criterion(real_output, real_labels)

        fake_output = D(fake_input)
        d_fake_loss = criterion(fake_output, fake_labels)

        d_loss = d_real_loss + d_fake_loss
        optim_D.zero_grad()
        d_loss.backward()
        optim_D.step()

        g_labels = torch.ones_like(real_labels)
        output = D(fake_images)

        g_loss = criterion(output, g_labels)
        optim_G.zero_grad()
        g_loss.backward()
        optim_G.step()



