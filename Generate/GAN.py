import sys
sys.path.append("..")
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
from torchvision.utils import save_image
from Classify.Utils import load_color_data
import pickle


class Generator(nn.Module):

    def __init__(self, d=64):
        super(Generator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d, 128 * 4 * 4),
        )
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU()
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU()
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 128, 4, 4)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        return x


class Discriminator(nn.Module):
    
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1, bias=False),
            nn.LeakyReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2)
        )
        self.fc = nn.Sequential(
            nn.Linear(4 * 4 * 128, 1),
            # nn.Tanh()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(-1, 4 * 4 * 128)
        x = self.fc(x)
        x = x.view(-1)
        return x



train_loader, test_loader = load_color_data()
epochs = 1000
batch_size = 128
lr_D = 1e-4
lr_G = 1e-4
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
g_dim = 64
batch_num = (9999 // batch_size) + 1
G = Generator(g_dim).to(device)
D = Discriminator().to(device)
optim_G = Adam(G.parameters(), lr_G)
optim_D = Adam(D.parameters(), lr_D)
criterion = nn.BCEWithLogitsLoss()
d_loss_list = []
g_loss_list = []
for epoch in range(epochs):
    torch.cuda.empty_cache()
    optim_D.zero_grad()
    optim_G.zero_grad()
    train_d_loss = None
    train_g_loss = None
    real_imgs = None
    fake_imgs = None
    D.train()
    G.train()
    for batch_idx, (images, labels) in enumerate(tqdm(train_loader)):
        real_input = images.to(device)
        z = torch.randn((images.shape[0], g_dim)).to(device)
        fake_images = G(z)
        fake_input = fake_images.detach()
        # fake_input = fake_imgs.to(device)

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

        # z = torch.randn((images.shape[0], g_dim)).to(device)
        # fake_images = G(z)
        g_labels = torch.ones_like(real_labels)
        output = D(fake_images)

        g_loss = criterion(output, g_labels)
        optim_G.zero_grad()
        g_loss.backward()
        optim_G.step()

        if batch_idx == batch_num - 1:
            train_d_loss = d_loss
            train_g_loss = g_loss
            real_imgs = images
            # fake_imgs = 0.5 * (fake_images + 1)
            # fake_imgs = fake_imgs.clamp(0, 1).view(-1, 1, 32, 32)
            fake_imgs = fake_images.cpu()
    print("Epoch:", epoch + 1, "\t", "D_Loss:", train_d_loss.item(), "\t", "G_Loss", train_g_loss.item())
    d_loss_list.append(train_d_loss.item())
    g_loss_list.append(train_g_loss.item())
    if epoch == 0:
        save_image(real_imgs, "./Images/GAN_REAL/real_pic.png", nrow=4, normalize=True)
    if (epoch + 1) % 10 == 0:
        save_image(fake_imgs, f"./Images/GAN_FAKE/fake_pic{epoch}.png", nrow=4, normalize=True)
        torch.save(G.state_dict(), "./Models/gan_g_model.pkl")
        torch.save(D.state_dict(), "./Models/gan_d_model.pkl")
        f = open("./Models/gan_loss.bin", "wb")
        pickle.dump((d_loss_list, g_loss_list), f)
        f.close()




