import sys

from torch.nn.modules.loss import BCELoss
from torchvision import transforms
sys.path.append("..")
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
from torchvision.utils import save_image
from Classify.Utils import load_color_data


class VAE(nn.Module):
    
    def __init__(self, input_channels=3, ldim=32):
        super(VAE, self).__init__()

        self.ldim = ldim

        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
        )
        self.encoder_mean = nn.Linear(32 * 8 * 4, ldim)
        self.encoder_logvar = nn.Linear(32 * 8 * 4, ldim)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.ldim, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, input_channels, 4, 2, 1),
            nn.Tanh(),
        )

    def generate_noise(self, log_var, mean):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mean)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.shape[0], -1)
        log_var = self.encoder_logvar(x)
        mean = self.encoder_mean(x)
        z = self.generate_noise(log_var, mean)
        x = z.view(-1, self.ldim, 1, 1)
        x = self.decoder(x)

        return x, mean, log_var


lr = 1e-3
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
epochs = 1000
batch_size = 128
batch_num = (9999 // batch_size) + 1
model = VAE().to(device)
optimizer = Adam(model.parameters(), lr=lr)
criterion = nn.BCEWithLogitsLoss(reduction='sum')
train_loader, test_loader = load_color_data()
# transform = transforms.Normalize(mean=(0.), std=(0.5))

# train_loss = []
for epoch in range(epochs):
    torch.cuda.empty_cache()
    optimizer.zero_grad()
    train_loss = None
    model.train()
    for batch_idx, (images, labels) in enumerate(tqdm(train_loader)):
        images = images.to(device)
        reconstructed_image, mean, log_var = model(images)
        CE = criterion(reconstructed_image, images)
        KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        loss = CE + KLD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx == batch_num - 1:
            train_loss = loss.item()
            real_imgs = images.cpu()
            fake_imgs = reconstructed_image.cpu()
            
            # print("Loss:", loss.item() / images.shape[0])
    print("Epoch:", epoch + 1, "\t", "Loss:", loss.item())
    if epoch == 0:
        save_image(real_imgs, "./Images/VAE_REAL/real_pic.png", nrow=4, normalize=True)
    if (epoch + 1) % 10 == 0:
        # for idx in range(fake_imgs.shape[0]):
        #     fake_imgs[idx] = transform(fake_imgs[idx])
        save_image(fake_imgs, f"./Images/VAE_FAKE/fake_pic{epoch}.png", nrow=4, normalize=True)
        torch.save(model.state_dict(), "./Models/vae_model.pkl")
