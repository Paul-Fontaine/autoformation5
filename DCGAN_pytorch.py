import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {device} device")


class Generator(nn.Module):
    def __init__(self, z_dim=100):
        """
        Initialize the Generator Module
        :param z_dim: the dimension of the input latent vector
        :param image_size: the size of the output image
        :param conv_dim: the depth of the first layer of the generator
        """
        super(Generator, self).__init__()
        self.linear = nn.Linear(z_dim, 128*7*7)
        self.deconv1 = nn.ConvTranspose2d(128, 128, 5, 2, padding=2, output_padding=1, bias=False)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 5, 2, padding=2, output_padding=1, bias=False)
        self.deconv3 = nn.ConvTranspose2d(64, 32, 5, 1, padding=2, bias=False)
        self.deconv4 = nn.ConvTranspose2d(32, 1, 5, 1, padding=2, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(128)
        self.batch_norm2 = nn.BatchNorm2d(128)
        self.batch_norm3 = nn.BatchNorm2d(64)
        self.batch_norm4 = nn.BatchNorm2d(32)

    def forward(self, x):
        x = self.linear(x)
        x = x.view(-1, 128, 7, 7)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = self.deconv1(x)
        x = self.batch_norm2(x)
        x = F.relu(x)
        x = self.deconv2(x)
        x = self.batch_norm3(x)
        x = F.relu(x)
        x = self.deconv3(x)
        x = self.batch_norm4(x)
        x = F.relu(x)
        x = self.deconv4(x)
        x = F.sigmoid(x)
        return x


class Discrimanator(nn.Module):
    def __init__(self):
        """
        Initialize the Discriminator Module
        :param image_size: the size of the input image
        :param conv_dim: the depth of the first layer of the discriminator
        """
        super(Discrimanator, self).__init__()
        layer_filters = [32, 64, 128, 256]
        self.conv1 = nn.Conv2d(1, 32, 5, 2, padding=2, bias=False)
        self.conv2 = nn.Conv2d(32, 64, 5, 2, padding=2, bias=False)
        self.conv3 = nn.Conv2d(64, 128, 5, 2, padding=2, bias=False)
        self.conv4 = nn.Conv2d(128, 256, 5, 1, padding=2, bias=False)
        self.fc = nn.Linear(4096, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.leaky_relu(x, 0.2)
        x = self.conv2(x)
        x = F.leaky_relu(x, 0.2)
        x = self.conv3(x)
        x = F.leaky_relu(x, 0.2)
        x = self.conv4(x)
        x = F.leaky_relu(x, 0.2)
        x = x.flatten(1)
        x = self.fc(x)
        x = F.sigmoid(x)
        return x


class DCGAN(nn.Module):
    def __init__(self, z_dim=100):
        super(DCGAN, self).__init__()
        self.z_dim = z_dim
        self.generator = Generator(z_dim)
        self.discriminator = Discrimanator()

    def forward(self, x):
        x = self.generator(x)
        x = self.discriminator(x)
        return x

    def train_(self, data_loader, num_epochs=10, lr=0.0002, beta1=0.5, beta2=0.999, use_tensorboard=True):
        if use_tensorboard:
            writer = SummaryWriter()

        self.generator.train()
        self.discriminator.train()
        criterion = nn.BCELoss()
        g_optimizer = optim.Adam(self.generator.parameters(), lr=lr, betas=(beta1, beta2))
        d_optimizer = optim.Adam(self.discriminator.parameters(), lr=lr, betas=(beta1, beta2))
        for epoch in range(num_epochs):
            running_g_loss = 0.0
            running_d_loss = 0.0
            with tqdm(data_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch") as tqdm_loader:
                for i, batch in enumerate(tqdm_loader):
                    # get real images from the dataset
                    real_images, _ = batch
                    batch_size = real_images.size(0)
                    real_images = real_images.to(device)
                    real_labels = torch.ones(batch_size, 1).to(device)

                    # get fake images from the generator
                    z = np.random.uniform(-1, 1, size=(batch_size, self.z_dim))
                    z = torch.from_numpy(z).float().to(device)
                    fake_images = self.generator(z)
                    fake_labels = torch.zeros(batch_size, 1).to(device)

                    # train the discriminator
                    d_optimizer.zero_grad()
                    d_real = self.discriminator(real_images)
                    d_fake = self.discriminator(fake_images)
                    d_real_loss = criterion(d_real, real_labels)
                    d_fake_loss = criterion(d_fake, fake_labels)
                    d_loss = d_real_loss + d_fake_loss
                    d_loss.backward(retain_graph=True)
                    d_optimizer.step()
                    g_optimizer.zero_grad()

                    # train the generator
                    g_optimizer.zero_grad()
                    g_fake = self.discriminator(fake_images)
                    g_loss = criterion(g_fake, real_labels)
                    g_loss.backward()
                    g_optimizer.step()

                    running_g_loss += g_loss.item()
                    running_d_loss += d_loss.item()
                    average_g_loss = running_g_loss / ((i+1) * batch_size)
                    average_d_loss = running_d_loss / ((i+1) * batch_size)
                    tqdm_loader.set_postfix(g_loss=average_g_loss, d_loss=average_d_loss)
                    episode = epoch*len(data_loader) + i
                    if use_tensorboard:
                        writer.add_scalar("Generator Loss", g_loss.item(), episode)
                        writer.add_scalar("Discriminator Loss", d_loss.item(), episode)
                        writer.add_scalar("Generator running Loss", average_g_loss, episode)
                        writer.add_scalar("Discriminator running Loss", average_d_loss, episode)
                    else:
                        print(f"Episode {episode}: Generator Loss: {g_loss.item()}, Discriminator Loss: {d_loss.item()}")

            self.save_weights()

            if use_tensorboard:
                images = self.generate(4, plot=False)
                writer.add_images("Generated Images", images, epoch)

    def load_from_checkpoint(self, generator_path:str = 'checkpoint/generator.pth', discriminator_path:str = 'checkpoint/discriminator.pth'):
        self.generator.load_state_dict(torch.load(generator_path, map_location=device))
        self.discriminator.load_state_dict(torch.load(discriminator_path, map_location=device))

    def generate(self, num_images=16, plot=True):
        self.generator.eval()
        z = np.random.uniform(-1, 1, size=(num_images, self.z_dim))
        z = torch.from_numpy(z).float().to(device)
        fake_images = self.generator(z)
        fake_images = fake_images.detach().cpu().numpy()

        if plot:
            n = math.sqrt(num_images)
            if not n.is_integer():
                n = math.ceil(math.sqrt(num_images))
            else:
                n = int(n)
            fake_images = np.squeeze(fake_images, axis=1)

            fig, axes = plt.subplots(n, n, figsize=(20, 20))
            for i, ax in enumerate(axes.flatten()):
                ax.imshow(fake_images[i], cmap='gray')
                ax.axis('off')
            plt.show()

        return fake_images

    def save_weights(self):
        if not os.path.exists('checkpoint'):
            os.makedirs('checkpoint')
        torch.save(self.generator.state_dict(), 'checkpoint/generator.pth')
        torch.save(self.discriminator.state_dict(), 'checkpoint/discriminator.pth')
