import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

class AutoEncoder(nn.Module):
    def __init__(self, input_shape, latent_dim=128):
        super().__init__()
        self.encoder_l1 = nn.Linear(in_features=input_shape, out_features=latent_dim)
        self.encoder_l2 = nn.Linear(in_features=latent_dim, out_features=latent_dim)

        self.decoder_l1 = nn.Linear(in_features=latent_dim, out_features=latent_dim)
        self.decoder_l2 = nn.Linear(in_features=latent_dim, out_features=input_shape)
    
    def run_decoder(self, latent):
        output = F.relu(self.decoder_l1(latent))
        x_hat = F.relu(self.decoder_l2(output))
        return x_hat
        
    def run_encoder(self, x):
        output = F.relu(self.encoder_l1(x))
        latent = F.relu(self.encoder_l2(output))
        return latent 

    def forward(self, x):
        latent = self.run_encoder(x)
        x_hat = self.run_decoder(latent)
        return x_hat

class ConvVAE(nn.Module):
    def __init__(self, image_channels=1, kernel_size=4, latent_dim=16, init_channels=8):
        super(ConvVAE, self).__init__()
        self.enc1 = nn.Conv2d(
            in_channels=image_channels,
            out_channels = init_channels,
            kernel_size=kernel_size,
            stride=2,
            padding=1
        )
        self.enc2 = nn.Conv2d(
            in_channels=init_channels,
            out_channels=init_channels*2,
            kernel_size=kernel_size,
            stride=2,
            padding=1
        )
        self.enc3 = nn.Conv2d(
            in_channels=init_channels*2,
            out_channels=init_channels*4,
            kernel_size=kernel_size,
            stride=2,
            padding=1
        )
        self.enc4 = nn.Conv2d(
            in_channels=init_channels*4,
            out_channels=64,
            kernel_size=kernel_size,
            stride=2,
            padding=0
        )
        self.fc1 = nn.Linear(64, 128)
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_log_var = nn.Linear(128, latent_dim)
        self.fc2 = nn.Linear(latent_dim, 64)

        # decoder
        self.dec1 = nn.ConvTranspose2d(
            in_channels=64,
            out_channels=init_channels*8,
            kernel_size=kernel_size,
            stride=1,
            padding=0
        )
        self.dec2 = nn.ConvTranspose2d(
            in_channels=init_channels*8,
            out_channels=init_channels*4,
            kernel_size=kernel_size,
            stride=2,
            padding=1
        )
        self.dec3 = nn.ConvTranspose2d(
            in_channels=init_channels*4,
            out_channels=init_channels*2,
            kernel_size=kernel_size,
            stride=2,
            padding=1
        )
        self.dec4 = nn.ConvTranspose2d(
            in_channels=init_channels*2,
            out_channels=image_channels,
            kernel_size=kernel_size,
            stride=2,
            padding=1
        )

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        sample = mu + (eps * std)
        return sample

    def forward(self, x):
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        x = F.relu(self.enc3(x))
        x = F.relu(self.enc4(x))
        batch, _, _, _ = x.shape
        x = F.adaptive_avg_pool2d(x, 1).reshape(batch, -1)
        hidden = self.fc1(x)
        mu = self.fc_mu(hidden)
        #get the latent vector through reparametarization
        log_var = self.fc_log_var(hidden)
        z = self.reparameterize(mu, log_var)
        z = self.fc2(z)
        z = z.view(-1, 64, 1, 1)
        # decoding
        x = F.relu(self.dec1(z))
        x = F.relu(self.dec2(x))
        x = F.relu(self.dec3(x))
        reconstruction = torch.sigmoid(self.dec4(x))
        return reconstruction, mu, log_var


class Discriminator(nn.Module):
    def __init__(self, channels_img, features_d):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            # input: N x channels_img x 64 x 64
            nn.Conv2d(
                channels_img, features_d, kernel_size=4, stride=2, padding=1
            ),
            nn.LeakyReLU(0.2),
            # _block(in_channels, out_channels, kernel_size, stride, padding)
            self._block(features_d, features_d * 2, 4, 2, 1),
            self._block(features_d * 2, features_d * 4, 4, 2, 1),
            self._block(features_d * 4, features_d * 8, 4, 2, 1),
            # After all _block img output is 4x4 (Conv2d below makes into 1x1)
            nn.Conv2d(features_d * 8, 1, kernel_size=4, stride=2, padding=0),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.disc(x)


class Generator(nn.Module):
    def __init__(self, channels_noise, channels_img, features_g):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            # Input: N x channels_noise x 1 x 1
            self._block(channels_noise, features_g * 16, 4, 1, 0),  # img: 4x4
            self._block(features_g * 16, features_g * 8, 4, 2, 1),  # img: 8x8
            self._block(features_g * 8, features_g * 4, 4, 2, 1),  # img: 16x16
            self._block(features_g * 4, features_g * 2, 4, 2, 1),  # img: 32x32
            nn.ConvTranspose2d(
                features_g * 2, channels_img, kernel_size=4, stride=2, padding=1
            ),
            # Output: N x channels_img x 64 x 64
            nn.Tanh(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)
