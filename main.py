from operator import mod
from numpy import require
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
# from data.dataloader import CustomDataSet
import torchvision.datasets as datasets

from config import args
from models import AutoEncoder, ConvVAE, Generator, Discriminator
from utils import transform_vae, transform_wgan, transform_autoencoder
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import torch.nn as nn
from train import train_autoencoder, fully_train_vae, train_wgan
from utils import plot
import os

if  __name__ == '__main__':
    print('---------------------------------------------')
    print('hello everyone, this project want to show you the difference between WGAN and VAE')
    print('---------------------------------------------')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('You are using '+ device + ' device')
    print('let start')

    metadata = args.metadata
    path = args.path
    train_size = args.train_size
    batch_size = args.bs
    batch_size_vae = args.bs_vae
    image_size = args.image_size
    data_path = args.data_path
    path_ae = args.path_ae_model
    path_vae = args.path_vae_model
    path_wgan = args.path_wgan_model

    if not os.path.exists(path_ae):
        # CHANNELS_IMG = 1, because we use Mnist
        train_dataset = datasets.MNIST(
            root=data_path, train=True, transform=transform_autoencoder, download=True
        )
        # CHANNELS_IMG = 1, because we use Mnist
        test_dataset = datasets.MNIST(
            root=data_path, train=False, transform=transform_autoencoder, download=True
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size,)

        #Train Autoencoder Model
        print("We are going to train autoencoder")
        model = AutoEncoder(input_shape=image_size*image_size)
        mse = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        n_epochs = 100

        model_trained = train_autoencoder(model, train_loader, optimizer,mse, n_epochs, device)

    if not os.path.exists(path_vae):
        print('let train vae ')
        # let train out Variational autoencoding now
        train_dataset = datasets.MNIST(
            root=data_path, train=True, transform=transform_vae, download=True
        )

        # CHANNELS_IMG = 1, because we use Mnist
        test_dataset = datasets.MNIST(
            root=data_path, train=False, transform=transform_vae, download=True
        )

        trainloader = DataLoader(train_dataset, batch_size=batch_size_vae, shuffle=True)
        testloader = DataLoader(test_dataset, batch_size=batch_size_vae, shuffle=True)
        lr = 0.001
        epochs = 50
        model = ConvVAE().to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.BCELoss(reduction='sum')
        # print(trainloader.batch_size, testloader.batch_size)
        fully_train_vae(model, trainloader, testloader, train_dataset, test_dataset, optimizer, criterion, epochs, device)
    
    if not os.path.exists(path_wgan):
        print('let train wgan')
        LEARNING_RATE = 5e-5
        BATCH_SIZE = 64
        IMAGE_SIZE = 64
        CHANNELS_IMG = 1
        Z_DIM = 100 
        NUM_EPOCHS = 5
        FEATURES_DISC = 64
        FEATURES_CRITIC = 64
        FEATURES_GEN = 64
        critic_iterations = 5
        WEIGHT_CLIP = 0.01

        # initializate optimizer
        # initialize gen and disc/critic

        gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
        critic = Discriminator(CHANNELS_IMG, FEATURES_CRITIC).to(device)
        
        opt_gen = optim.RMSprop(gen.parameters(), lr=LEARNING_RATE)
        opt_critic = optim.RMSprop(critic.parameters(), lr=LEARNING_RATE)
        dataset = datasets.MNIST(
            root=data_path, train=True, transform=transform_wgan, download=True
        )
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

        train_wgan(gen, critic, opt_gen, opt_critic, loader, NUM_EPOCHS, critic_iterations, Z_DIM, WEIGHT_CLIP, device)
    print('everything is already train, just predict or generate something')