
import argparse
import models
import torchvision.transforms as transforms
from PIL import Image
from models import AutoEncoder, ConvVAE, Generator
import torch 
import warnings
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader
from config import args
from utils import transform_autoencoder, transform_vae, transform_wgan
import matplotlib.pyplot as plt 
from utils import img_display
from torchvision.utils import save_image
from pathlib import Path

# Prediction
def image_transform(imagepath):
    test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
    image = Image.open(imagepath)
    imagetensor = test_transforms(image)
    return imagetensor

def predict_autoencoder(image_idx, verbose=False):

    image_size = args.image_size
    data_path = args.data_path
    path_ae_model = args.path_ae_model

    model = AutoEncoder(image_size*image_size)
    model.load_state_dict(torch.load(path_ae_model))
    model.eval()
    test_dataset = datasets.MNIST(
        root=data_path, train=False, transform=transform_autoencoder, download=True
    )
    image = test_dataset[image_idx][0]
    x_hat = model(image)
    Path("figures/ae/").mkdir(parents=True, exist_ok=True)

    save_image(image.detach().cpu().reshape((28, 28)),"figures/ae/" +str(image_idx)+"_ae_original.jpg")
    save_image(x_hat.detach().cpu().reshape((28, 28)), "figures/ae/" +str(image_idx)+"_ae_suggest.jpg")
    

def predict_vae(image_idx, verbose=False):
    image_size = args.image_size
    data_path = args.data_path
    path_model = args.path_vae_model

    model = ConvVAE()
    model.load_state_dict(torch.load(path_model))
    model.eval()
    test_dataset = datasets.MNIST(
        root=data_path, train=False, transform=transform_vae, download=True
    )
    image = test_dataset[image_idx][0]
    image = image.unsqueeze(0)
    x_hat, _, _ = model(image)
    Path("figures/ae/").mkdir(parents=True, exist_ok=True)
    save_image(image.cpu(), "figures/vae/test_" + str(image_idx) + "_ae_original.jpg")
    save_image(x_hat.cpu(), "figures/vae/test_" + str(image_idx) + "_ae_suggest.jpg")
    

def predict_wgan(image_idx, verbose=False):
    CHANNELS_IMG = 1
    Z_DIM = 100 
    FEATURES_GEN = 64

    image_size = args.image_size
    data_path = args.data_path
    path_model = args.path_wgan_model

    model = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN)
    model.load_state_dict(torch.load(path_model, map_location='cpu'))
    model.eval()
    test_dataset = datasets.MNIST(
        root=data_path, train=False, transform=transform_wgan, download=True
    )
    image = test_dataset[image_idx][0]
    image = image.unsqueeze(0)
    cur_batch_size = image.shape[0]
    noise = torch.randn(cur_batch_size, Z_DIM, 1, 1)
    print(image.shape)
    x_hat = model(noise)
    Path("figures/ae/").mkdir(parents=True, exist_ok=True)
    save_image(image.cpu(), "figures/wgan/test_" + str(image_idx) + "_ae_original.jpg")
    save_image(x_hat.cpu(), "figures/wgan/test_" + str(image_idx) + "_ae_suggest.jpg")
    
if  __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-id', '--image_idx', required=True, help="upload the image", type=int)
    parser.add_argument('--type', required=True, help="type of prediction", type=str)
    mains_args = vars(parser.parse_args())
    
    image_idx = mains_args['image_idx']
    type = mains_args['type']

    if type == 'ae':
        print('test with autoencoder')
        predict_autoencoder(image_idx)

    if type == 'vae':
        print('we should test variational autoencoding')
        predict_vae(image_idx)

    if type == 'wgan':
        print('we check with wgan')
        predict_wgan(image_idx)


