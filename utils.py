import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import random
import torch 

IMAGE_SIZE = 64
CHANNELS_IMG = 1

transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)]
    )            
])

transform_autoencoder = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: torch.flatten(x)),     
])
transform_vae = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),   
])
transform_wgan = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)]
    )            
])

def plot(train_loss, val_loss):
    plt.title("Training results: Acc")
    plt.plot(val_loss,label='val_acc')
    plt.plot(train_loss, label="train_acc")
    plt.legend()
    increment = random.randint(0, 50000)
    plt.savefig('./figures/train_res'+ str(increment)+ '.png')
    plt.show()

def img_display(img):
    # img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    npimg = np.transpose(npimg, (1, 2, 0))
    return npimg