import argparse

args = argparse.Namespace(
    lr=1e-4,
    bs=8,
    bs_vae=64,
    train_size=0.8,
    image_size=28,
    data_path="data/dataset/",
    # epoch=30,
    path="./data/Images",
    metadata="./data/metadata_ok.csv",
    path_ae_model = './model/model_autoencoder_ok.ckpt',
    path_vae_model = './model/model_vae_ok.ckpt',
    path_wgan_model = './model/model_wgan_ok.ckpt',
)

batch_size= 64
train_size= 0.8
num_epochs= 30
