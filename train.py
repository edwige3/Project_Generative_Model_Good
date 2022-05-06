import numpy as np
import torch
from torchvision.utils import save_image, make_grid
from tqdm import tqdm
from pathlib import Path
# from torch.utils.tensorboard import SummaryWriter
import torchvision

# device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train_autoencoder(model, train_loader, optimizer, mse, n_epochs, device):
    for epoch in range(n_epochs):
        loss = 0
        for batch_idx, (batch_features, target) in enumerate(train_loader):
            batch_features = batch_features.to(device)
            optimizer.zero_grad()
            outputs = model(batch_features)
            train_loss = mse(outputs, batch_features)
            train_loss.backward()
            optimizer.step()
            loss += train_loss.item()
        loss = loss/len(train_loader)
        if epoch %5 == 0:
            print("epochs : {}/{}, loss = {:.6f}".format(epoch+1, n_epochs, loss))

    # Validation
    with torch.no_grad():
        model.eval()
        torch.save(model.state_dict(), './model/model_autoencoder_ok.ckpt')
        print('The best model is detected')
    return model

#Variational autoencoder
def final_loss(bce_loss, mu, logvar):
    BCE = bce_loss
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

def save_reconstructed_images(recon_images, epoch, type):
    save_image(recon_images.cpu(),"./figures/"+type+"/" +str(epoch)+"_reconstructed.jpg")

def save_original_images(recon_images, epoch, type):
    save_image(recon_images.cpu(), "./figures/"+type+"/" +str(epoch)+"_original.jpg")

def train_vae(model, dataloader, dataset_size, device, optimizer, criterion):
    model.train()
    running_loss = 0.0
    counter = 0
    
    for i, data in tqdm(enumerate(dataloader), total=int(dataset_size/dataloader.batch_size)):  
        counter += 1
        data = data[0]
        data = data.to(device)
        optimizer.zero_grad()
        reconstruction, mu, logvar = model(data)

        bce_loss = criterion(reconstruction, data)
        loss = final_loss(bce_loss, mu, logvar)
        loss.backward()
        running_loss += loss.item()
        optimizer.step()

    train_loss = running_loss / counter
    return train_loss


def validate_vae(model, dataloader, dataset, device, criterion):
    model.eval()
    running_loss = 0.0
    counter = 0
    # print(dataloader.batch_size)
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=int(len(dataset)/dataloader.batch_size)):  
            counter += 1
            data = data[0]
            # print(data.shape)
            data = data.to(device)
            reconstruction, mu, logvar = model(data)
            bce_loss = criterion(reconstruction, data)
            loss = final_loss(bce_loss, mu, logvar)
            running_loss += loss.item()

            # save the last batch input ad output of every epoch
            if i == len(dataloader) -2 :
                recon_images = reconstruction
                original_images = data
    val_loss = running_loss / counter
    return val_loss, recon_images, original_images

def fully_train_vae(model, trainloader, testloader, train_dataset, test_dataset, optimizer, criterion, epochs, device):
    grid_images = []
    train_loss = []
    valid_loss = []
    Path("figures/vae/").mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} of {epochs}")
        train_epoch_loss = train_vae(
            model, trainloader, len(train_dataset), device, optimizer, criterion
        )
        valid_epoch_loss, recon_images, original_images = validate_vae(
            model, testloader, test_dataset, device, criterion
        )
        train_loss.append(train_epoch_loss)
        valid_loss.append(valid_epoch_loss)
        # print(recon_images.shape, 'recons')
        # print(original_images.shape, 'original')
        #save the reconstructed image from the validation loop
        save_reconstructed_images(recon_images, epoch+1, 'vae')
        save_original_images(original_images, epoch+1, 'vae')
        image_grid = make_grid(recon_images.detach().cpu())
        grid_images.append(image_grid)
        print(f"Train loss:{train_epoch_loss:.4f}")
        print(f"Train loss:{valid_epoch_loss:.4f}")
    
    with torch.no_grad():
        model.eval()
        torch.save(model.state_dict(), './model/model_vae_ok.ckpt')
        print('The best model is detected')

    return model

# WGAN
def train_wgan(gen, critic, opt_gen, opt_critic, loader, epochs, critic_iterations,Z_DIM, WEIGHT_CLIP, device):
    fixed_noise = torch.randn(32, Z_DIM, 1, 1).to(device)

    gen.train()
    critic.train()
    for epoch in range(epochs):
        # Target labels not needed! <3 unsupervised
        for batch_idx, (data, _) in enumerate(loader):
            data = data.to(device)
            cur_batch_size = data.shape[0]

            # Train Critic: max E[critic(real)] - E[critic(fake)]
            for _ in range(critic_iterations):
                noise = torch.randn(cur_batch_size, Z_DIM, 1, 1).to(device)
                fake = gen(noise)
                critic_real = critic(data).reshape(-1)
                critic_fake = critic(fake).reshape(-1)
                loss_critic = -(torch.mean(critic_real) - torch.mean(critic_fake))
                critic.zero_grad()
                loss_critic.backward(retain_graph=True)
                opt_critic.step()

                # clip critic weights between -0.01, 0.01
                for p in critic.parameters():
                    p.data.clamp_(-WEIGHT_CLIP, WEIGHT_CLIP)

            # Train Generator: max E[critic(gen_fake)] <-> min -E[critic(gen_fake)]
            gen_fake = critic(fake).reshape(-1)
            loss_gen = -torch.mean(gen_fake)
            gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()

            # Print losses occasionally and print to tensorboard
            if batch_idx % 100 == 0 and batch_idx > 0:
                gen.eval()
                critic.eval()
                print(
                    f"Epoch [{epoch}/{epochs}] Batch {batch_idx}/{len(loader)} \
                    Loss D: {loss_critic:.4f}, loss G: {loss_gen:.4f}"
                )

                with torch.no_grad():
                    fake = gen(noise)
                    # take out (up to) 32 examples
                    # img_grid_real = torchvision.utils.make_grid(
                    #     data[:32], normalize=True
                    # )
                    # img_grid_fake = torchvision.utils.make_grid(
                    #     fake[:32], normalize=True
                    # )

                    # writer_real.add_image("Real", img_grid_real, global_step=step)
                    # writer_fake.add_image("Fake", img_grid_fake, global_step=step)
                    
                    if batch_idx == (len(loader)//100)*100:
                        save_reconstructed_images(fake[:32], epoch+1, 'wgan')
                        save_original_images(data[:32], epoch+1, 'wgan')

                step += 1
                gen.train()
                critic.train()

    with torch.no_grad():
        gen.eval()
        torch.save(gen.state_dict(), './model/model_wgan_ok.ckpt')
        print('The best model is detected')
    