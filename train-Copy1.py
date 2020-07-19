import os
import time
import torch
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10
from torch.utils.data import DataLoader
from collections import defaultdict
from utils import add_noise, get_lid
from models import VAE, VAE_CNN
import numpy as np
from PIL import Image


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    datapath = os.path.abspath('..') + 'data/LID'
    if args.dataset == "MNIST":
        dataset = MNIST(
            root=datapath+'/data', train=True, transform=transforms.ToTensor(),
            download=True)
        vae = VAE(
            encoder_layer_sizes=args.encoder_layer_sizes,
            latent_size=args.latent_size,
            decoder_layer_sizes=args.encoder_layer_sizes.reverse(),
            conditional=args.conditional,
            num_labels=10 if args.conditional else 0).to(device)
    elif args.dataset == "CIFAR10":
        SetRange = transforms.Lambda(lambda X: 2 * X - 1.)
        transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                            transforms.CenterCrop(148),
                            transforms.ToTensor()])
        train_set = CIFAR10(
            root=datapath+'/data', train=True, transform=transforms.ToTensor(),
            download=True)
        test_set = CIFAR10(
            root=datapath+'/data', train=False, transform=transforms.ToTensor(),
            download=True)
        vae = VAE_CNN(
            encoder_filter_sizes=args.encoder_filter_sizes,
            latent_size=args.latent_size,
            decoder_filter_sizes=args.encoder_filter_sizes[::-1],
            conditional=args.conditional,
            num_labels=10 if args.conditional else 0).to(device)

    train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True)
    train_loader_orig = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=False)
    train_images = [sample_i.transpose((2,0,1)) for sample_i in train_loader_orig.sampler.data_source.data]
    test_images = [sample_i.transpose((2,0,1)) for sample_i in test_loader.sampler.data_source.data]
    
    clean_index = np.random.choice(len(train_set), int(len(train_set) * args.clean_ratio), replace=False)
    train_clean_images = np.array(train_images)[clean_index]
    train_clean_images = torch.from_numpy(train_clean_images).float().to(device)
    noise_index = add_noise(train_loader, clean_index, args.noise_level, args.noise_sigma, args.seed)
    clean_index_test = np.random.choice(len(test_set), int(len(test_set) * args.clean_ratio), replace=False)
    noise_index_test = add_noise(test_loader, clean_index, args.noise_level, args.noise_sigma, args.seed)
    
#     data_track_loader.sampler.data_source.data = data_loader.sampler.data_source.data

    def loss_fn(recon_x, x, mean, log_var, mean_clean, log_var_clean):
        BCE = torch.nn.functional.mse_loss(recon_x, x, reduction='sum')        
        KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        
        dist_mean = torch.cdist(mean, mean_clean, 2)
        dist_log_var = torch.cdist(log_var, log_var_clean, 2)
        
        kvals_mean, _ = dist_mean.topk(k=20, dim=1)
        kvals_log_var, _ = dist_log_var.topk(k=20, dim=1)
               
        LID_mean = torch.sum(torch.log(kvals_mean)-torch.log(kvals_mean)[:, 0].reshape(-1, 1))/20
        LID_log_var = torch.sum(torch.log(kvals_log_var)-torch.log(kvals_log_var)[:, 0].reshape(-1, 1))/20

        return (BCE + KLD + LID_mean + LID_log_var) / x.size(0)

    optimizer = torch.optim.Adam(vae.parameters(), lr=args.learning_rate)

    logs = defaultdict(list)
    
    f = open(datapath + "/result/%s/noise_%.2f_sigma_%d_n_%d.txt" % (args.dataset, args.noise_level, args.noise_sigma, args.latent_size), "a+")
    
    for epoch in range(args.epochs):
        vae.train()
        for iteration, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)            
            if args.conditional:
                recon_x, mean, log_var, z = vae(x, y)
            else:
                recon_x, mean, log_var, z = vae(x)
            mean_clean, log_var_clean = vae.encoder(train_clean_images)
            
            loss = loss_fn(recon_x, x, mean, log_var, mean_clean, log_var_clean)
            print("training process: epoch %d, iteration %d, loss %.4f" % (epoch, iteration, loss))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            logs['loss'].append(loss.item())
        
        vae.eval()
        mse = 0
        psnr = 0
        total = 0

        with torch.no_grad():
            for iteration, (x, y) in enumerate(test_loader):
                x, y = x.to(device), y.to(device)
                output = vae(x)
                output = np.array(output[0][0].cpu())*255
                mse_i = (np.abs(output - test_images[iteration]) ** 2).mean()
                psnr_i = 10 * np.log10(255 * 255 / mse_i)
                mse += mse_i
                psnr += psnr_i
                total += 1
        f.write('epoch: %d,  mse: %.3f, psnr: %.3f \n' % (epoch, mse, psnr))
        f.flush()                
        print('evaluation process: epoch: %d, mse: %.3f, psnr: %.3f' % (epoch, mse, psnr))
              
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dataset", type=str, default='MNIST')
    parser.add_argument("--clean_ratio", type=float, default=0.05)
    parser.add_argument("--noise_level", type=float, default=0.5)
    parser.add_argument("--noise_sigma", type=float, default=10)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--encoder_layer_sizes", type=list, default=[784, 400, 200, 100])
    parser.add_argument("--encoder_filter_sizes", type=list, default=[32, 64, 128, 256])
    parser.add_argument("--latent_size", type=int, default=2)
    parser.add_argument("--print_every", type=int, default=100)
    parser.add_argument("--fig_root", type=str, default='figs')
    parser.add_argument("--conditional", action='store_true')

    args = parser.parse_args()
    
    device = torch.device("cuda")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    for noise in [0.5]:
        for sigma in [10]:
            for n in [50]:
                args.noise_level = noise
                args.noise_sigma = sigma
                args.latent_size = n
                args.dataset = "CIFAR10"
                args.epochs = 10
                main(args)
