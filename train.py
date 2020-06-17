import os
import time
import torch
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from collections import defaultdict
from utils import add_noise, get_lid
from models import VAE
import numpy as np


def main(args):

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = torch.device("cuda")
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

    ts = time.time()
    if args.dataset == "MNIST":
        dataset = MNIST(
            root='data', train=True, transform=transforms.ToTensor(),
            download=True)
    data_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True)
    clean_index = np.random.choice(len(dataset), int(len(dataset)*args.clean_ratio), replace=False)
    noise_index = add_noise(data_loader, clean_index, args.noise_level, args.noise_sigma, args.clean_ratio, args.seed)
    data_track_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=False)
    data_track_loader.sampler.data_source.train_data = data_loader.sampler.data_source.train_data
        
    def loss_fn(recon_x, x, mean, log_var):
        BCE = torch.nn.functional.binary_cross_entropy(
            recon_x.view(-1, 28*28), x.view(-1, 28*28), reduction='sum')
        KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

        return (BCE + KLD) / x.size(0)

    vae = VAE(
        encoder_layer_sizes=args.encoder_layer_sizes,
        latent_size=args.latent_size,
        decoder_layer_sizes=args.decoder_layer_sizes,
        conditional=args.conditional,
        num_labels=10 if args.conditional else 0).to(device)

    optimizer = torch.optim.Adam(vae.parameters(), lr=args.learning_rate)

    logs = defaultdict(list)

    for epoch in range(args.epochs):

        tracker_epoch = defaultdict(lambda: defaultdict(dict))

        for iteration, (x, y) in enumerate(data_loader):

            x, y = x.to(device), y.to(device)

            if args.conditional:
                recon_x, mean, log_var, z = vae(x, y)
            else:
                recon_x, mean, log_var, z = vae(x)

            for i, yi in enumerate(y):
                id = len(tracker_epoch)
                tracker_epoch[id]['x'] = z[i, 0].item()
                tracker_epoch[id]['y'] = z[i, 1].item()
                tracker_epoch[id]['label'] = yi.item()

            loss = loss_fn(recon_x, x, mean, log_var)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            logs['loss'].append(loss.item())

#             if iteration % args.print_every == 0 or iteration == len(data_loader)-1:
#                 print("Epoch {:02d}/{:02d} Batch {:04d}/{:d}, Loss {:9.4f}".format(
#                     epoch, args.epochs, iteration, len(data_loader)-1, loss.item()))

#                 if args.conditional:
#                     c = torch.arange(0, 10).long().unsqueeze(1)
#                     x = vae.inference(n=c.size(0), c=c)
#                 else:
#                     x = vae.inference(n=10)

#                 plt.figure()
#                 plt.figure(figsize=(5, 10))
#                 for p in range(10):
#                     plt.subplot(5, 2, p+1)
#                     if args.conditional:
#                         plt.text(
#                             0, 0, "c={:d}".format(c[p].item()), color='black',
#                             backgroundcolor='white', fontsize=8)
#                     plt.imshow(x[p].view(28, 28).data.cpu().numpy())
#                     plt.axis('off')
                
#                 fig_path = os.path.join(args.fig_root, 'noise_%.2f_sigma_%.1f_n_%d'%(args.noise_level, args.noise_sigma, args.latent_size))
#                 if not os.path.exists(fig_path):
#                     os.makedirs(fig_path)

#                 plt.savefig(
#                     os.path.join(fig_path,
#                                  "E{:d}I{:d}.png".format(epoch, iteration)),
#                     dpi=300)
#                 plt.clf()
#                 plt.close('all')

#         df = pd.DataFrame.from_dict(tracker_epoch, orient='index')
#         g = sns.lmplot(
#             x='x', y='y', hue='label', data=df.groupby('label').head(200),
#             fit_reg=False, legend=True)
#         g.savefig(os.path.join(
#             fig_path, "E{:d}-Dist.png".format(epoch)),
#             dpi=300)
        
        lid_features = get_lid(data_track_loader, clean_index, vae, args.latent_size, args.seed)
        df = pd.DataFrame(lid_features)
        df.columns = [str(i) for i in range(10)] + ['mean %d'%i for i in range(args.latent_size)] + ['std %d'%i for i in range(args.latent_size)]
        df['clean'] = [i in clean_index for i in range(len(lid_features))]
        df['add_noise'] = list(noise_index.detach().cpu().numpy())
        file_path = 'record/%s/noise_%.2f_sigma_%.1f_n_%d'%(args.dataset, args.noise_level, args.noise_sigma, args.latent_size)
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        df.to_csv(file_path+'/epoch%d.csv'%epoch, index=True)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)    
    parser.add_argument("--dataset", type=str, default='MNIST')
    parser.add_argument("--clean_ratio", type=float, default=0.05)
    parser.add_argument("--noise_level", type=float, default=0.5)
    parser.add_argument("--noise_sigma", type=float, default=10)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--encoder_layer_sizes", type=list, default=[784, 400, 200, 100])
    parser.add_argument("--decoder_layer_sizes", type=list, default=[100, 200, 400, 784])
    parser.add_argument("--latent_size", type=int, default=2)
    parser.add_argument("--print_every", type=int, default=100)
    parser.add_argument("--fig_root", type=str, default='figs')
    parser.add_argument("--conditional", action='store_true')

    args = parser.parse_args()
    
    for noise in [0.1, 0.3, 0.5, 0.7, 0.9]:
        for sigma in [5, 10, 20, 50]:
            for n in [10]:  
                args.noise_level = noise
                args.noise_sigma = sigma
                args.latent_size = n
                main(args)
