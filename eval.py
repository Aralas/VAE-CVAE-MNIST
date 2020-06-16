import os
import time
import torch
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from collections import defaultdict
from utils import add_noise
from models import VAE


def eval_VAE(dataset, noise_level, noise_sigma, latent_size):
    path = 'record/%s/noise_%.2f_sigma_%.1f_n_%d' % (dataset, noise_level, noise_sigma, latent_size)
    fig_path = 'figs/%s/noise_%.2f_sigma_%.1f_n_%d' % (dataset, noise_level, noise_sigma, latent_size)
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
    mean_col = ['mean %d' % i for i in range(latent_size)]
    std_col = ['std %d' % i for i in range(latent_size)]
    for epoch in range(10):
        data = pd.read_csv(path + '/epoch%d.csv' % epoch)
        data['label'] = None
        data.loc[(data['add_noise'] == 1), 'label'] = 'with noise'
        data.loc[(data['add_noise'] == 0), 'label'] = 'without noise'
        data.loc[(data['clean'] == True), 'label'] = 'clean'
        plt.figure(figsize=(12, 6))
        plt.subplots_adjust(wspace=0.3, hspace=0.2)
        plt.subplot(121)
        sns.scatterplot(x="mean 0", y="mean 1", hue="label", hue_order=["clean", "with noise", "without noise"],
                        style="label", data=data.groupby('label').head(200), legend="brief")
        plt.title("Scatter plot of mean vectors", fontsize=16)

        plt.subplot(122)
        sns.scatterplot(x="std 0", y="std 1", hue="label", hue_order=["clean", "with noise", "without noise"],
                        style="label", data=data.groupby('label').head(200), legend="brief")
        plt.title("Scatter plot of std vectors", fontsize=16)

        plt.suptitle("After %d epochs" % (epoch + 1), fontsize=20)
        plt.savefig(fig_path + "/VAE_eval_epoch%d.png" % epoch)


def eval_LID(dataset, noise_level, noise_sigma, latent_size, feature_n=10, upper_bound=200):
    path = 'record/%s/noise_%.2f_sigma_%.1f_n_%d' % (dataset, noise_level, noise_sigma, latent_size)
    fig_path = 'figs/%s/noise_%.2f_sigma_%.1f_n_%d' % (dataset, noise_level, noise_sigma, latent_size)
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
    mean_col = ['mean %d' % i for i in range(latent_size)]
    std_col = ['std %d' % i for i in range(latent_size)]

    clean_process = []
    noise_process = []
    add_noise_process = []
    no_noise_process = []
    for epoch in range(10):
        data = pd.read_csv(path + '/epoch%d.csv' % epoch)
        clean_lid = data[(data['clean'] == True)][[str(i) for i in range(feature_n)]].dropna().to_numpy()
        noise_lid = data[(data['clean'] == False)][[str(i) for i in range(feature_n)]].dropna().to_numpy()
        add_noise_lid = data[(data['add_noise'] == 1) & (data['clean'] == False)][
            [str(i) for i in range(feature_n)]].dropna().to_numpy()
        no_noise_lid = data[((data['add_noise'] == 1) & (data['clean'] == True)) | (data['add_noise'] == 0)][
            [str(i) for i in range(feature_n)]].dropna().to_numpy()

        clean_lid_mean = np.clip(np.mean(clean_lid, axis=1), 0, upper_bound)
        noise_lid_mean = np.clip(np.mean(noise_lid, axis=1), 0, upper_bound)
        add_noise_lid_mean = np.clip(np.mean(add_noise_lid, axis=1), 0, upper_bound)
        no_noise_lid_mean = np.clip(np.mean(no_noise_lid, axis=1), 0, upper_bound)

        clean_lid_variance = np.clip(np.var(clean_lid, axis=1), 0, upper_bound)
        noise_lid_variance = np.clip(np.var(noise_lid, axis=1), 0, upper_bound)
        add_noise_lid_variance = np.clip(np.var(add_noise_lid, axis=1), 0, upper_bound)
        no_noise_lid_variance = np.clip(np.var(no_noise_lid, axis=1), 0, upper_bound)

        plt.figure(figsize=(12, 8))
        plt.subplots_adjust(wspace=0.3, hspace=0.25)
        plt.subplot(221)
        plt.hist(clean_lid_mean, bins=100, density=False, alpha=0.7, rwidth=0.85)
        plt.title('Small clean set')
        plt.subplot(222)
        plt.hist(noise_lid_mean, bins=100, density=False, alpha=0.7, rwidth=0.85)
        plt.title('Complementary set')
        plt.subplot(223)
        plt.hist(no_noise_lid_mean, bins=100, alpha=0.7, rwidth=0.85)
        plt.title('Without noise, includeing clean set')
        plt.subplot(224)
        plt.hist(add_noise_lid_mean, bins=100, alpha=0.7, rwidth=0.85)
        plt.title('With noise')
        plt.suptitle('epoch %d' % epoch, fontsize=20)
        plt.savefig(fig_path + "/LID_mean_epoch%d.png" % epoch)

        plt.figure(figsize=(12, 8))
        plt.subplots_adjust(wspace=0.3, hspace=0.25)
        plt.subplot(221)
        plt.hist(clean_lid_variance, bins=100, alpha=0.7, rwidth=0.85)
        plt.title('Small clean set')
        plt.subplot(222)
        plt.hist(noise_lid_variance, bins=100, alpha=0.7, rwidth=0.85)
        plt.title('Complementary set')
        plt.subplot(223)
        plt.hist(no_noise_lid_variance, bins=100, alpha=0.7, rwidth=0.85)
        plt.title('Without noise, includeing clean set')
        plt.subplot(224)
        plt.hist(add_noise_lid_variance, bins=100, alpha=0.7, rwidth=0.85)
        plt.title('With noise')
        plt.suptitle('epoch %d' % epoch, fontsize=20)
        plt.savefig(fig_path + "/LID_var_epoch%d.png" % epoch)



for noise_level in [0.1, 0.3, 0.5, 0.7, 0.9]:
    for noise_sigma in [5, 10, 20, 50]:
        for latent_size in [2]:
            dataset = "MNIST"
            try:
                eval_VAE(dataset, noise_level, noise_sigma, latent_size)
            except:
                pass

for noise_level in [0.1, 0.3, 0.5, 0.7, 0.9]:
    for noise_sigma in [5, 10, 20, 50]:
        for latent_size in [2, 5, 10]:
            dataset = "MNIST"
            try:
                eval_LID(dataset, noise_level, noise_sigma, latent_size)
            except:
                pass
