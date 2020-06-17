import torch
import torch.nn as nn

from utils import idx2onehot


class VAE(nn.Module):

    def __init__(self, encoder_layer_sizes, latent_size, decoder_layer_sizes,
                 conditional=False, num_labels=0):

        super().__init__()

        if conditional:
            assert num_labels > 0

        assert type(encoder_layer_sizes) == list
        assert type(latent_size) == int
        assert type(decoder_layer_sizes) == list

        self.latent_size = latent_size

        self.encoder = Encoder(
            encoder_layer_sizes, latent_size, conditional, num_labels)
        self.decoder = Decoder(
            decoder_layer_sizes, latent_size, conditional, num_labels)
        self.device = torch.device("cuda")

    def forward(self, x, c=None):

        if x.dim() > 2:
            x = x.view(-1, 28 * 28)

        batch_size = x.size(0)

        means, log_var = self.encoder(x, c)

        std = torch.exp(0.5 * log_var).to(self.device)
        eps = torch.randn([batch_size, self.latent_size]).to(self.device)
        z = eps * std + means

        recon_x = self.decoder(z, c)

        return recon_x, means, log_var, z

    def inference(self, n=1, c=None):

        batch_size = n
        z = torch.randn([batch_size, self.latent_size]).to(self.device)

        recon_x = self.decoder(z, c)

        return recon_x


class Encoder(nn.Module):

    def __init__(self, layer_sizes, latent_size, conditional, num_labels):

        super().__init__()

        self.conditional = conditional
        if self.conditional:
            layer_sizes[0] += num_labels

        self.MLP = nn.Sequential()

        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())

        self.linear_means = nn.Linear(layer_sizes[-1], latent_size)
        self.linear_log_var = nn.Linear(layer_sizes[-1], latent_size)

    def forward(self, x, c=None):

        if self.conditional:
            c = idx2onehot(c, n=10)
            x = torch.cat((x, c), dim=-1)

        x = self.MLP(x)

        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)

        return means, log_vars


class Decoder(nn.Module):

    def __init__(self, layer_sizes, latent_size, conditional, num_labels):

        super().__init__()

        self.MLP = nn.Sequential()

        self.conditional = conditional
        if self.conditional:
            input_size = latent_size + num_labels
        else:
            input_size = latent_size

        for i, (in_size, out_size) in enumerate(zip([input_size] + layer_sizes[:-1], layer_sizes)):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            if i + 1 < len(layer_sizes):
                self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())
            else:
                self.MLP.add_module(name="sigmoid", module=nn.Sigmoid())

    def forward(self, z, c):

        if self.conditional:
            c = idx2onehot(c, n=10)
            z = torch.cat((z, c), dim=-1)

        x = self.MLP(z)

        return x


class VAE_CNN(nn.Module):

    def __init__(self, encoder_filter_sizes, latent_size, decoder_filter_sizes,
                 conditional=False, num_labels=0, in_channels=3):

        super().__init__()

        if conditional:
            assert num_labels > 0

        assert type(encoder_filter_sizes) == list
        assert type(latent_size) == int
        assert type(decoder_filter_sizes) == list

        self.latent_size = latent_size

        self.encoder = Encoder_CNN(
            encoder_filter_sizes, latent_size, conditional, num_labels, in_channels)
        self.decoder = Decoder_CNN(
            decoder_filter_sizes, latent_size, conditional, num_labels, in_channels)
        self.device = torch.device("cuda")

    def forward(self, x, c=None):

        # if x.dim() > 2:
        #     x = x.view(-1, 28 * 28)

        batch_size = x.size(0)

        means, log_var = self.encoder(x, c)

        std = torch.exp(0.5 * log_var).to(self.device)
        eps = torch.randn([batch_size, self.latent_size]).to(self.device)
        z = eps * std + means

        recon_x = self.decoder(z, c)

        return recon_x, means, log_var, z

    def inference(self, n=1, c=None):

        batch_size = n
        z = torch.randn([batch_size, self.latent_size]).to(self.device)

        recon_x = self.decoder(z, c)

        return recon_x


class Encoder_CNN(nn.Module):

    def __init__(self, filter_sizes, latent_size, conditional, num_labels, in_channels):

        super().__init__()

        self.conditional = conditional
        if self.conditional:
            layer_sizes[0] += num_labels

        self.MLP = nn.Sequential()

        for i, f_size in enumerate(filter_sizes):
            self.MLP.add_module(
                name="C{:d}".format(i),
                module=nn.Conv2d(in_channels, out_channels=f_size, kernel_size=3, stride=2, padding=1))
            self.MLP.add_module(name="N{:d}".format(i), module=nn.BatchNorm2d(f_size))
            self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())
            in_channels = f_size

        self.linear_means = nn.Linear(filter_sizes[-1] * 4, latent_size)
        self.linear_log_var = nn.Linear(filter_sizes[-1] * 4, latent_size)

    def forward(self, x, c=None):

        if self.conditional:
            c = idx2onehot(c, n=10)
            x = torch.cat((x, c), dim=-1)

        x = self.MLP(x)
        x = torch.flatten(x, start_dim=1)
        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)

        return means, log_vars


class Decoder_CNN(nn.Module):

    def __init__(self, filter_sizes, latent_size, conditional, num_labels, out_channels=3):

        super().__init__()
        self.filter_sizes = filter_sizes
        self.MLP = nn.Sequential()

        self.conditional = conditional
        if self.conditional:
            input_size = latent_size + num_labels
        else:
            input_size = latent_size

        self.decoder_input = nn.Linear(input_size, filter_sizes[0] * 4)

        for i, (in_size, out_size) in enumerate(zip(filter_sizes, filter_sizes[1:] + [filter_sizes[-1]])):
            self.MLP.add_module(
                name="CT{:d}".format(i), module=nn.ConvTranspose2d(in_size, out_size, kernel_size=3, stride=2,
                padding=1, output_padding=1))
            self.MLP.add_module(name="N{:d}".format(i), module=nn.BatchNorm2d(out_size))
            self.MLP.add_module(name="A{:d}".format(i), module=nn.LeakyReLU())

        self.MLP.add_module(
            name="C{:d}".format(0), module=nn.Conv2d(out_size, out_channels=out_channels, kernel_size=3, padding=1))
        self.MLP.add_module(name="A{:d}".format(i), module=nn.LeakyReLU())

    def forward(self, z, c):

        if self.conditional:
            c = idx2onehot(c, n=10)
            z = torch.cat((z, c), dim=-1)

        x = self.decoder_input(z)
        x = x.view(-1, self.filter_sizes[0], 2, 2)
        x = self.MLP(x)

        return x
