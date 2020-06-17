import torch
import numpy as np
from scipy.spatial.distance import cdist

def idx2onehot(idx, n):

    assert torch.max(idx).item() < n
    if idx.dim() == 1:
        idx = idx.unsqueeze(1)

    onehot = torch.zeros(idx.size(0), n)
    onehot.scatter_(1, idx, 1)

    return onehot

def add_noise(loader, clean_index, noise_level, noise_sigma, seed):
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    images = [sample_i for sample_i in loader.sampler.data_source.data]
    probs_to_change = torch.rand((len(images),))
    idx_to_change = probs_to_change >= (1 - noise_level)

    for n, image_i in enumerate(images):
        if idx_to_change[n] == 1 and n not in clean_index:
            noise = np.random.randn(*images[0].shape) * noise_sigma
            images[n] += noise.astype(np.uint8)

    loader.sampler.data_source.data = images
    return idx_to_change

    
def get_lid(loader, clean_index, vae, latent_size, seed):
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda")
    
    k = 20
    clean_latent_n = 5
    feature_n = 10
    
    images = [sample_i for sample_i in loader.sampler.data_source.data]
    clean_latent = np.zeros((len(clean_index)*clean_latent_n, latent_size))
    whole_latent = np.zeros((len(images)*feature_n, latent_size))
    lid_features = np.zeros((len(images), feature_n))
    mean_list = np.zeros((len(images), latent_size))
    std_list = np.zeros((len(images), latent_size))
    
    for n, index in enumerate(clean_index):
        image_i = images[index][np.newaxis, :]
        image_i = image_i.transpose(0,3,1,2)
        means, log_var = vae.encoder(torch.from_numpy(image_i).to(device).float())
        std = torch.exp(0.5 * log_var).detach().cpu().numpy()                
        eps = torch.randn([clean_latent_n, latent_size]).detach().cpu().numpy()
        z = eps * std + means.detach().cpu().numpy()
        clean_latent[n*clean_latent_n:(n+1)*clean_latent_n, :] = z
    
    f = lambda v: - k / np.sum(np.log(v/v[-1]))
    for n, image_i in enumerate(images):
        image_i = image_i[np.newaxis, :]
        image_i = image_i.transpose(0,3,1,2)
        means, log_var = vae.encoder(torch.from_numpy(image_i).to(device).float())
        std = torch.exp(0.5 * log_var).detach().cpu().numpy()        
        eps = torch.randn([feature_n, latent_size]).detach().cpu().numpy()
        z = eps * std + means.detach().cpu().numpy()
        whole_latent[n*feature_n:(n+1)*feature_n, :] = z
        mean_list[n, :] = means.detach().cpu().numpy()
        std_list[n, :] = std
    a = cdist(whole_latent, clean_latent, metric='euclidean')
    a = np.apply_along_axis(np.sort, axis=1, arr=a)[:,0:k]
    a = np.maximum(a, 1e-6)
    a = np.apply_along_axis(f, axis=1, arr=a)
    lid_features = a.reshape(len(images), feature_n)
    lid_features = np.hstack((lid_features, mean_list, std_list))
    return lid_features


