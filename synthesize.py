import torch
from torch.distributions.poisson import Poisson

import numpy as np


def generate_poisson_(y, k=1):
    y = torch.poisson(y / k) * k
    return y


def generate_read_noise(shape, noise_type, scale, loc=0):
    noise_type = noise_type.lower()
    if noise_type == 'norm':
        read = torch.FloatTensor(shape).normal_(loc, scale)
    else:
        raise NotImplementedError('Read noise type error.')
    return read


def noise_profiles(camera):
    camera = camera.lower()
    if camera == 'ip':  # iPhone
        iso_set = [100, 200, 400, 800, 1600, 2000]
        cshot = [0.00093595, 0.00104404, 0.00116461, 0.00129911, 0.00144915, 0.00150104]
        cread = [4.697713410870357e-07, 6.904488905478659e-07, 6.739473744228789e-07,
                 6.776787431555864e-07, 6.781983208034481e-07, 6.783184262356993e-07]
    elif camera == 's6':  # Sumsung s6 edge
        iso_set = [100, 200, 400, 800, 1600, 3200]
        cshot = [0.00162521, 0.00256175, 0.00403799, 0.00636492, 0.01003277, 0.01581424]
        cread = [1.1792188420255036e-06, 1.607602896683437e-06, 2.9872611575167216e-06,
                 5.19157563906707e-06, 1.0011034196248119e-05, 2.0652668477786836e-05]
    elif camera == 'gp':  # Google Pixel
        iso_set = [100, 200, 400, 800, 1600, 3200, 6400]
        cshot = [0.00024718, 0.00048489, 0.00095121, 0.001866, 0.00366055, 0.00718092, 0.01408686]
        cread = [1.6819349659429324e-06, 2.0556981890860545e-06, 2.703070976302046e-06,
                 4.116405515789963e-06, 7.569256436438246e-06, 1.5199001098203388e-05, 5.331422827048082e-05]
    elif camera == 'sony':  # Sony a7s2
        iso_set = [800, 1600, 3200]
        cshot = [1.0028880020069384, 1.804521362114003, 3.246920234173119]
        cread = [4.053034401667052, 6.692229120425673, 4.283115294604881]
    elif camera == 'nikon':  # Nikon D850
        iso_set = [800, 1600, 3200]
        cshot = [3.355988883536526, 6.688199969242411, 13.32901281288985]
        cread = [4.4959735547955635, 8.360429952584846, 15.684213053647735]
    else:
        assert NotImplementedError
    return iso_set, cshot, cread


def pg_noise_demo(clean_tensor, camera='IP'):
    iso_set, k_set, read_scale_set = noise_profiles(camera)

    # sample randomly
    i = np.random.choice(len(k_set))
    k, read_scale = k_set[i], read_scale_set[i]

    noisy_shot = generate_poisson_(clean_tensor, k)
    read_noise = generate_read_noise(clean_tensor.shape, noise_type='norm', scale=read_scale)
    noisy = noisy_shot + read_noise
    return noisy


if __name__ == '__main__':
    clean = torch.randn(48, 48).clamp(0, 1)
    noisy = pg_noise_demo(clean, camera='ip')
    print(noisy.shape, noisy.mean())
