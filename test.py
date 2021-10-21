import os
import argparse
import torch

import numpy as np
import torch.nn.functional as F
import scipy.io as sio

from skimage.metrics import peak_signal_noise_ratio, structural_similarity

import utils
from model import UNetSeeInDark


def forward_patches(model, noisy, patch_size=256 * 3, pad=32):
    shift = patch_size - pad * 2

    noisy = torch.FloatTensor(noisy).cuda()
    noisy = utils.raw2stack(noisy).unsqueeze(0)
    noisy = F.pad(noisy, (pad, pad, pad, pad), mode='reflect')
    denoised = torch.zeros_like(noisy)

    _, _, H, W = noisy.shape
    for i in np.arange(0, H, shift):
        for j in np.arange(0, W, shift):
            h_end, w_end = min(i + patch_size, H), min(j + patch_size, W)
            h_start, w_start = h_end - patch_size, w_end - patch_size

            input_var = noisy[..., h_start: h_end, w_start: w_end]
            with torch.no_grad():
                out_var = model(input_var)
            denoised[..., h_start + pad: h_end - pad, w_start + pad: w_end - pad] = \
                out_var[..., pad:-pad, pad:-pad]

    denoised = denoised[..., pad:-pad, pad:-pad]
    denoised = utils.stack2raw(denoised[0]).detach().cpu().numpy()

    denoised = denoised.clip(0, 1)
    return denoised


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='/mnt/lustre/zhangyi3/data/SIDD_Medium/Data/')
    parser.add_argument('--camera', choices=['s6', 'gp', 'ip'], required=True, help='camera name')
    args = parser.parse_args()

    camera = args.camera
    root = args.root

    # save_dir = './results/' + camera
    # if not os.path.exists(save_dir):
    #         os.makedirs(save_dir)
    print('test', camera, 'root', root)

    test_data_list = [item for item in os.listdir(root) if int(item.split('_')[1]) in [2, 3, 5] and camera in item.lower()]

    # build model
    model = UNetSeeInDark()
    model = model.cuda()
    model = torch.nn.DataParallel(model)

    model_path = './checkpoints/%s.pth' % camera.lower()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))

    psnr_list = []
    for idx, item in enumerate(test_data_list):
        head = item[:4]
        for tail in ['GT_RAW_010', 'GT_RAW_011']:
            print('processing', idx, item, tail, end=' ')
            mat = utils.open_hdf5(os.path.join(root, item, '%s_%s.MAT' % (head, tail)))
            gt = np.array(mat['x'], dtype=np.float32)
            mat = utils.open_hdf5(os.path.join(root, item, '%s_%s.MAT' % (head, tail.replace('GT', 'NOISY'))))
            noisy = np.array(mat['x'], dtype=np.float32)

            meta = sio.loadmat(os.path.join(root, item, '%s_%s.MAT' % (head, tail.replace('GT', 'METADATA'))))
            meta = meta['metadata'][0][0]

            # transform to rggb pattern
            py_meta = utils.extract_metainfo(
                os.path.join(root, item, '%s_%s.MAT' % (head, tail.replace('GT', 'METADATA'))))
            pattern = py_meta['pattern']
            noisy = utils.transform_to_rggb(noisy, pattern)
            gt = utils.transform_to_rggb(gt, pattern)

            denoised = forward_patches(model, noisy)

            psnr = peak_signal_noise_ratio(gt, denoised, data_range=1)
            psnr_list.append(psnr)
            print('psnr %.2f' % psnr)

    print('Camera %s, average PSNR %.2f' % (camera, np.mean(psnr_list)))
