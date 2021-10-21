import h5py
import time
import torch

import scipy.io as sio

import numpy as np


def open_hdf5(filename):
    while True:
        try:
            hdf5_file = h5py.File(filename, 'r')
            return hdf5_file
        except OSError:
            print(filename, ' waiting')
            time.sleep(3)  # Wait a bit


def extract_metainfo(path='0151_METADATA_RAW_010.MAT'):
    meta = sio.loadmat(path)['metadata']
    mat_vals = meta[0][0]
    mat_keys = mat_vals.dtype.descr

    keys = []
    for item in mat_keys:
        keys.append(item[0])

    py_dict = {}
    for key in keys:
        py_dict[key] = mat_vals[key]

    device = py_dict['Model'][0].lower()
    bitDepth = py_dict['BitDepth'][0][0]
    if 'iphone' in device or bitDepth != 16:
        noise = py_dict['UnknownTags'][-2][0][-1][0][:2]
        iso = py_dict['DigitalCamera'][0, 0]['ISOSpeedRatings'][0][0]
        pattern = py_dict['SubIFDs'][0][0]['UnknownTags'][0][0][1][0][-1][0]
        time = py_dict['DigitalCamera'][0, 0]['ExposureTime'][0][0]

    else:
        noise = py_dict['UnknownTags'][-1][0][-1][0][:2]
        iso = py_dict['ISOSpeedRatings'][0][0]
        pattern = py_dict['UnknownTags'][1][0][-1][0]
        time = py_dict['ExposureTime'][0][0]  # the 0th row and 0th line item

    rgb = ['R', 'G', 'B']
    pattern = ''.join([rgb[i] for i in pattern])

    asShotNeutral = py_dict['AsShotNeutral'][0]
    b_gain, _, r_gain = asShotNeutral

    # only load ccm1
    ccm = py_dict['ColorMatrix1'][0].astype(float).reshape((3, 3))

    return {'device': device,
            'pattern': pattern,
            'iso': iso,
            'noise': noise,
            'time': time,
            'wb': np.array([r_gain, 1, b_gain]),
            'ccm': ccm, }


def transform_to_rggb(img, pattern):
    assert len(img.shape) == 2 and type(img) == np.ndarray

    if pattern.lower() == 'bggr':  # same pattern
        img = np.roll(np.roll(img, 1, axis=1), 1, axis=0)
    elif pattern.lower() == 'rggb':
        pass
    elif pattern.lower() == 'grbg':
        img = np.roll(img, 1, axis=1)
    elif pattern.lower() == 'gbrg':
        img = np.roll(img, 1, axis=0)
    else:
        assert 'no support'

    return img


def raw2stack(var):
    h, w = var.shape
    if var.is_cuda:
        res = torch.cuda.FloatTensor(4, h // 2, w // 2).fill_(0)
    else:
        res = torch.FloatTensor(4, h // 2, w // 2).fill_(0)
    res[0] = var[0::2, 0::2]
    res[1] = var[0::2, 1::2]
    res[2] = var[1::2, 0::2]
    res[3] = var[1::2, 1::2]
    return res


def stack2raw(var):
    _, h, w = var.shape
    if var.is_cuda:
        res = torch.cuda.FloatTensor(h * 2, w * 2)
    else:
        res = torch.FloatTensor(h * 2, w * 2)
    res[0::2, 0::2] = var[0]
    res[0::2, 1::2] = var[1]
    res[1::2, 0::2] = var[2]
    res[1::2, 1::2] = var[3]
    return res
