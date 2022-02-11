
# @Author: Enea Duka
# @Date: 5/18/21
import torch
import random
# import kornia as K
import torchvision
from torchvision import transforms
import torchvision.transforms.functional as F
import numpy as np

def crop(vid, i, j, h, w):
    return vid[..., i:(i + h), j:(j + w)]

def _get_centers(tensor, n_frames):
    w = tensor.shape[2]
    h = tensor.shape[3]
    return torch.FloatTensor([w // 2, h // 2]).repeat(n_frames, 1)

# def rotate(tensor, rot_angle, p=1.0):
#     if random.random() < p:
#         tensor = tensor.permute(1, 0, 2, 3)
#         f, c, w, h = tensor.shape
#         # tensor = tensor.reshape(f*c, w, h)
#         rnd = torch.rand((1, ))
#         rot_angle_tensor = ((rnd * rot_angle * 2) - rnd * rot_angle).repeat(f, 1).squeeze()
#         center_tensor = _get_centers(tensor, f)
#         tensor = K.geometry.transform.rotate(tensor, rot_angle_tensor, center_tensor)
#         tensor = tensor.permute(1, 0, 2, 3)
#     return tensor


def center_crop(vid, output_size):
    h, w = vid.shape[-2:]
    th, tw = output_size

    i = int(round((h - th) / 2.))
    j = int(round((w - tw) / 2.))
    return crop(vid, i, j, th, tw)


def hflip(vid):
    return vid.flip(dims=(-1,))

def _get_output_dim(h, w, mode, base_size):
    if mode == 'train':
        size = np.random.randint(base_size + 2, max(h, w, base_size))
    else:
        size = base_size
    if h >= w:
        return int(h * size / w), size
    else:
        return size, int(w * size / h)

# NOTE: for those functions, which generally expect mini-batches, we keep them
# as non-minibatch so that they are applied as if they were 4d (thus image).
# this way, we only apply the transformation in the spatial domain
# def resize(vid, size, mode, interpolation='bilinear'):
#     # NOTE: using bilinear interpolation because we don't work on minibatches
#     # at this level
#     w, h = _get_output_dim(vid.shape[2], vid.shape[3], mode, size)
#     if w < 112 or h < 112:
#         print('here')
#     size = (w, h)
#     scale = None
#     if isinstance(size, int):
#         scale = float(size) / min(vid.shape[-2:])
#         size = None
#         vid = torch.nn.functional.interpolate(
#         vid, size=size, scale_factor=scale, mode=interpolation, align_corners=False)
#     return vid
def resize(vid, size, interpolation='bilinear'):
    # NOTE: using bilinear interpolation because we don't work on minibatches
    # at this level
    scale = None
    if isinstance(size, int):
        scale = float(size) / min(vid.shape[-2:])
        size = None
    return torch.nn.functional.interpolate(
        vid, size=size, scale_factor=scale, mode=interpolation, align_corners=False)

def pad(vid, padding, fill=0, padding_mode="constant"):
    # NOTE: don't want to pad on temporal dimension, so let as non-batch
    # (4d) before padding. This works as expected
    return torch.nn.functional.pad(vid, padding, value=fill, mode=padding_mode)


def to_normalized_float_tensor(vid):
    return vid.permute(1, 0, 2, 3).to(torch.float32) / 255


def normalize(vid, mean, std):
    # vid = vid.permute(1, 0, 2, 3)
    shape = (-1,) + (1,) * (vid.dim() - 1)
    mean = torch.as_tensor(mean).reshape(shape)
    std = torch.as_tensor(std).reshape(shape)
    vid = (vid - mean) / std
    return vid

def unnormalize(vid, mean, std):
    shape = (-1,) + (1,) * (vid.dim() - 1)
    mean = torch.as_tensor(mean).reshape(shape)
    std = torch.as_tensor(std).reshape(shape)
    return (vid * std) + mean


def random_gray(vid, p):
    if random.random() < p:
        # first select a channel
        channel = random.randint(0, 2)
        vid = vid[:, :, :, channel].unsqueeze(-1)
        vid = vid.repeat(1, 1, 1, 3)
    return vid



# Class interface

class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    @staticmethod
    def get_params(vid, output_size):
        """Get parameters for ``crop`` for a random crop.
        """
        h, w = vid.shape[-2:]
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, vid):
        i, j, h, w = self.get_params(vid, self.size)
        return crop(vid, i, j, h, w)


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, vid):
        return center_crop(vid, self.size)


class Resize(object):
    def __init__(self, size):
        self.size = size
        # self.mode = mode
    def __call__(self, vid):
        return resize(vid, self.size)


class ToFloatTensorInZeroOne(object):
    def __call__(self, vid):
        return to_normalized_float_tensor(vid)


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, vid):
        return normalize(vid, self.mean, self.std)


class Unnormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, vid):
        return unnormalize(vid, self.mean, self.std)


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, vid):
        if random.random() < self.p:
            return hflip(vid)
        return vid


class Pad(object):
    def __init__(self, padding, fill=0):
        self.padding = padding
        self.fill = fill

    def __call__(self, vid):
        return pad(vid, self.padding, self.fill)

class RandomRotate(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, vid):
        angle = ((random.random() * 2) - 1) * 15
        # return rotate(vid, 15, self.p)
        return vid

class RandomGray(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, vid):
        return random_gray(vid, self.p)

# class ColorJitter(object):
#     def __init__(self, brightness, contrast, saturation, hue, p):
#         self.brightness = brightness
#         self.contrast = contrast
#         self.saturation = saturation
#         self.hue = hue
#         self.p = p
#
#     def __call__(self, vid):
