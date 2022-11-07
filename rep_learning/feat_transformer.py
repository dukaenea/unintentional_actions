# @Author: Enea Duka
# @Date: 4/27/21
import torch
from scipy import ndimage
import torch.nn as nn
from random import random, randint
import kornia as K


def _get_centers(tensor):
    w = tensor.shape[2]
    h = tensor.shape[3]
    return torch.FloatTensor([w // 2, h // 2]).repeat(tensor.shape[0], 1)


def rotate(tensor, rot_angle, p=1.0):
    out = {}
    if random() < p:
        rot_angle_tensor = (torch.rand(tensor.shape[0]) * rot_angle * 2) - rot_angle
        center_tensor = _get_centers(tensor)
        tensor = K.geometry.transform.rotate(tensor, rot_angle_tensor, center_tensor)
        out["trn_name"] = "rotate"
    out["tensor"] = tensor
    return out


def shift(tensor, axial_shift, p=1.0):
    out = {}
    if random() < p:
        translation = torch.FloatTensor(axial_shift).repeat(tensor.shape[0], 1)
        tensor = K.geometry.transform.translate(tensor, translation)
        out["trn_name"] = "shift"
    out["tensor"] = tensor
    return out


def scale(tensor, scale_factor, p=1.0):
    out = {}
    if random() < p:
        scale_tensor = torch.FloatTensor([scale_factor]).repeat(tensor.shape[0], 2)
        center_tensor = _get_centers(tensor)

        tensor = K.geometry.transform.scale(tensor, scale_tensor, center_tensor)
        out["trn_name"] = "scale"
    out["tensor"] = tensor
    return out


def center_crop(tensor, axial_out_size):
    out = {}
    tensor = K.geometry.transform.center_crop(tensor, axial_out_size)
    out["trn_name"] = "center_crop"
    out["tensor"] = tensor
    return out


def spacial_cutout(tensor, p=1.0):
    out = {}
    if random() < p:
        w = tensor.shape[2]
        h = tensor.shape[3]

        # sample a tuple of random numbers to use for the center of the cutout
        c_x = randint(0, w)
        c_y = randint(0, h)

        # sample a couple of random numbers to use as the width and height of the cutout
        c_w = randint(2, w - 2)
        c_h = randint(2, h - 2)

        # fill the corresponding rectangle with 0s

        tensor[
            :,
            :,
            max(0, c_x - c_w // 2) : min(c_x + c_w // 2, w),
            max(0, c_y - c_h // 2) : min(c_y + c_h // 2, h),
        ] = 0
        out["trn_name"] = "spacial_cutout"
    # else:
    #     c_size = tensor.shape[2] if tensor.shape[2] > tensor.shape[3] else tensor.shape[3]
    #     tensor = center_crop(tensor, [c_size, c_size])
    #     out['trn_name'] = 'center_crop'
    out["tensor"] = tensor
    return out


def random_crop(tensor, p=1.0):
    out = {}
    c_size = tensor.shape[2] if tensor.shape[2] < tensor.shape[3] else tensor.shape[3]
    if random() < p:
        r_c = K.augmentation.RandomCrop((c_size, c_size), p=1.0, same_on_batch=False)
        tensor = r_c(tensor)
        out["trn_name"] = "random_crop"
    # else:
    #     tensor = center_crop(tensor, [c_size, c_size])
    out["tensor"] = tensor
    return out


def flip(tensor, p=0):
    out = {}
    if random() > p:
        tensor = K.geometry.transform.hflip(tensor)
        out["trn_name"] = "flip"
    out["tensor"] = tensor
    return out


def temporal_cutout(tensor, p=1.0, fill_gaussian=False, merge_temp=False):
    out = {}
    if merge_temp:
        tensor = merge_along_time(tensor)
    if random() < p:
        duration = tensor.shape[1]
        width = tensor.shape[2]
        height = tensor.shape[3]
        cut_length = randint(duration // 4, duration // 2)
        cut_center = randint(0, duration)
        if fill_gaussian:
            noise = torch.randn(
                (
                    min(cut_center + cut_length // 2, duration)
                    - max(0, cut_center - cut_length // 2),
                    width,
                    height,
                )
            )
            tensor[
                :,
                max(0, cut_center - cut_length // 2) : min(
                    cut_center + cut_length // 2, duration
                ),
                :,
                :,
            ] = noise
        else:
            tensor[
                :,
                max(0, cut_center - cut_length // 2) : min(
                    cut_center + cut_length // 2, duration
                ),
                :,
                :,
            ] = 0
        out["trn_name"] = "temporal_cutout"
    out["tensor"] = tensor
    return out


def merge_along_time(tensor):
    # reshape the tensor from
    # [n_chunk, n_channels, t, w, h] to
    # [n_channels, t*n_chunk, w, h]
    tensor = torch.cat(list(tensor), dim=1)

    return tensor


def fast_forward(tensor, speed, p=1.0, merge_temp=False):
    out = {}
    if merge_temp:
        tensor = merge_along_time(tensor)
    if random() < p:
        # take every even time slice
        # it should emulate the video being
        # sped up
        if len(tensor.shape) == 4:
            indexes = list(range(0, tensor.shape[0], speed))
            tensor = tensor[indexes, :, :, :]
        elif len(tensor.shape) == 2:
            indexes = list(range(0, tensor.shape[0], speed))
            tensor = tensor[indexes, :]
        out["trn_name"] = "fast_forward"
    out["tensor"] = (tensor, indexes)
    return out


def pad_time_zeros(t):
    pad = torch.zeros((t.shape[0], 10 - t.shape[1], t.shape[2], t.shape[3]))
    return torch.cat([t, pad], dim=1)


def pad_time_mirror(t, max_len, mean_padding=False):
    try:
        pad_len = max_len - t.shape[1]
        repeat = pad_len // t.shape[1]  # how many times do we need to replicate t
        w, h = t.shape[2], t.shape[3]
        tc = t.detach().clone()
        if repeat > 1:
            backwards = True
            for _ in range(repeat - 1):
                if backwards:
                    t = torch.cat([t, torch.flip(tc, dims=[1])], dim=1)
                else:
                    t = torch.cat([t, tc], dim=1)
                backwards = not backwards  # flip direction for the next copy
        remain = max_len - t.shape[1]
        if pad_len > 0:
            if mean_padding:
                means = t.mean(3).mean(2).mean(1)
                pad = (
                    means.unsqueeze(1).unsqueeze(2).unsqueeze(2).repeat(1, remain, w, h)
                )
            else:
                pad = tc[:, -remain:, :, :]
            t = torch.cat([t, pad], dim=1)
    except RuntimeError:
        print(t.shape)
        print(pad.shape)
        raise

    return t


def time_warp(t, p=1):
    out = {}
    if random() < p:
        # matrix of the interpolation weights
        c, w, h = t.shape[0], t.shape[2], t.shape[3]
        I = torch.rand(c, w, h)
        duration = t.shape[1] - 1
        # for the fast forward segment
        ff_c = randint(0, duration)
        ff_l = randint(duration // 6, duration // 3)

        # for the slow down segment
        # no overlap with the ff segment
        # else the segments will both be shorten
        sd_l = randint(duration // 6, duration // 3)
        # do slow down before or after ff
        lb = max(0, ff_c - ff_l // 2)
        la = duration - min(ff_c + ff_l // 2, duration)
        if lb > la:
            sd_c = max(0, ff_c - ff_l - sd_l // 2)
        else:
            sd_c = min(ff_c + ff_l + sd_l // 2, duration)

        # do the slow down
        t = t.permute(1, 0, 2, 3)
        pivot = max(0, sd_c - sd_l // 2)
        end = min(sd_c + sd_l // 2, duration)
        while pivot < end <= duration:
            try:
                nf = (t[pivot] * I + t[pivot + 1] * (1 - I)).unsqueeze(0)
                t = torch.cat([t[: pivot + 1], nf, t[pivot + 1 :]], dim=0)
            except IndexError:
                print("-----------------------")
                print("sd_c: " + str(sd_c))
                print("sd_l: " + str(sd_l))
                print("dur: " + str(t.shape[0]))
                print("pivot: " + str(pivot))
                print("end: " + str(end))
                raise
            pivot += 2
            if end <= duration:
                end += 1

        t = t.permute(1, 0, 2, 3)

        # do the fast forward
        i_rm = list(range(max(0, ff_c - ff_l), min(ff_c + ff_l, duration), 2))
        i_k = list(set(range(0, duration + 1)) - set(i_rm))
        t = t[:, i_k, :, :]
        out["trn_name"] = "time_warp"
    out["tensor"] = t
    return out


def transform_tensor_time(tensor, p):
    transforms = []
    out = temporal_cutout(tensor, p)
    tensor = out["tensor"]
    if "trn_name" in out.keys():
        transforms.append(out["trn_name"])

    out = fast_forward(tensor, randint(1, 5), p)
    tensor = out["tensor"]
    if "trn_name" in out.keys():
        transforms.append(out["trn_name"])

    out = time_warp(tensor, p)
    tensor = out["tensor"]
    if "trn_name" in out.keys():
        transforms.append(out["trn_name"])

    return tensor, transforms


def transform_tensor_space_time(tensor, p):
    transforms = []

    out = rotate(tensor, p)
    tensor = out["tensor"]
    if "trn_name" in out.keys():
        transforms.append(out["trn_name"])

    out = shift(tensor, [2, 2], p)
    tensor = out["tensor"]
    if "trn_name" in out.keys():
        transforms.append(out["trn_name"])

    out = scale(tensor, 2, p)
    tensor = out["tensor"]
    if "trn_name" in out.keys():
        transforms.append(out["trn_name"])

    out = center_crop(tensor, [4, 4])
    tensor = out["tensor"]

    out = flip(tensor, p)
    tensor = out["tensor"]
    if "trn_name" in out.keys():
        transforms.append(out["trn_name"])

    out = spacial_cutout(tensor, p)
    tensor = out["tensor"]
    if "trn_name" in out.keys():
        transforms.append(out["trn_name"])

    out = temporal_cutout(tensor, p)
    tensor = out["tensor"]
    if "trn_name" in out.keys():
        transforms.append(out["trn_name"])

    out = fast_forward(tensor, randint(1, 5), p)
    tensor = out["tensor"]
    if "trn_name" in out.keys():
        transforms.append(out["trn_name"])

    out = time_warp(tensor, p)
    tensor = out["tensor"]
    if "trn_name" in out.keys():
        transforms.append(out["trn_name"])

    return tensor, transforms
