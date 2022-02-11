# @Author: Enea Duka
# @Date: 4/21/21

import random
import torch
import numpy as np
import cv2
import glob
import os
import scipy.io as scio
from torch.utils.data import Dataset
from dataloaders.dl_utils import tensor_to_zero_one
from random import randint, random, shuffle
from torch.utils.data.dataloader import default_collate
import torchvision.transforms as trn
from dataloaders.dl_utils import Normalize
from utils.arg_parse import opt



def np_load_frame(filename, resize_h, resize_w, gray_scale=False):
    img = cv2.imread(filename)
    if gray_scale:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    image_resized = cv2.resize(img, (resize_w, resize_h)).astype('float32')
    # image_resized = (image_resized / 127.5) - 1.0  # to -1 ~ 1
    if not gray_scale:
        image_resized = np.transpose(image_resized, [2, 0, 1])  # to (C, W, H)
    return image_resized


class AvenueDataset(Dataset):
    """
    No data augmentation.
    Normalized from [0, 255] to [-1, 1], the channels are BGR due to cv2 and liteFlownet.
    """

    def __init__(self, mode, img_size, train_data, in_channels=0, gray_scale=False):
        self.mode = mode
        self.img_h = img_size[0]
        self.img_w = img_size[1]
        self.clip_length = 5
        self.ccrop = trn.CenterCrop(img_size)
        self.gray_scale = gray_scale
        self.in_channels = in_channels
        # self.norm = Normalize(mean=[0.3982], std=[0.2472])
        self.norm = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        self.npy_path = '/BS/unintentional_actions/nobackup/avenue/avenue/%sing_npy' % mode
        if self.mode == 'test':
            label_loader = Label_loader(
                sorted([x[0] for x in os.walk('/BS/unintentional_actions/nobackup/avenue/avenue/testing')][1:]))
            self.labels = label_loader.load_ucsd_avenue()

        self.videos = []
        self.all_seqs = []
        for folder in sorted(glob.glob(f'{train_data}/*')):
            all_imgs = glob.glob(f'{folder}/*.jpg')
            all_imgs.sort()
            self.videos.append(all_imgs)

            random_seq = list(range(len(all_imgs) - 4))
            shuffle(random_seq)
            self.all_seqs.append(random_seq)

    def __len__(self):  # This decide the indice range of the PyTorch Dataloader.
        return len(self.videos)

    def tem_reg_collate_fn(self, batch):
        new_batch = []
        def sample_clip(i):
            video = batch[i]['video']
            video_len = video.shape[0]
            rand = random()
            if rand < 0.25:
                sample_start = randint(0, video_len - self.in_channels)
                video = video[sample_start:sample_start + self.in_channels]
            elif 0.25 <= rand < 0.5:
                sample_start = randint(0, video_len - self.in_channels * 2)
                video = video[sample_start:sample_start + self.in_channels * 2:2]
            elif 0.5 <= rand < 0.75:
                sample_start = randint(0, video_len - self.in_channels * 3)
                video = video[sample_start:sample_start + self.in_channels * 3:3]
            else:
                sample_start = randint(0, video_len - self.in_channels * 4)
                video = video[sample_start:sample_start + self.in_channels * 4:4]
            video = video.squeeze()
            return video

        for i in range(len(batch)):
            sampled_clip = sample_clip(i)
            new_batch.append({'video': sampled_clip})

        # avenue has few but long videos, let's resample the videos until we get batch size num of clips
        while len(new_batch) < opt.batch_size:
            i = randint(0, len(batch)-1)
            sampled_clip = sample_clip(i)
            new_batch.append({'video': sampled_clip})

        return default_collate(new_batch)

    def _pad_zeros(self, video, labels, in_channels):
        vid_len = video.shape[0]
        w, h = video.shape[1], video.shape[2]
        pad_size = in_channels - (vid_len % in_channels)
        pad = torch.zeros((pad_size, w, h))
        labels_pad = torch.ones(pad_size) * labels[-1]
        return torch.cat([video, pad], dim=0), torch.cat([labels, labels_pad], dim=0)

    def tem_reg_inference_collate_fn(self, batch):
        # the batch size during inference is one due to the different length of the
        # videos in the test set
        for i in range(len(batch)):
            video = batch[i]['video']
            labels = batch[i]['label']
            if video.shape[0] % self.in_channels != 0:
                video, labels = self._pad_zeros(video, labels, self.in_channels)
            video = torch.stack(list(torch.split(video, self.in_channels, dim=0)))
            batch[i]['video'] = video
            batch[i]['label'] = labels
        return default_collate(batch)

    def __getitem__(self, idx):  # Indice decide which video folder to be loaded.
        one_folder = self.videos[idx]
        video_name = one_folder[0].split('/')[-2]
        video = []
        start = self.all_seqs[idx][-1]  # Always use the last index in self.all_seqs.
        # for i in range(start, start + self.clip_length):
        for i in range(len(one_folder)):
            video.append(np_load_frame(one_folder[i], self.img_h, self.img_w, self.gray_scale))
        if self.gray_scale:
            video = np.array(video).reshape((-1, self.img_h, self.img_w))
        else:
            video = np.array(video).reshape((-1, 3, self.img_h, self.img_w))
        video = torch.from_numpy(video)
        video = tensor_to_zero_one(video)
        video = self.norm(video)
        out = {}
        out['features'] = video
        # out['label'] = self.labels[idx]
        out['filename'] = video_name

        return out

        npy_video = video.cpu().detach().numpy()
        np.savez_compressed(self.npy_path+'/'+video_name+'.npz', npy_video)
        compressed_file = np.load(self.npy_path+'/'+video_name+'.npz')
        array = compressed_file['arr_0']
        video = torch.from_numpy(array)
        out = {'video': video}
        if self.mode == 'test':
            out['label'] = torch.tensor(self.labels[idx])
        return out


# class test_dataset(Dataset):
#     def __init__(self, img_size, video_folder):
#         self.img_h = img_size[0]
#         self.img_w = img_size[1]
#         self.clip_length = 5
#         self.imgs = glob.glob(video_folder + '/*.jpg')
#         self.imgs.sort()
#
#     def __len__(self):
#         return len(self.imgs) - (self.clip_length - 1)  # The first [input_num] frames are unpredictable.
#
#     def __getitem__(self, indice):
#         video_clips = []
#         for frame_id in range(indice, indice + self.clip_length):
#             video_clips.append(np_load_frame(self.imgs[frame_id], self.img_h, self.img_w))
#
#         video_clips = np.array(video_clips).reshape((-1, self.img_h, self.img_w))
#         return video_clips


class Label_loader:
    def __init__(self, video_folders):
        # assert cfg.dataset in ('ped2', 'avenue', 'shanghaitech'), f'Did not find the related gt for \'{cfg.dataset}\'.'
        self.mat_path = '/BS/unintentional_actions/nobackup/avenue/avenue/avenue.mat'
        self.video_folders = video_folders

    def __call__(self):
        if self.name == 'shanghaitech':
            gt = self.load_shanghaitech()
        else:
            gt = self.load_ucsd_avenue()
        return gt

    def load_ucsd_avenue(self):
        abnormal_events = scio.loadmat(self.mat_path, squeeze_me=True)['gt']

        all_gt = []
        len_sum = 0
        for i in range(abnormal_events.shape[0]):
            length = len(os.listdir(self.video_folders[i]))
            len_sum += length
            sub_video_gt = np.zeros((length,), dtype=np.int8)

            one_abnormal = abnormal_events[i]
            if one_abnormal.ndim == 1:
                one_abnormal = one_abnormal.reshape((one_abnormal.shape[0], -1))

            for j in range(one_abnormal.shape[1]):
                start = one_abnormal[0, j] - 1
                end = one_abnormal[1, j]

                sub_video_gt[start: end] = 1

            all_gt.append(sub_video_gt.tolist())
        return all_gt

    def load_shanghaitech(self):
        np_list = glob.glob(f'{self.cfg.data_root + self.name}/frame_masks/')
        np_list.sort()

        gt = []
        for npy in np_list:
            gt.append(np.load(npy))

        return gt