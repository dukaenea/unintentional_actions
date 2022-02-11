# @Author: Enea Duka
# @Date: 4/20/21

import torch
from PIL import Image
from torch.utils.data import Dataset
import scipy.io
import numpy as np
import os
import torchvision.transforms as trn
# from skimage.feature import hog
from random import randint, random
from torch.utils.data.dataloader import default_collate
from dataloaders.dl_utils import Normalize



class PedestrianDataset(Dataset):
    def __init__(self, mode, return_hog=False, gray_scale=False, in_channels=0):
        super(PedestrianDataset, self).__init__()

        self.mode = mode
        self.ped1_path = '/BS/unintentional_actions/nobackup/ucsd/UCSD_Anomaly_Dataset.v1p2/UCSDped1/%s' % mode
        self.ped2_path = '/BS/unintentional_actions/nobackup/ucsd/UCSD_Anomaly_Dataset.v1p2/UCSDped2/%s' % mode

        self.ped1_vids = [f[0] for f in os.walk(self.ped1_path) if not f[0].endswith('_gt')][1:]
        self.ped2_vids = [f[0] for f in os.walk(self.ped2_path) if not f[0].endswith('_gt')][1:]

        self.vids = self.ped1_vids + self.ped2_vids
        self.return_hog = return_hog
        self.ccrop = trn.CenterCrop((158, 238))
        self.gray_scale = gray_scale
        self.in_channels = in_channels
        self.norm = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        # self.norm = Normalize(mean=[0.4246], std=[0.2045])

        if mode == 'Test':
            p1_labels = None
            p2_labels = None
            with open(
                    '/BS/unintentional_actions/nobackup/ucsd/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Test/labels.txt') as f:
                p1_labels = [line.rstrip() for line in f]
            with open(
                    '/BS/unintentional_actions/nobackup/ucsd/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Test/labels.txt') as f:
                p2_labels = [line.rstrip() for line in f]

            self.labels = p1_labels + p2_labels

    def __len__(self):
        return len(self.vids)

    def tem_reg_collate_fn(self, batch):
        for i in range(len(batch)):
            video = batch[i]['video']
            video_len = video.shape[0]
            rand = random()
            if rand < 0.25:
                sample_start = randint(0, video_len - self.in_channels)
                video = video[sample_start:sample_start+self.in_channels]
            elif 0.25 <= rand < 0.5:
                sample_start = randint(0, video_len - self.in_channels*2)
                video = video[sample_start:sample_start + self.in_channels*2:2]
            elif 0.5 <= rand < 0.75:
                sample_start = randint(0, video_len - self.in_channels * 3)
                video = video[sample_start:sample_start + self.in_channels * 3:3]
            else:
                sample_start = randint(0, video_len - self.in_channels * 4)
                video = video[sample_start:sample_start + self.in_channels * 4:4]
            video = video.squeeze()
            batch[i]['video'] = video

        return default_collate(batch)

    def _pad_zeros(self, video, labels, in_channels):
        vid_len = video.shape[0]
        w, h = video.shape[1], video.shape[2]
        pad_size = in_channels - (vid_len % in_channels)
        pad = torch.zeros((pad_size, w, h))
        labels_pad = torch.ones(pad_size) * -1
        return torch.cat([video, pad], dim=0), torch.cat([labels, labels_pad], dim=0)

    def tem_reg_inference_collate_fn(self, batch):
        # the batch size during inference is one due to the diffeent length of the
        # videos in the test set
        for i in range(len(batch)):
            video = batch[i]['video']
            labels = batch[i]['label']
            if video.shape[0] % self.in_channels != 0:
                video = self._pad_zeros(video, labels, self.in_channels)[0]
            video = torch.stack(list(torch.split(video, self.in_channels, dim=0)))
            batch[i]['video'] = video

        return default_collate(batch)

    def _load_video(self, video_dir):
        frames = []
        for path, subdirs, files in os.walk(video_dir):
            if not path.endswith('_gt'):
                size = None
                image_files = [f for f in files if '.tif' in f]
                for name in sorted(image_files):
                    try:
                        image = Image.open(os.path.join(path, name)).convert('RGB')
                        image = self._process_image(image, 224)
                        image = trn.ToTensor()(image)
                        if size is None:
                            size = image.shape
                    except Exception:
                        print("File %s corrupted! Replacing with black frame" % name)
                        image = torch.zeros(size)
                    # scale the video to the model input size when working on model
                    # if self.return_hog:
                    #     image = image.permute(1, 2, 0)
                    #     image = image.detach().numpy()
                    #     image, hog_vis = hog(image, orientations=8, pixels_per_cell=(16, 16),
                    #                          cells_per_block=(1, 1), visualize=True, multichannel=True,
                    #                          feature_vector=True)
                    #     image = torch.tensor(image)

                    frames.append(image)
        frames = torch.stack(frames)
        frames = self.norm(frames.squeeze())
        return frames

    def _process_image(self, image, size):
        width, height = image.size
        if width < height:
            new_size = (size, int((size * height) / width))
        elif height < width:
            new_size = (int((size * width) / height), size)
        else:
            new_size = (size, size)

        image = image.resize(new_size)
        crop = trn.CenterCrop(size)
        image = crop(image)
        if self.gray_scale:
            image = image.convert('L')
        return image

    def _get_label(self, idx, len):
        label_tensor = torch.zeros(len)
        label = self.labels[idx]
        intervals = label.split(',')

        for interval in intervals:
            boundries = interval.split(':')
            start = int(boundries[0]) - 1
            end = int(boundries[1]) - 1
            label_tensor[start:end] = 1

        return label_tensor

    def __getitem__(self, idx):
        output = {}
        video = self._load_video(self.vids[idx])
        output['features'] = video
        output['filename'] = self.vids[idx].split('/')[-1]+'_'+self.vids[idx].split('/')[-3][-1]
        return output
        if self.mode == 'Train':
            feat_path = self.vids[idx].replace('Train/', 'Train_npy/')
        elif self.mode == 'Test':
            feat_path = self.vids[idx].replace('Test/', 'Test_npy/')
        # npy_video = video.cpu().detach().numpy()
        # np.savez_compressed(feat_path+'.npz', npy_video)
        compressed_file = np.load(feat_path+'.npz')
        array = compressed_file['arr_0']
        video = torch.from_numpy(array)

        output['video'] = video

        if self.mode == 'Test':
            output['label'] = self._get_label(idx, output['video'].shape[0])

        return output
