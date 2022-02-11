import torch.utils.data as data
import numpy as np
import torch
from torch.utils.data import DataLoader
torch.set_default_tensor_type('torch.cuda.FloatTensor')
import os
from tqdm import tqdm
import random

class Dataset(data.Dataset):
    def __init__(self, modality, is_normal=True, transform=None, test_mode=False):
        self.modality = modality
        self.is_normal = is_normal

        if test_mode:
            self.rgb_list_file = 'list/ucf-i3d-test.list'
        else:
            self.rgb_list_file = 'list/ucf-i3d.list'

        self.tranform = transform
        self.test_mode = test_mode
        self._parse_list()
        self.num_frame = 0
        self.labels = None

        if not self.is_normal and not self.test_mode:
            self.resamplable_idxs = []
            for idx, path in enumerate(tqdm(self.list)):
                features = np.load(path.strip('\n'), allow_pickle=True)
                features = np.array(features, dtype=np.float32)
                if len(features) < 512:
                    self.resamplable_idxs.append(idx)

    def _parse_list(self):
        self.list = list(open(self.rgb_list_file))
        if self.test_mode is False:
            if self.is_normal:
                self.list = self.list[810:]
                print('normal list for ucf')
                print(self.list)
            else:
                self.list = self.list[:810]
                print('abnormal list for ucf')
                print(self.list)

    def __getitem__(self, index):

        label = self.get_label()  # get video level label 0/1
        features = np.load(self.list[index].strip('\n'), allow_pickle=True)
        features = np.array(features, dtype=np.float32)

        if not self.is_normal and len(features) > 512:
            index = random.choice(self.resamplable_idxs)
            features = np.load(self.list[index].strip('\n'), allow_pickle=True)
            features = np.array(features, dtype=np.float32)

        if self.tranform is not None:
            features = self.tranform(features)
        if self.test_mode:
            input_len = features.shape[0]
            features = features.transpose(1, 0, 2)  # [10, B, T, F]
            # divided_features = []
            # for feature in features:
            #     feature, rem_clip_len = process_test_feat(feature, 16)  # divide a video into 32 segments
            #     divided_features.append(feature)
            # divided_features = np.array(divided_features, dtype=np.float32)
            return features
        else:
            # process 10-cropped snippet feature
            features = features.transpose(1, 0, 2)  # [10, B, T, F]
            divided_features = []
            for feature in features:
                feature = process_feat(feature, 32)  # divide a video into 32 segments
                divided_features.append(feature)
            divided_features = np.array(divided_features, dtype=np.float32)

            return divided_features, label

    def get_label(self):

        if self.is_normal:
            label = torch.tensor(0.0)
        else:
            label = torch.tensor(1.0)

        return label

    def __len__(self):
        return len(self.list)

    def get_num_frames(self):
        return self.num_frame

def process_test_feat(feat, window):
    feat_len = feat.shape[0]

    full_clips = feat_len // window
    rem_clip_len = feat_len % window
    term = 0 if rem_clip_len == 0 else 1

    new_feature = np.zeros((full_clips+term, feat.shape[1])).astype(np.float32)
    if term == 0:
        a = 1
    for i in range(full_clips+term):
        if ((i+1)*window) < feat_len:
            if len(feat[(i*window):((i+1)*window), :]) == 0:
                a = 1
            new_feature[i, :] = np.mean(feat[(i*window):((i+1)*window), :], 0)
        else:
            if len(feat[(i * window):(feat_len-1), :]) == 0:
                a = 1
            new_feature[i, :] = np.mean(feat[(i * window):feat_len, :], 0)
    if len(new_feature) == 0:
        a = 1
    return new_feature, rem_clip_len


def process_feat(feat, length):
    new_feat = np.zeros((length, feat.shape[1])).astype(np.float32)

    r = np.linspace(0, len(feat), length + 1, dtype=int)
    for i in range(length):
        if r[i] != r[i + 1]:
            new_feat[i, :] = np.mean(feat[r[i]:r[i + 1], :], 0)
        else:
            new_feat[i, :] = feat[r[i], :]
    return new_feat

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    train_nloader = DataLoader(Dataset('train', test_mode=False, is_normal=False),
                               batch_size=1, shuffle=True,
                               num_workers=0, pin_memory=False, drop_last=True)

    for idx, data in enumerate(tqdm(train_nloader)):
        pass