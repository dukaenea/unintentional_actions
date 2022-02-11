# @Author: Enea Duka
# @Date: 4/15/21
import sys

sys.path.append('/BS/unintentional_actions/work/unintentional_actions')
from torch.utils.data import Dataset
from utils.util_functions import Labels, trn_label2idx
from dataloaders.dl_utils import Normalize, tensor_to_zero_one
from rep_learning.feat_transformer import merge_along_time, pad_time_mirror, transform_tensor_time, \
    transform_tensor_space_time, center_crop, fast_forward
from random import random, randint
from torch.utils.data.dataloader import default_collate
from utils.arg_parse import opt
from tqdm import tqdm
import torch.nn.functional as F
import os.path as ops
import numpy as np
import pandas as pd
import ffmpeg
import torch
import math
import json
import os
import av


class KineticsDataset(Dataset):
    def __init__(self, mode, fps, spat_crop, hflip, norm_statistics,
                 spat_scale=False, fpc=1, feat_ext=False,
                 data_level='frames', feat_set='r2plus1d_feats'):
        super(KineticsDataset, self).__init__()

        self.mode = mode
        self.fps = fps
        self.fpc = fpc
        self.feat_ext = feat_ext
        self.spat_crop = spat_crop
        self.spat_scale = spat_scale
        self.hflip = hflip
        self.size = 224
        self.train_n_splits = 1
        self.test_n_splits = 10
        self.training = True if mode == 'train' else False
        self.norm = Normalize(mean=norm_statistics['mean'], std=norm_statistics['std'])
        self.data_level = data_level
        self.speeds = [0, 1, 2, 3]

        self.classes_path = '/BS/unintentional_actions/work/data/kinetics/splits/rep_lrn_classes.txt'
        self.csv_path = '/BS/unintentional_actions/work/data/kinetics/splits/%s_rep_lrn.csv' % mode
        if data_level == 'frames':
            self.video_path = '/BS/unintentional_actions/nobackup/kinetics400/data/%s' % mode
        if data_level == 'features':
            # self.video_path = '/BS/feat_augm/nobackup/kinetics/%s/rep_lrn/%s' % (feat_set, mode)
            self.video_path = '/BS/unintentional_actions/work/data/kinetics/vit_features/%s' % mode

        self.labels = Labels(self.classes_path)
        self.csv = pd.read_csv(self.csv_path)
        self.nclasses = len(self.labels)

    def __len__(self):
        return len(self.csv)

    def _get_video_dim(self, video_path):
        probe = ffmpeg.probe(video_path)
        video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        return int(video_stream['height']), int(video_stream['width'])

    def _get_output_dim(self, h, w):
        if self.spat_scale and self.training:
            size = np.random.randint(self.size + 2, max(h, w, self.size) * 2)
        else:
            size = self.size
        if h >= w:
            return int(h * size / w), size
        else:
            return size, int(w * size / h)

    def _preprocess_video(self, tensor):

        def _zeropad(tensor, size):
            n = size - len(tensor) % size
            z = torch.zeros((n, tensor.shape[1], tensor.shape[2], tensor.shape[3]))
            return torch.cat((tensor, z), dim=0)

        tensor = _zeropad(tensor, self.fpc)
        # tensor = self._cut(tensor, 16)
        tensor = tensor_to_zero_one(tensor)
        tensor = self.norm(tensor)
        penult_dim, last_dim = tensor.shape[-2:]
        tensor = tensor.view(-1, self.fpc, 3, penult_dim, last_dim)
        tensor = tensor.transpose(1, 2)
        if self.feat_ext:
            return tensor

        if self.training:
            snips = np.random.choice(range(tensor.shape[0]), self.train_n_splits)
            tensor = tensor[snips]
        else:
            snips = np.linspace(0, tensor.shape[0], num=self.test_n_splits, endpoint=False, dtype=int)
            tensor = tensor[snips]
        # print('Preprocessing :', tensor.shape)
        return tensor

    def _get_video_feats(self, video_path):
        try:
            compressed_file = np.load(video_path)
            array = compressed_file['arr_0']
            video = torch.from_numpy(array)
        except Exception:
            return None
        # features are saved as float16
        # this is not compatible with other elements of the model
        # which use float32
        # cast the video feats to float 32
        video = video.type(torch.float32)
        if torch.any(torch.isnan(video)):
            video = torch.zeros_like(video)
        shape_len = len(video.shape)
        if shape_len == 5:
            CN, C, T, W, H = video.shape
        elif shape_len == 3:
            CN, C, T = video.shape
        # video = video.flatten()
        # video = F.normalize(video, p=2, dim=0)
        # if shape_len == 5:
        #     video = video.reshape(CN, C, T, W, H)
        # else:
        #     video = video.reshape(CN, C, T)
        # print(torch.max(video))
        # print(torch.min(video))
        return video

    def _get_video(self, video_path):
        h, w = self._get_video_dim(video_path)
        height, width = self._get_output_dim(h, w)
        cmd = (
            ffmpeg
                .input(video_path)
                .filter('scale', width, height)
        )
        video_fps = av.open(video_path).streams.video[0].average_rate
        cmd = cmd.filter('fps', fps=video_fps)

        if self.spat_crop:
            if not self.training:
                x = int((width - self.size) / 2.0)
                y = int((height - self.size) / 2.0)
                cmd = cmd.crop(x, y, self.size, self.size)
                height, width = self.size, self.size
            else:
                if (width - self.size) // 2 <= 0:
                    x = 0
                else:
                    x = np.random.randint(0, (width - self.size) // 2)

                if (height - self.size) // 2 <= 0:
                    y = 0
                else:
                    y = np.random.randint(0, (height - self.size) // 2)

                cmd = cmd.crop(x, y, self.size, self.size)
                height, width = self.size, self.size

        if self.hflip:
            if np.random.rand() > 0.5:
                cmd = cmd.hflip()

        out, _ = (
            cmd.output('pipe:', format='rawvideo', pix_fmt='rgb24')
                .run(capture_stdout=True, quiet=True)
        )

        video = np.frombuffer(out, np.uint8).reshape([-1, height, width, 3])
        video = torch.from_numpy(video.astype('float32'))
        video = video.permute(0, 3, 1, 2)

        # make difference between train and test
        if self.fpc > 1:  # we want clips in this case, not the whole video
            video = self._preprocess_video(video)
        return video

    def get_video_path(self, idx):
        video_name = ops.splitext(self.csv.iloc[idx]['filename'])[0] + (
            '.mp4' if self.data_level == 'frames' else '.npz')
        if self.data_level == 'frames':
            video_path = ops.join(self.video_path, self.csv['label'][idx].replace('_', ' '), video_name)
        else:
            video_path = ops.join(self.video_path, self.csv['label'][idx], video_name)
        return video_path

    def __getitem__(self, idx):

        output = {}

        video_path = self.get_video_path(idx)
        video = self._get_video(video_path) if self.data_level == 'frames' else self._get_video_feats(
            video_path)
        if video is None:
            print(video_path)
        if torch.any(torch.isnan(video)) or video is None:
            counter = 0
            while counter < 10:
                video_path = self.get_video_path(randint(0, len(self.csv)))
                video = self._get_video(video_path) if self.data_level == 'frames' else self._get_video_feats(
                    video_path)
                counter += 1
                if not torch.any(torch.isnan(video)) or video is not None:
                    break

        output['features'] = video
        output['label'] = self.labels[self.csv['label'][idx]]
        output['label_text'] = self.csv['label'][idx]
        output['video_name'] = self.csv['filename'][idx]
        return output

    def _random_crop_vector(self, vector, crop_size):
        try:
            max_start_pos = (vector.shape[0] - crop_size) // 2
            start_pos = randint(0, max_start_pos)
            return vector[start_pos:start_pos + crop_size]
        except Exception:
            print(vector.shape[0])
            print(crop_size)
            print('here')
            if crop_size > vector.shape[0]:
                return vector
            else:
                return vector[:crop_size]

    def _speed_up_video(self, video, min_seq_len):
        if len(video.shape) == 4:
            num_frames = video.shape[1]
        elif len(video.shape) == 2:
            num_frames = video.shape[0]

        # find the max speed possible given the video seq
        max_speed = int(min(math.log2(num_frames/min_seq_len), self.speeds[-1]))
        try:
            speed_label = randint(0, max_speed)
        except Exception:
            print('')
        eff_skip = 2 ** speed_label

        # subsample the video
        video = fast_forward(video, eff_skip)['tensor']
        if video.shape[0] > min_seq_len:
            vid_idx = torch.arange(video.shape[0])
            crop_idx =self._random_crop_vector(vid_idx, min_seq_len)
            video = video[crop_idx]
        return video, speed_label, eff_skip, max_speed

    def _shuffle_video(self, video, min_seq_len):
        if len(video.shape) == 4:
            full_idx = torch.arange(0, video.shape[1])
        elif len(video.shape) == 2:
            full_idx = torch.arange(0, video.shape[0])

        vid_idx = torch.randperm(min_seq_len-1)
        vid_idx = torch.gather(full_idx, 0, vid_idx.type(torch.int64))
        if len(video.shape) == 4:
            video = video[:, vid_idx.type(torch.long), :, :]
        elif len(video.shape) == 2:
            video = video[vid_idx.type(torch.long), :]
        return video

    def _foba_video(self, video, eff_skip, min_seq_len):
        if len(video.shape) == 4:
            vid_len = video.shape[1]
        elif len(video.shape) == 2:
            vid_len = video.shape[0]

        vid_idx = torch.arange(0, vid_len)
        sub_len = min(vid_len, min_seq_len if eff_skip == 1 else (min_seq_len-1) * eff_skip)
        crop_idx = self._random_crop_vector(vid_idx, sub_len)
        idxs = torch.arange(0, (min_seq_len-2)*eff_skip, step=eff_skip)
        start = 1 if eff_skip > 1 else 0
        rand_offset = randint(start, eff_skip + 1 - start)
        inds_foba = torch.cat([idxs, rand_offset + torch.flip(idxs, dims=(0, ))], 0)
        inds_foba = self._random_crop_vector(inds_foba, min_seq_len-1)
        try:
            inds_foba = torch.gather(input=crop_idx, dim=0, index=inds_foba.type(torch.int64))
        except Exception:
            print('')
        if len(video.shape) == 4:
            video = video[:, inds_foba.type(torch.long), :, :]
        elif len(video.shape) == 2:
            try:
                video = video[inds_foba.type(torch.long), :]
            except Exception:
                print('')
        return video

    def _twarp_video(self, video, max_speed, min_seq_len):
        if len(video.shape) == 4:
            full_inds = torch.arange(0, video.shape[1])
        elif len(video.shape) == 2:
            full_inds = torch.arange(0, video.shape[0])
        sub_len = int((min_seq_len-1) * 2 ** max_speed)
        max_offset = 2 ** max_speed
        off_sets = torch.randint(1, max_offset+1, size=(min_seq_len, ))
        inds_warp_v3 = torch.cumsum(off_sets, dim=0)

        inds_warp_v1 = torch.randperm(sub_len)
        inds_warp_v1 = inds_warp_v1[:min_seq_len]
        inds_warp_v1, _ = torch.sort(inds_warp_v1)
        inds_warp = inds_warp_v3 if max_speed > 1 else inds_warp_v1

        vid_warp_idx = self._random_crop_vector(full_inds, sub_len)
        if vid_warp_idx is None:
            print(full_inds)
            print(sub_len)
        vid_warp_idx = torch.gather(vid_warp_idx, 0, inds_warp.type(torch.int64))
        # warped_vid = video[:, vid_warp_idx.type(torch.long), :, :]
        if len(video.shape) == 4:
            warped_vid = video[:, vid_warp_idx.type(torch.long), :, :]
        elif len(video.shape) == 2:
            warped_vid = video[vid_warp_idx.type(torch.long), :]
        return warped_vid

    def speed_and_motion_collate_fn(self, batch):
        def _zeropad(tensor, size):
            n = size - tensor.shape[0] % size
            z = -torch.ones((n, tensor.shape[1]))
            return torch.cat((tensor, z), dim=0)
        new_batch = []
        if opt.consist_lrn:
            new_nrm_batch = []
        min_seq_len = 20
        for idx, data in enumerate(batch):
            video = data['features']
            p_speed = 1
            p_motion = 1
            if video.shape[0] > min_seq_len:
                if opt.consist_lrn:
                    new_nrm_batch.append(
                        {'features': video, 'label': 0, 'pure_nr_frames': video.shape[0], 'org_vid_idx': -1})
                # first modify the speed of the video
                s_video, speed_label, eff_skip, max_speed = self._speed_up_video(video, min_seq_len)
                if random() < p_speed:
                    new_batch.append({'features': s_video, 'label': speed_label, 'pure_nr_frames': s_video.shape[0],
                                      'org_vid_idx': idx if opt.consist_lrn else -1})
                # then shuffle the video
                sh_video = self._shuffle_video(video, min_seq_len)
                if random() < p_motion:
                    new_batch.append({'features': sh_video, 'label': 4, 'pure_nr_frames': sh_video.shape[0],
                                      'org_vid_idx': idx if opt.consist_lrn else -1})

                # then do e foba
                foba_video = self._foba_video(video, eff_skip, min_seq_len)
                if random() < p_motion:
                    new_batch.append({'features': foba_video, 'label': 5, 'pure_nr_frames': foba_video.shape[0],
                                      'org_vid_idx': idx if opt.consist_lrn else -1})

                # and at the end warp the time
                twarp_video = self._twarp_video(video, max_speed, min_seq_len)
                if random() < p_motion:
                    new_batch.append({'features': twarp_video, 'label': 6, 'pure_nr_frames': twarp_video.shape[0],
                                      'org_vid_idx': idx if opt.consist_lrn else -1})
        if opt.consist_lrn:
            new_batch += new_nrm_batch

        if len(new_batch) > 0:
            max_len = max([s['features'].shape[0] for s in new_batch])
            for data in new_batch:
                if data['features'].shape[0] < max_len:
                    data['features'] = _zeropad(data['features'], max_len)

        return default_collate(new_batch)

    # def pad_videos_collate_fn(self, batch):
    #     def _zeropad(tensor, size):
    #         n = size - tensor.shape[1] % size
    #         z = torch.zeros((tensor.shape[0], n, tensor.shape[2], tensor.shape[3]))
    #         return torch.cat((tensor, z), dim=1)
    #
    #     max_batch_len = max([s['features'].shape[0] * s['features'].shape[2] for s in batch])
    #     for data in batch:
    #         video = data['features']
    #         video = merge_along_time(video).squeeze()
    #         data['pure_nr_frames'] = video.shape[1]
    #         if video.shape[1] < max_batch_len:
    #             video = _zeropad(video, max_batch_len)
    #         data['features'] = video
    #     return default_collate(batch)

    def pad_videos_collate_fn(self, batch):
        def _zeropad(tensor, size):
            n = size - tensor.shape[0] % size
            z = torch.zeros((n, tensor.shape[1]))
            return torch.cat((tensor, z), dim=0)

        max_batch_len = max([s['features'].shape[0] for s in batch])
        for data in batch:
            video = data['features']
            # video = merge_along_time(video).squeeze()
            data['pure_nr_frames'] = video.shape[0]
            if video.shape[0] < max_batch_len:
                video = _zeropad(video, max_batch_len)
            data['features'] = video
        return default_collate(batch)

    def trn_rec_coll_fn(self, batch):
        max_batch_len = max([s['features'].shape[0] * s['features'].shape[2] for s in batch])
        for i in range(len(batch)):
            video = batch[i]['features']
            # concatenate all the clips of the video
            video = merge_along_time(video)

            # avg over the spatial components if they are present
            if len(video.shape) == 4:
                video = video.mean(2).mean(2)

    def feat_load_act_rec_collate_fn(self, batch):
        max_batch_len = max([s['features'].shape[0] * s['features'].shape[2] for s in batch])
        min_spat_size = 4
        for i in range(len(batch)):
            vid_feats = batch[i]['features']
            vid_feats = merge_along_time(vid_feats)
            # if sample has no spatial component, fake it by adding two dummy dimensions
            # they will be gone when the feature is flattened to pass it to vtn
            if len(vid_feats.shape) == 2:
                vid_feats = vid_feats.unsqueeze(-1).unsqueeze(-1)
                vid_feats = vid_feats.repeat(1, 1, 4, 4)
            else:
                vid_feats = center_crop(vid_feats, [min_spat_size, min_spat_size])['tensor']
            if vid_feats.shape[1] < max_batch_len:
                vid_feats = pad_time_mirror(vid_feats, max_batch_len, mean_padding=True)
            if torch.any(torch.isnan(vid_feats)):
                print('here')
            batch[i]['features'] = vid_feats
        return default_collate(batch)

    def feat_load_collate_fn(self, batch):
        # find the longest video in the batch
        max_batch_len = max([s['features'].shape[0] * s['features'].shape[2] for s in batch])
        min_spat_size = 4
        for i in range(len(batch)):
            vid_feats = batch[i]['features']
            vid_feats = merge_along_time(vid_feats)
            # if sample has no spatial component, fake it by adding two dummy dimensions
            # they will be gone when the feature is flattened to pass it to vtn
            if len(vid_feats.shape) == 2:
                vid_feats = vid_feats.unsqueeze(-1).unsqueeze(-1)

            # apply transformations to the features
            if vid_feats.shape[2] == 1 and vid_feats.shape[3] == 1:  # case when only having temp component
                # firstly repeat the value in the last two dimensions to get a 4x4 spat resolution
                vid_feats = vid_feats.repeat(1, 1, 4, 4)
                vid_feats, trn_labels = transform_tensor_time(vid_feats, 0.5)
                # since the spatial resolution of the samples without spatial component is the
                # same as of those with spatial component we add the below label to make the
                # samples distinguishable
            else:
                vid_feats, trn_labels = transform_tensor_space_time(vid_feats, 0.5)

            # pad when needed
            if vid_feats.shape[1] < max_batch_len:
                vid_feats = pad_time_mirror(vid_feats, max_batch_len, mean_padding=True)

            # if 'fast_forward' in trn_labels or 'time_warp' in trn_labels:
            #     vid_feats = pad_time_mirror(vid_feats, max_batch_len, mean_padding=True)

            batch[i]['features'] = vid_feats
            batch[i]['trn_labels'] = trn_label2idx(trn_labels)

        return default_collate(batch)
