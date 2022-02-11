# @Author: Enea Duka
# @Date: 8/26/21

import sys

sys.path.append('/BS/unintentional_actions/work/unintentional_actions')

import torch
from torch.utils.data import Dataset
from glob import glob
import os
from dataloaders.dl_utils import Normalize
import av
import torchvision
import utils.py12transforms as T
import numpy as np
import ffmpeg
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
from random import random, randint
import math
from rep_learning.feat_transformer import fast_forward
from torch.utils.data.dataloader import default_collate



normalize = T.Normalize(mean=[0.5, 0.5, 0.5],
                        std=[0.5, 0.5, 0.5])

train_transform = torchvision.transforms.Compose([
    T.ToFloatTensorInZeroOne(),
    T.Resize(224*4),
    T.RandomHorizontalFlip(),
    T.RandomCrop((224, 224)),
    normalize
])

test_transform = torchvision.transforms.Compose([
    T.ToFloatTensorInZeroOne(),
    T.Resize(230),
    # T.RandomHorizontalFlip(),
    T.RandomCrop((224, 224)),
    normalize,
])


class UCFCrimeVideoLoader(Dataset):
    def __init__(self, mode, load_frames=False):
        self.mode = mode
        self.load_frames = load_frames
        self.max_video_frames = 200
        self.speeds = [0, 1, 2, 3]
        self.norm = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        self.csv = pd.read_csv('/BS/unintentional_actions/nobackup/ucf_crime/Temporal_Anomaly_Annotation_for_Testing_Videos.csv')
        self.train_videos_filename = '/BS/unintentional_actions/nobackup/ucf_crime/Anomaly_Train.txt'

        if self.mode == 'train':
            if load_frames:
                self.video_list = []
                with open(self.train_videos_filename) as f:
                    lines = f.read().splitlines()
                    for video_path in lines:
                        if not 'Normal' in video_path:
                            self.video_list.append(os.path.join('/BS/unintentional_actions/nobackup/ucf_crime/Anomaly_Videos', video_path))
                # self.base_video_path = '/BS/unintentional_actions/nobackup/ucf_crime/Training-Normal-Videos'
                # self.video_list = sorted(glob(os.path.join(self.base_video_path, '**', '*.mp4'), recursive=True))
            else:
                self.base_video_path = '/BS/unintentional_actions/nobackup/ucf_crime/vit_features/Training-Normal-Videos-200'
                self.video_list = sorted(glob(os.path.join(self.base_video_path, '**', '*.npz'), recursive=True))

                base_anomaly_path = '/BS/unintentional_actions/nobackup/ucf_crime/vit_features/Anomaly_Videos-200'
                anomaly_video_list = sorted(glob(os.path.join(base_anomaly_path, '**', '*.npz'), recursive=True))

                with open(self.train_videos_filename) as f:
                    lines = f.read().splitlines()
                    for video_path in lines:
                        if not 'Normal' in video_path:
                            video_name = video_path.split('/')[1].split('.')[0]
                            for an_video in anomaly_video_list:
                                if video_name in an_video:
                                    self.video_list.append(an_video)


                # filtered_videos = []
                # for video_path in self.video_list:
                #     original_video_idx = int(video_path.split('/')[-1].split('_')[1][-3:])
                #     if original_video_idx < 123:
                #         filtered_videos.append(video_path)
                # self.video_list = filtered_videos

        elif self.mode == 'val':
            # self.base_video_path = '/BS/unintentional_actions/nobackup/ucf_crime/Testing_Normal_Videos_Anomaly'
            # self.video_list = sorted(glob(os.path.join(self.base_video_path, '**', '*.mp4'), recursive=True))
            if self.load_frames:
                self.video_list = []
                for idx, row in self.csv.iterrows():
                    filename = row['filename']
                    action = row['action']

                    if action == 'Normal':
                        continue
                        # self.video_list.append(
                        #     os.path.join('/BS/unintentional_actions/nobackup/ucf_crime/Testing_Normal_Videos_Anomaly',
                        #                  filename))
                    else:
                        # continue
                        self.video_list.append(
                            os.path.join('/BS/unintentional_actions/nobackup/ucf_crime/Anomaly_Videos', action, filename))
            else:
                self.base_video_path = '/BS/unintentional_actions/nobackup/ucf_crime/vit_features/Testing_Normal_Videos_Anomaly-200'
                self.video_list = sorted(glob(os.path.join(self.base_video_path, '**', '*.npz'), recursive=True))

        if self.load_frames:
            self._cut_up_video_to_fit_in_memory(self.video_list)
        return

    def _cut_up_video_to_fit_in_memory(self, video_list):
        total_seconds = 0
        long_videos = []
        name_to_path = {}
        for video_path in video_list:
            container = av.open(video_path)
            frames = container.streams.video[0].frames
            fr = container.streams.video[0].average_rate
            chunks = frames // self.max_video_frames + 1
            video_name = video_path.split('/')[-1]
            if chunks > 1:
                name, ext = video_name.split('.')
                for i in range(chunks):
                    new_name = name + '_%d.' % i + ext
                    name_to_path[new_name] = video_path
            else:
                name_to_path[video_name] = video_path
            print("%s ------ Frames: %d ---- FPS: %d ----- Chunks: %d" % (video_name, frames, fr, chunks))

            total_seconds += frames / fr

        hrs = total_seconds / 3600

        print("Dataset length in hours: %f" % hrs)
        print(long_videos)

        self.name_to_path = name_to_path

    def __len__(self):
        return len(list(self.name_to_path.keys())) if self.load_frames else len(self.video_list)

    def _get_video_dim(self, video_path):
        probe = ffmpeg.probe(video_path)
        video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        return int(video_stream['height']), int(video_stream['width'])

    def _get_raw_video(self, video_path, chunk):
        height, width = self._get_video_dim(video_path)
        cmd = (ffmpeg.input(video_path))
        container = av.open(video_path)
        frames = container.streams.video[0].frames
        video_fps = 30  # av.open(video_path).streams.video[0].average_rate
        cmd = cmd.filter('fps', fps=video_fps)
        if chunk > 0:
            end = min(self.max_video_frames*(chunk+1), frames)
        else:
            end = self.max_video_frames*(chunk+1)
        start = self.max_video_frames*chunk
        cmd = cmd.trim(start_frame=start, end_frame=end)
        cmd = cmd.setpts('PTS-STARTPTS')

        out, _ = (
            cmd.output('pipe:', format='rawvideo', pix_fmt='rgb24')
                .run(capture_stdout=True, quiet=True)
        )

        video = np.frombuffer(out, np.uint8).reshape([-1, width, height, 3])
        video = torch.from_numpy(video)

        return video

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
        num_frames = video.shape[0]

        # find the max speed possible given the video seq
        max_speed = int(min(math.log2(num_frames / min_seq_len), self.speeds[-1]))
        try:
            speed_label = randint(0, max_speed)
        except Exception:
            print('')
        eff_skip = 2 ** speed_label

        # subsample the video
        video, _ = fast_forward(video, eff_skip)['tensor']
        if video.shape[0] > min_seq_len:
            vid_idx = torch.arange(video.shape[0])
            crop_idx = self._random_crop_vector(vid_idx, min_seq_len)
            video = video[crop_idx]
        return video, speed_label, eff_skip, max_speed

    def _shuffle_video(self, video, min_seq_len):
        full_idx = torch.arange(0, video.shape[0])
        video = self._crop_video_temporally(video, min_seq_len)
        # video = video[crop_vid_idx]
        vid_idx = torch.randperm(min_seq_len - 1)
        vid_idx = torch.gather(full_idx, 0, vid_idx.type(torch.int64))
        if len(video.shape) == 4:
            video = video[vid_idx.type(torch.long), :, :, :]
        elif len(video.shape) == 2:
            video = video[vid_idx.type(torch.long), :]
        return video

    def _foba_video(self, video, eff_skip, min_seq_len):
        vid_len = video.shape[0]

        vid_idx = torch.arange(0, vid_len)
        sub_len = min(vid_len, min_seq_len)  # if eff_skip == 1 else (min_seq_len - 1) * eff_skip)
        crop_idx = self._random_crop_vector(vid_idx, sub_len)
        video = video[crop_idx]
        idxs = torch.arange(0, video.shape[0])  # torch.arange(0, (min_seq_len - 2) * eff_skip, step=eff_skip)
        start = 0  # 1 if eff_skip > 1 else 0
        # rand_offset = randint(start, eff_skip + 1 - start)
        inds_foba = torch.cat([idxs, torch.flip(idxs, dims=(0,))], 0)
        inds_foba = self._random_crop_central_vector(inds_foba, min_seq_len)
        # try:
        #     inds_foba = torch.gather(input=crop_idx, dim=0, index=inds_foba.type(torch.int64))
        # except Exception as e:
        #     print(e)
        #     raise
        if len(video.shape) == 4:
            video = video[inds_foba.type(torch.long), :, :, :]
        elif len(video.shape) == 2:
            try:
                video = video[inds_foba.type(torch.long), :]
            except Exception:
                print('')
        return video

    def _twarp_video(self, video, max_speed, min_seq_len):
        full_inds = torch.arange(0, video.shape[0])
        sub_len = int((min_seq_len - 1) * 2 ** max_speed)
        max_offset = 2 ** max_speed
        off_sets = torch.randint(1, max_offset + 1, size=(min_seq_len,))
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
            warped_vid = video[vid_warp_idx.type(torch.long), :, :, :]
        elif len(video.shape) == 2:
            warped_vid = video[vid_warp_idx.type(torch.long), :]
        return warped_vid

    def _crop_video_temporally(self, video, min_seq_len):
        vid_len = video.shape[0]
        vid_idx = torch.arange(0, vid_len)
        crop_idx = self._random_crop_vector(vid_idx, min_seq_len)
        return video[crop_idx]

    def _random_crop_central_vector(self, vector, crop_len):
        vec_len = vector.shape[0]
        min_start_idx = (vec_len // 2) - (crop_len - 2)
        max_start_idx = (vec_len // 2) - 2

        start_idx = randint(min_start_idx, max_start_idx + 1)
        return vector[start_idx: start_idx + crop_len]

    # def _crop_video_temporally_controlled(self, video, min_seq_len):
    #     video_len = video.shape[0]
    #     video_idx = torch.arange(0, video_len)
    #     max_start_pos = (video_idx.shape[0] - min_seq_len) // 2
    #     start_pos = randint(0, max_start_pos)

    def _stitch_videos(self, nrm_video, abnormal_video, min_seq_len):
        stitched_video = torch.cat([nrm_video, abnormal_video], dim=0)
        quarter_len = nrm_video.shape[0] // 4
        start_idx = randint(quarter_len, quarter_len * 3)
        s_vid = stitched_video[start_idx: start_idx + min_seq_len]
        return s_vid

    def _rep_lrn_collate_fn(self, batch):
        def _zeropad(tensor, size):
            n = size - tensor.shape[0] % size
            if len(tensor.shape) == 2:
                z = -torch.ones((n, tensor.shape[1]))
            elif len(tensor.shape) == 4:
                z = -torch.ones((n, tensor.shape[1], tensor.shape[2], tensor.shape[2]))
            return torch.cat((tensor, z), dim=0)

        new_batch = []
        min_seq_len = 20
        for _ in range(1):
            for idx, data in enumerate(batch):
                video = data['features']
                if video is None:
                    continue
                if self.mode == 'train' and self.load_frames:
                    video = video.permute(0, 3, 1, 2)
                p_speed = 1
                p_motion = 1
                # dset = 'oops'
                dset = 'avenue'
                if video.shape[0] > min_seq_len:

                    # first modify the speed of the video
                    s_video, speed_label, eff_skip, max_speed = self._speed_up_video(video, min_seq_len)
                    nrm_cropped_video = self._crop_video_temporally(video, min_seq_len)
                    trn = randint(0, 7)
                    # if random() < p_speed:
                    # if (trn==0 and speed_label==0) or trn == 1 or trn == 2 or trn == 3:
                    new_batch.append({'features': s_video, 'label': speed_label, 'pure_nr_frames': s_video.shape[0],
                                      'org_vid_idx': -1, 'dset': dset})
                    # then shuffle the video
                    # if random() > 0.5:
                    # if trn == 0 and speed_label != 0:
                    new_batch.append(
                        {'features': nrm_cropped_video, 'label': 0, 'pure_nr_frames': nrm_cropped_video.shape[0],
                         'org_vid_idx': -1, 'dset': dset})
                    # else:
                    # num_noise_frames = randint(1, nrm_cropped_video.shape[0])
                    # num_noise_feats = randint(nrm_cropped_video.shape[1]//10, nrm_cropped_video.shape[1]//5)
                    # noise_feats = torch.randn((num_noise_feats,))
                    # start_idx = randint(0, nrm_cropped_video.shape[1] - num_noise_feats)
                    # if start_idx > 0:
                    #     left = torch.zeros((start_idx,))
                    #     noise_feats = torch.cat((left, noise_feats))
                    # if start_idx < nrm_cropped_video.shape[1] - num_noise_feats:
                    #     right = torch.zeros((nrm_cropped_video.shape[1] - noise_feats.shape[0],))
                    #     noise_feats = torch.cat((noise_feats, right))
                    # noise = noise_feats.repeat(num_noise_frames, 1)
                    # #noise = torch.randn((num_noise_frames, nrm_cropped_video.shape[1]))
                    # pad = torch.zeros((nrm_cropped_video.shape[0]-num_noise_frames, nrm_cropped_video.shape[1]))
                    # noise = torch.cat((pad, noise) if random() > 0.5 else (noise, pad))
                    # pert = nrm_cropped_video + noise
                    #
                    # new_batch.append(
                    #     {'features': pert, 'label': 1, 'pure_nr_frames': nrm_cropped_video.shape[0],
                    #      'org_vid_idx': -1, 'dset': dset})

                    # num_noise_frames = nrm_cropped_video.shape[0]
                    # num_noise_feats = randint(nrm_cropped_video.shape[1] // 10, nrm_cropped_video.shape[1] // 5)
                    # noise_feats = torch.randn((num_noise_feats,))
                    # start_idx = randint(0, nrm_cropped_video.shape[1] - num_noise_feats)
                    # if start_idx > 0:
                    #     left = torch.zeros((start_idx,))
                    #     noise_feats = torch.cat((left, noise_feats))
                    # if start_idx < nrm_cropped_video.shape[1] - num_noise_feats:
                    #     right = torch.zeros((nrm_cropped_video.shape[1] - noise_feats.shape[0],))
                    #     noise_feats = torch.cat((noise_feats, right))
                    # noise = noise_feats.repeat(num_noise_frames, 1)
                    # # noise = torch.randn((num_noise_frames, nrm_cropped_video.shape[1]))
                    # # pad = torch.zeros((nrm_cropped_video.shape[0] - num_noise_frames, nrm_cropped_video.shape[1]))
                    # # noise = torch.cat((pad, noise) if random() > 0.5 else (noise, pad))
                    # pert = nrm_cropped_video + noise
                    #
                    # new_batch.append(
                    #     {'features': pert, 'label': 2, 'pure_nr_frames': nrm_cropped_video.shape[0],
                    #      'org_vid_idx': -1, 'dset': dset})

                    nn_videos = []
                    # if trn == 4:
                    sh_video = self._shuffle_video(video, min_seq_len)
                    # if random() < p_motion:
                    new_batch.append({'features': sh_video, 'label': 4, 'pure_nr_frames': sh_video.shape[0],
                                      'org_vid_idx': -1, 'dset': dset})

                    # then do e foba
                    # if trn == 5:
                    foba_video = self._foba_video(video, eff_skip, min_seq_len)
                    # if random() < p_motion:
                    new_batch.append({'features': foba_video, 'label': 5, 'pure_nr_frames': foba_video.shape[0],
                                      'org_vid_idx': -1, 'dset': dset})

                    # if trn == 6:
                    stitched_video = self._stitch_videos(nrm_cropped_video, s_video, min_seq_len)
                    new_batch.append({'features': stitched_video, 'label': 6, 'pure_nr_frames': stitched_video.shape[0],
                                      'org_vid_idx':  -1, 'dset': dset})
                        # # and at the end warp the time
                    # if trn == 7:
                    twarp_video = self._twarp_video(video, max_speed, min_seq_len)
                    # if random() < p_motion:
                    new_batch.append({'features': twarp_video, 'label': 7, 'pure_nr_frames': twarp_video.shape[0],
                                      'org_vid_idx': -1, 'dset': dset})
                        #
                        # ridx = randint(0, 3)
                    # new_batch.append(nn_videos[ridx])

        if len(new_batch) > 0:
            max_len = max([s['features'].shape[0] for s in new_batch])
            for data in new_batch:
                if data['features'].shape[0] < max_len:
                    if self.mode == 'train' and self.load_frames:
                        data['features'] = _zeropad(train_transform(data['features'].permute(0, 2, 3, 1)), max_len)
                    else:
                        data['features'] = _zeropad(data['features'], max_len)
                else:
                    if self.load_frames:
                        data['features'] = train_transform(data['features'].permute(0, 2, 3, 1))
        return default_collate(new_batch)

    def _load_video_features(self, video_path):
        compressed_file = np.load(video_path)
        video = compressed_file['arr_0']
        video = torch.from_numpy(video)

        return video

    def __getitem__(self, idx):
        # if idx < 870:
        #     return torch.empty((1, ))
        out = {}
        if self.load_frames:
            key = list(self.name_to_path.keys())[idx]
            video_path = self.name_to_path[key]
            name_parts = key.split('_')
            chunk = 0
            if len(name_parts) > 4:
                chunk = int(name_parts[-1].split('.')[0])
            video = self._get_raw_video(video_path, chunk)
            if self.mode == 'val':
                video = test_transform(video)
                # video = video.permute(1, 0, 2, 3)
            else:
                video = train_transform(video)
        else:
            video_path = self.video_list[idx]
            video = self._load_video_features(video_path)
        out['features'] = video
        if self.mode == 'train':
            video_name = video_path.split('/')[-1]
        else:
            elms = video_path.split('/')
            video_name = '/'.join(elms[-2:])
        # out['filename'] = key
        return out


if __name__ == '__main__':
    dset = UCFCrimeVideoLoader('train', load_frames=True)
    loader = DataLoader(dset,
                        batch_size=64,
                        num_workers=32,
                        collate_fn=dset._rep_lrn_collate_fn)

    for i, data in enumerate(tqdm(loader)):
        # print('%d -> %d' % (i, data['features'][0].shape[0]))
        pass
