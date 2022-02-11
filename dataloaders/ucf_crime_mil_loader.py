# @Author: Enea Duka
# @Date: 10/10/21

import sys

sys.path.append('/BS/unintentional_actions/work/unintentional_actions')

import torch
from torch.utils.data import Dataset
from glob import glob
import os
from utils.logging_setup import logger
from dataloaders.dl_utils import Normalize
import av
import torchvision
import utils.py12transforms as T
import numpy as np
import ffmpeg
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
from random import random, randint, choice
import math
from rep_learning.feat_transformer import fast_forward
from torch.utils.data.dataloader import default_collate


class CrimeFeatDataset(Dataset):
    def __init__(self, iterations, frames_per_clip, clip_stride, segments_per_video):

        self.iterations = iterations
        self.frames_per_clip = frames_per_clip
        self.clip_stride = clip_stride
        self.segments_per_video = segments_per_video
        self.i = 0

        self.training_file_names = '/BS/unintentional_actions/nobackup/ucf_crime/Anomaly_Train.txt'
        self.testing_csv = pd.read_csv(
            '/BS/unintentional_actions/nobackup/ucf_crime/Temporal_Anomaly_Annotation_for_Testing_Videos.csv')

        base_anomaly_path = '/BS/unintentional_actions/nobackup/ucf_crime/vit_features/Anomaly_Videos-200'
        base_normal_path = '/BS/unintentional_actions/nobackup/ucf_crime/vit_features/Training-Normal-Videos-200'

        self.anomalous_chunks = sorted(glob(os.path.join(base_anomaly_path, '**', '*.npz'), recursive=True))
        self.normal_chunks = sorted(glob(os.path.join(base_normal_path, '**', '*.npz'), recursive=True))

        # first get all the filenames for the training samples
        self.normal_videos_to_chunks = {}
        self.anomalous_videos_to_chunks = {}

        self.used_normal_keys = []
        self.used_anomal_keys = []

        with open(self.training_file_names) as f:
            lines = f.read().splitlines()

            for line in tqdm(lines):
                if 'Normal' in line:
                    self.normal_videos_to_chunks[line] = []
                    video_name = line.split('.')[0]
                else:
                    self.anomalous_videos_to_chunks[line] = []
                    video_name = line.split('/')[1].split('.')[0]

                if 'Normal' in line:
                    for video_path in self.normal_chunks:
                        if video_name in video_path:
                            self.normal_videos_to_chunks[line].append(video_path)
                else:
                    for video_path in self.anomalous_chunks:
                        if video_name in video_path:
                            self.anomalous_videos_to_chunks[line].append(video_path)

    def __len__(self):
        return self.iterations

    def __getitem__(self, idx):
        if self.i == self.iterations:
            self.i = 0

        succ = False
        while not succ:
            try:
                feature, label, pnf = self._get_features()
                succ = True
            except Exception as e:
                index = np.random.choice(range(0, self.__len__()))
                # print("VideoIter:: ERROR!! (Force using another index:\n{})\n{}".format(index, e))

        self.i += 1
        out = {}
        out['features'] = feature
        out['label'] = label
        out['pure_nr_frames'] = pnf

        return out

    def _get_features(self):
        normal_keys = list(self.normal_videos_to_chunks)
        anomalous_keys = list(self.anomalous_videos_to_chunks)

        if len(self.used_anomal_keys) == len(normal_keys):
            self.used_normal_keys = []

        if len(self.used_anomal_keys) == len(anomalous_keys):
            self.used_anomal_keys = []

        normal_keys = list(set(normal_keys) - set(self.used_normal_keys))
        anomalous_keys = list(set(anomalous_keys) - set(self.used_normal_keys))

        normal_video = choice(normal_keys)
        anomalous_video = choice(anomalous_keys)

        self.used_normal_keys.append(normal_video)
        self.used_anomal_keys.append(anomalous_video)

        normal_paths = self.normal_videos_to_chunks[normal_video]
        anomalous_paths = self.anomalous_videos_to_chunks[anomalous_video]

        # print('Number of normal paths: %d' % len(normal_paths))
        # print('Number of anormal paths: %d' % len(anomalous_paths))

        normal_features = None
        if len(normal_paths) > 40:
            max_rand_idx = len(normal_paths) - 40
            start_idx = randint(0, max_rand_idx)
            normal_paths = normal_paths[start_idx:start_idx + 40]

        for normal_path in normal_paths:

            nf = self._load_video_chunk(normal_path)
            if normal_features is None:
                normal_features = nf
            else:
                normal_features = torch.cat([normal_features, nf], dim=0)

        num_segments = normal_features.shape[0] // self.frames_per_clip
        normal_features = normal_features[:num_segments * self.frames_per_clip]
        # normal_features = torch.stack(list(normal_features.split(16)), dim=0)
        # max_rand_idx = normal_features.shape[0] - 32
        # start_idx = randint(0, max_rand_idx)
        # normal_features = normal_features[start_idx:start_idx + 32]

        anomalous_features = None
        if len(anomalous_paths) > 40:
            raise
            max_rand_idx = len(anomalous_paths) - 30
            start_idx = randint(0, max_rand_idx)
            anomalous_paths = anomalous_paths[start_idx:start_idx + 30]
        for anomalous_path in anomalous_paths:

            af = self._load_video_chunk(anomalous_path)
            if anomalous_features is None:
                anomalous_features = af
            else:
                anomalous_features = torch.cat([anomalous_features, af], dim=0)

        num_segments = anomalous_features.shape[0] // self.frames_per_clip
        anomalous_features = anomalous_features[:num_segments * self.frames_per_clip]
        # anomalous_features = torch.stack(list(anomalous_features.split(16)), dim=0)
        # max_rand_idx = anomalous_features.shape[0] - 32
        # start_idx = randint(0, max_rand_idx)
        # anomalous_features = anomalous_features[start_idx:start_idx + 32]
        pure_nr_frames = torch.tensor([normal_features.shape[0], anomalous_features.shape[0]])
        if normal_features.shape[0] > anomalous_features.shape[0]:
            anomalous_features = self._zeropad(anomalous_features, normal_features.shape[0])
        elif normal_features.shape[0] < anomalous_features.shape[0]:
            normal_features = self._zeropad(normal_features, anomalous_features.shape[0])

        features = torch.stack([normal_features, anomalous_features], dim=0)
        labels = torch.tensor([0, 1])

        return features, labels, pure_nr_frames



    def _load_video_chunk(self, video_path):
        compressed_file = np.load(video_path)
        video = compressed_file['arr_0']
        video = torch.from_numpy(video)
        return video

    def _zeropad(self, tensor, size):
        n = size - tensor.shape[0] % size
        z = -torch.ones((n, tensor.shape[1]))
        return torch.cat((tensor, z), dim=0)

    def _load_videos_collate_fn(self, batch):
        max_pnf = max([max(x['pure_nr_frames']) for x in batch])

        # m_rem = math.ceil(max_pnf / self.segments_per_video)
        # comp_len = m_rem * self.segments_per_video
        # if comp_len > max_pnf:
        #     m_rem = comp_len - max_pnf
        # else:
        #     m_rem = 0
        # max_segment_len = (max_pnf + m_rem) / self.segments_per_video
        max_segment_len = math.floor(max_pnf / self.segments_per_video)

        # m_rem = max_pnf % self.segments_per_video
        # max_segment_len = (max_pnf + m_rem) / self.segments_per_video

        for sample in batch:
            video_couple = sample['features']
            pnf_couple = sample['pure_nr_frames']
            new_video_couple = []
            new_pnf_couple = []
            for video, pnf in zip(video_couple, pnf_couple):
                video_segments = []

                # rem_pnf = math.ceil(pnf / self.segments_per_video)
                # comp_len = rem_pnf * self.segments_per_video
                # if comp_len > pnf:
                #     rem_pnf = comp_len - pnf
                # else:
                #     rem_pnf = 0
                #
                # if pnf+rem_pnf <= video.shape[0]:
                #     video = video[:pnf+rem_pnf]
                # else:
                #     video = self._zeropad(video, int((video.shape[0]+rem_pnf).item()))
                segment_len = math.floor(pnf / self.segments_per_video)
                new_vide_len = segment_len * self.segments_per_video
                video = video[:pnf]
                if new_vide_len < pnf:
                    if random() < 0.5:
                        video = video[:new_vide_len]
                    else:
                        video = video[-new_vide_len:]
                rem_pnf = 0
                # segment_len = (pnf+rem_pnf) / self.segments_per_video
                segments = list(video.split(segment_len))
                if len(segments) == self.segments_per_video - 1:
                    print('here')
                # if segment_len-rem_pnf < 0:
                #     logger.debug('Error!!!!!!!!!!!!!')
                pnf_segments = torch.tensor([segment_len] * (self.segments_per_video-1)+ ([segment_len] if rem_pnf == 0 else [segment_len-rem_pnf]))

                for segment in segments:
                    if segment.shape[0] < max_segment_len:
                        video_segments.append(self._zeropad(segment, int(max_segment_len)))
                    else:
                        video_segments.append(segment)

                video_segments = torch.stack(video_segments, dim=0)

                new_video_couple.append(video_segments)
                new_pnf_couple.append(pnf_segments)

            sample['features'] = torch.stack(new_video_couple, dim=0)
            sample['pure_nr_frames'] = torch.stack(new_pnf_couple, dim=0)

        return default_collate(batch)


class CrimeFeatDatasetVal(Dataset):
    def __init__(self, segments_per_video):
        super(CrimeFeatDatasetVal, self).__init__()

        self.segments_per_video = segments_per_video
        self.testing_csv = pd.read_csv(
            '/BS/unintentional_actions/nobackup/ucf_crime/Temporal_Anomaly_Annotation_for_Testing_Videos.csv')

        base_anomaly_path = '/BS/unintentional_actions/nobackup/ucf_crime/vit_features/Anomaly_Videos-200'
        base_normal_path = '/BS/unintentional_actions/nobackup/ucf_crime/vit_features/Testing_Normal_Videos_Anomaly-200'

        self.anomalous_chunks = sorted(glob(os.path.join(base_anomaly_path, '**', '*.npz'), recursive=True))
        self.normal_chunks = sorted(glob(os.path.join(base_normal_path, '**', '*.npz'), recursive=True))

        # first get all the filenames for the training samples
        self.videos_to_chunks = {}
        # self.anomalous_videos_to_chunks = {}

        for idx, row in self.testing_csv.iterrows():
            line = row['filename']
            video_name = line.split('.')[0]

            if 'Normal' in line:
                self.videos_to_chunks[line] = []
                for video_path in self.normal_chunks:
                    if video_name in video_path:
                        self.videos_to_chunks[line].append(video_path)
            else:
                self.videos_to_chunks[line] = []
                for video_path in self.anomalous_chunks:
                    if video_name in video_path:
                        self.videos_to_chunks[line].append(video_path)

    def __len__(self):
        return len(list(self.videos_to_chunks))

    def __getitem__(self, idx):
        filename = list(self.videos_to_chunks.keys())[idx]
        video, pnf, vid_len = self._get_features(filename)
        pd_row = self.testing_csv.loc[self.testing_csv['filename']==filename]
        boundries_1 = [int(pd_row['sf1']), int(pd_row['ef1'])]
        boundries_2 = [int(pd_row['sf2']), int(pd_row['ef2'])]
        boundries = torch.tensor([boundries_1, boundries_2])

        return {'features': video, 'pure_nr_frames': pnf, 'boundries': boundries, 'vid_len': vid_len}



    def _get_features(self, filename):
        clip_paths = self.videos_to_chunks[filename]

        video = None
        for path in clip_paths:
            f = self._load_video_chunk(path)
            if video is None:
                video = f
            else:
                video = torch.cat([video, f], dim=0)

        video_segments = []
        pnf = video.shape[0]
        segment_len = math.floor(pnf / self.segments_per_video)
        # print('Cut %d frames' % (video.shape[0] - segment_len * self.segments_per_video))
        video = video[:segment_len * self.segments_per_video]
        # comp_len = rem_pnf * self.segments_per_video
        # if comp_len > pnf:
        #     rem_pnf = comp_len - pnf
        # else:
        #     rem_pnf = 0
        # vid_len = video.shape[0]
        # if pnf + rem_pnf <= video.shape[0]:
        #     video = video[:pnf + rem_pnf]
        # else:
        #     video = self._zeropad(video, int(video.shape[0] + rem_pnf))
        rem_pnf = 0

        segment_len = int((pnf + rem_pnf) / self.segments_per_video)
        segments = list(video.split(int(segment_len)))
        pnf_segments = torch.tensor([segment_len] * (self.segments_per_video-1) + ([segment_len] if rem_pnf == 0 else [segment_len - rem_pnf]))
        video_segments = torch.stack(segments, dim=0)

        return video_segments, pnf_segments, video.shape[0]


    def _load_video_chunk(self, video_path):
        compressed_file = np.load(video_path)
        video = compressed_file['arr_0']
        video = torch.from_numpy(video)
        return video

    def _zeropad(self, tensor, size):
        n = size - tensor.shape[0] % size
        z = -torch.ones((n, tensor.shape[1]))
        return torch.cat((tensor, z), dim=0)

if __name__ == '__main__':
    dset = CrimeFeatDatasetVal()
    loader = DataLoader(dset,
                        batch_size=2,
                        num_workers=32,
                        collate_fn=dset._load_videos_collate_fn)

    for i, data in enumerate(tqdm(loader)):
        pass

    # all_paths = '/BS/unintentional_actions/nobackup/ucf_crime/vit_features/Training-Normal-Videos-200'
    # all_paths = sorted(glob(os.path.join(all_paths, '**', '*.npz'), recursive=True))
    # max_nr_chunks = 0
    # max_chunk_path = ''
    # for path in all_paths:
    #     name = path.split('/')[-1].split('_')
    #     if len(name) == 4:
    #         name = name[-1]
    #         lp = name.split('.')
    #         chunk = lp[0]
    #         if int(chunk) > max_nr_chunks:
    #             max_nr_chunks = int(chunk)
    #             max_chunk_path = path
    # print(max_nr_chunks)
    # print(max_chunk_path)


