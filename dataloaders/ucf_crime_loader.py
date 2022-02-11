# @Author: Enea Duka
# @Date: 9/12/21

import sys
import warnings

sys.path.append('/BS/unintentional_actions/work/unintentional_actions')

import json
import os
import random
import statistics
from argparse import Namespace
from glob import glob

import av
import torch
import math
import numpy as np
import torch.utils.data as data
import torchvision
from torch.utils.data import ConcatDataset
from torchvision.datasets.video_utils import VideoClips
from torchvision.datasets.vision import VisionDataset
from tqdm import tqdm
from utils.arg_parse import opt
import pandas as pd
from torch.utils.data.dataloader import default_collate
import time

import utils.py12transforms as T
from utils.sampler import DistributedSampler, UniformClipSampler, RandomClipSampler, ConcatSampler


class UCFCrime(VisionDataset):

    def __init__(self, frames_per_clip, step_between_clips, fps, transform=None,
                 video_clips=None, val=False, clip_interval_factor=None, anticipate_label=0,
                 data_proportion=1, t_units=None, **kwargs):
        self.clip_len = frames_per_clip / fps
        self.clip_step = step_between_clips / fps
        self.clip_interval_factor = clip_interval_factor
        self.fps = fps
        self.t = transform
        self.video_clips = None
        self.anticipate_label = anticipate_label
        data_proportion = 1 if val else data_proportion
        self.base_features_path = '/BS/unintentional_actions/nobackup/ucf_crime/vit_features'
        self.csv = pd.read_csv('/BS/unintentional_actions/nobackup/ucf_crime/Temporal_Anomaly_Annotation_for_Testing_Videos.csv')
        self.train_filenames = '/BS/unintentional_actions/nobackup/ucf_crime/Anomaly_Train.txt'
        self.video_time_units = t_units
        self.chunk_size = 200
        self.val = val


        if video_clips:
            self.video_clips = video_clips # use the provided video clips
        else:
            # load the clips from storage
            # assert fails_path is None or fails_video_list is None
            if val:
                video_list = []
                for idx, row in self.csv.iterrows():
                    filename = row['filename']
                    action = row['action']

                    if action == 'Normal':
                        # continue
                        video_list.append(os.path.join('/BS/unintentional_actions/nobackup/ucf_crime/Testing_Normal_Videos_Anomaly', filename))
                    else:
                        video_list.append(os.path.join('/BS/unintentional_actions/nobackup/ucf_crime/Anomaly_Videos', action, filename))
            else:
                video_list = []
                with open(self.train_filenames) as f:
                    lines = f.read().splitlines()
                    for line in lines:
                        if 'Normal' in line:
                            video_list.append(os.path.join('/BS/unintentional_actions/nobackup/ucf_crime/Training-Normal-Videos', line))
                        # else:
                        #     video_list.append(os.path.join('/BS/unintentional_actions/nobackup/ucf_crime/Anomaly_Videos', line))
                    # base_video_path = '/BS/unintentional_actions/nobackup/ucf_crime/Training-Normal-Videos'
                    # video_list = sorted(glob(os.path.join(base_video_path, '**', '*.mp4'), recursive=True))

            self.video_clips = VideoClips(video_list, frames_per_clip, step_between_clips, fps, num_workers=16)

        # if not val:
        #     base_video_path = '/BS/unintentional_actions/nobackup/ucf_crime/Training-Normal-Videos'
        #     video_list = sorted(glob(os.path.join(base_video_path, '**', '*.mp4'), recursive=True))
        #
        #     self.name_to_frames = {}
        #     for video_path in video_list:
        #
        #

        # get start and end time of the clip
        self.t_from_clip_idx = lambda idx: (
            (step_between_clips * idx) / fps, (step_between_clips * idx + frames_per_clip) / fps)
        if val:
            if video_clips is None:
                self.video_clips.labels = []
                self.video_clips.compute_clips(frames_per_clip, step_between_clips, fps)
                self.video_time_units = []
                for video_idx, vid_clips in tqdm(enumerate(self.video_clips.clips), total=len(self.video_clips.clips)):
                    self.video_clips.labels.append([])
                    video_path = self.video_clips.video_paths[video_idx]
                    try:
                        container = av.open(video_path, metadata_errors='ignore')
                        t_unit = container.streams[0].time_base
                        frame_rate = container.streams[0].average_rate
                        self.video_time_units.append(t_unit)
                    except av.AVError as e:
                        print('Encountered av error...continuing')
                        pass
                    # t_fail = sorted(self.fails_data[os.path.splitext(os.path.basename(video_path))[0]]['t'])
                    filename = video_path.split('/')[-1]
                    row = self.csv.loc[self.csv['filename'] == filename]
                    sf1 = row['sf1'].tolist()[0]
                    ef1 = row['ef1'].tolist()[0]
                    sf2 = row['sf2'].tolist()[0]
                    ef2 = row['ef2'].tolist()[0]
                    for clip_idx, clip in enumerate(vid_clips):
                        start_pts = clip[0].item()
                        end_pts = clip[-1].item()
                        t_start = float(t_unit * start_pts)
                        t_end = float(t_unit * end_pts)
                        f_start = t_start * frame_rate
                        f_end = t_end * frame_rate
                        label = 0
                        if sf1 <= f_start <= f_end <= ef1 or sf2 <= f_start <= f_end <= ef2:
                            label = 2
                        if f_start <= sf1 <= f_end <= ef1 or sf1 <= f_start <= ef1 <= f_end \
                            or f_start <= sf2 <= f_end <= ef2 or sf2 <= f_start <= ef2 <= f_end:
                            label = 1
                        # if sf2 != -1 and ef2 != -1:

                        self.video_clips.labels[-1].append(label)
                clip_lengths = torch.as_tensor([len(v) for v in self.video_clips.clips])
                self.video_clips.cumulative_sizes = clip_lengths.cumsum(0).tolist()
                # labels_m1 = 0
                # labels_0 = 0
                # labels_1 = 0
                # labels_2 = 0
                # for label in self.video_clips.labels:
                #     if len(label) != 0:
                #         if label[0] == -1:
                #             labels_m1 += 1
                #         if label[0] == 0:
                #             labels_0 += 1
                #         if label[0] == 1:
                #             labels_1 += 1
                #         if label[0] == 2:
                #             labels_2 += 1
                # print(labels_m1)
                # print(labels_0)
                # print(labels_1)
                # print(labels_2)

            # for i, p in enumerate(self.video_clips.video_paths):
            #     self.video_clips.video_paths[i] = p.replace("PATH/TO/scenes",
            #                                                 os.path.dirname(fails_path))
            # self.debug_dataset = debug_dataset
            # self.sampled_clips = [False] * len(self.video_clips.clips)

    def __len__(self):
        return self.video_clips.num_clips()

    def compute_clip_times(self, video_idx, clip_idx):
        video_path = self.video_clips.video_paths[video_idx]
        video_path = os.path.join(self.fails_path, os.path.sep.join(video_path.rsplit(os.path.sep, 2)[-2:]))
        clip_pts = self.video_clips.clips[video_idx][clip_idx]
        start_pts = clip_pts[0].item()
        end_pts = clip_pts[-1].item()
        t_unit = av.open(video_path, metadata_errors='ignore').streams[0].time_base
        t_start = float(t_unit * start_pts)
        t_end = float(t_unit * end_pts)
        return t_start, t_end

    def _get_video_t_unit(self, video_idx):
        video_path = self.video_clips.video_paths[video_idx]
        # video_path = os.path.join(self.fails_path, os.path.sep.join(video_path.rsplit(os.path.sep, 2)[-2:]))
        with av.open(video_path, metadata_errors='ignore') as container:
            return container.streams.video[0].time_base, container.streams.video[0].average_rate


    def _compute_clip_times(self, video_idx, start_pts, end_pts, terminal_pts, offset):
        try:
            video_t_unit, original_fps = self._get_video_t_unit(video_idx)
            original_fps = float(original_fps)
            video_fps = self.fps # self.video_clips.video_fps[video_idx]
            fpc = self.fps * self.clip_len
            start_time = start_pts * float(video_t_unit)
            end_time = end_pts * float(video_t_unit)
            terminal_time = float(terminal_pts * video_t_unit)
            start_frame = int(start_time * float(original_fps)) - offset
            end_frame = int(math.ceil(end_time * float(original_fps))) - offset
            terminal_frame = int(terminal_time * video_fps)
            # if end_frame - start_frame < fpc:
            #     new_end_frame = end_frame + int(fpc - (end_frame-start_frame))
            #     if new_end_frame > terminal_frame:
            #         start_frame -= int(fpc - (end_frame-start_frame))
            #     else:
            #         end_frame = new_end_frame
            # if (end_frame - start_frame) < fpc:
            #     print('here')
            # end_frame = int(start_frame + (self.clip_len * self.fps))
            # if ret == 'frames':
            return start_frame, end_frame, original_fps, start_time, end_time
            # elif ret == 'times':
                # return start_time, end_time, original_fps
        except Exception:
            print('here')

    def _get_chunked_video_path(self, pts_start, pts_end, video_path, video_idx):
        time_unit, original_fps = self._get_video_t_unit(video_idx)
        # convert clip boundries from pts to frames
        t_start = float(time_unit * pts_start)
        t_end = float(time_unit * pts_end)
        f_start = t_start * original_fps
        f_end = t_end * original_fps

        start_chunk = int(f_start // self.chunk_size)
        end_chunk = int(f_end // self.chunk_size)

        path_elms = video_path.split('/')
        name_elms = path_elms[-1].split('_')
        last_part_elms = name_elms[-1].split('.')
        last_part = last_part_elms[0] + '_' + str(start_chunk) + '.mp4' + '.npz'
        new_name = '_'.join(name_elms[:-1]) + '_' + last_part
        new_path = '/'.join(path_elms[:-1]) + '/' + new_name

        if start_chunk == end_chunk:
            return new_path, None
        else:
            # the clip is between two chunks
            # for simplicity we do not use these clips
            return new_path, self.chunk_size - 1

    def _filter_coll_fn(self, batch):
        new_batch = [s for s in batch if s['times'][0] != -1]
        try:
            return default_collate(new_batch)
        except Exception:
            print('here')

    def _read_video_features(self, video_path, video_idx, clip_idx,  start_pts, end_pts, terminal_pts):
        new_end_frame = None
        if not os.path.isfile(video_path):
            video_path, new_end_frame = self._get_chunked_video_path(start_pts, end_pts, video_path, video_idx)
            # if video_path is None:
            #     video_path = self._get_chunked_video_path(start_pts, end_pts, video_path, video_idx)
            #     return torch.empty((16, 768)), -1, -1

        video_name = video_path.split('/')[-1]
        name_parts = video_name.split('_')
        offset = 0
        if (self.val and len(name_parts) > 2) or (not self.val and (len(name_parts) == 4)):
        # if (len(name_parts) > 2) if self.val else (len(name_parts) == 4):
            chunk = int(name_parts[-1].split('.')[0])
            offset = chunk * self.chunk_size
        start_time = time.time()
        compressed_file = np.load(video_path)
        array = compressed_file['arr_0']
        if np.isnan(array).any():
            print('Nan in numpy array')
        video = torch.from_numpy(array)
        end_time = time.time()
        # print(end_time - start_time)
        start_frame, end_frame, original_fps, start_time, end_time = self._compute_clip_times(video_idx, start_pts, end_pts, terminal_pts, offset)
        if new_end_frame is not None:
            end_frame = new_end_frame
        if end_frame >= video.shape[0]:
            # print('corrected boundries')
            to = end_frame - video.shape[0]
            start_frame -= to
            end_frame -= to
        new_video = video[start_frame:end_frame] # since torch slicing has exclusive target
        if new_video.shape[0] < 16:
            remaining = 16 - new_video.shape[0]
            if (end_frame + remaining) < (self.chunk_size - 1):
                trtn_video = video[start_frame:end_frame+remaining]
            else:
                trtn_video = video[start_frame-remaining:end_frame]
        else:
            trtn_video = new_video

        if trtn_video.shape[0] == 0:
            print('here')

        # if new_video.shape[0] < 16:
        #     remaining = 16 - new_video.shape[0]
        #     try:
        #         trtn_video = video[start_frame:end_frame+remaining]
        #     except IndexError:
        #         trtn_video = video[start_frame-remaining:end_frame]
        # else:
        #     trtn_video = new_video
        # if torch.isnan(new_video).any():
        #     print('Found Nan!!!')
        array = trtn_video.cpu().numpy()
        return array, start_time, end_time

    def _resample_video_idx(self, num_frames, original_fps, new_fps):
        step = float(original_fps) / new_fps
        if step.is_integer():
            # optimization: if step is integer, don't need to perform
            # advanced indexing
            step = int(step)
            return slice(None, None, step)
        idxs = torch.arange(num_frames, dtype=torch.float32) * step
        idxs = idxs.floor().to(torch.int64)
        return idxs

    def _get_clip(self, clip_idx):

        if clip_idx >= self.video_clips.num_clips():
            raise IndexError(
                "Index {} out of range "
                "({} number of clips)".format(idx, self.video_clips.num_clips())
            )
        video_idx, clip_idx = self.video_clips.get_clip_location(clip_idx)
        video_path = self.video_clips.video_paths[video_idx]
        video_path = video_path + '.npz'
        elms = video_path.split('/')
        if len(elms) == 7:
            video_path = '/'.join(elms[:5]) + '/vit_features/' + elms[5] + '-200/' + '/'.join(elms[6:])
        else:
            video_path = '/'.join(elms[:5]) + '/vit_features/' + elms[5] + '-200/' + elms[7]
        # video_path = os.path.join(self.base_features_path, video_name)
        clip_pts = self.video_clips.clips[video_idx][clip_idx]

        start_pts = clip_pts[0].item()
        end_pts = clip_pts[-1].item()
        terminal_pts = self.video_clips.clips[video_idx][-1][-1].item()
        video_clip, start_time, end_time = self._read_video_features(video_path, video_idx, clip_idx, start_pts, end_pts, terminal_pts)
        # if torch.isnan(video_clip).any():
        #     print('Found Nan!!!')
        return video_clip, start_time, end_time

    def __getitem__(self, idx):

        video_idx, clip_idx = self.video_clips.get_clip_location(idx)
        video, start_time, end_time = self._get_clip(idx)

        if video.shape[0] == 0:
            print('Found video 0')
            video = None
        video_path = self.video_clips.video_paths[video_idx]
        # print(video_path)
        try:
            # labels = self.video_clips.labels[video_idx]
            # if len(labels) == 0:
            #     print('here')
            label = self.video_clips.labels[video_idx][clip_idx]
            if self.anticipate_label:
                video_path = self.video_clips.video_paths[video_idx]
                t_fail = statistics.median(self.fails_data[os.path.splitext(os.path.basename(video_path))[0]]['t'])
                # t_start, t_end = self.compute_clip_times(video_idx, clip_idx)% ('test' if val else 'train')
                t_start = start_time
                t_end = end_time
                t_start += self.anticipate_label
                t_end += self.anticipate_label
                label = 0
                if t_start <= t_fail <= t_end:
                    label = 1
                elif t_start > t_fail:
                    label = 2
        except:
            label = -1

        out = {}
        video = torch.from_numpy(video)
        out['features'] = video
        out['pure_nr_frames'] = int(self.clip_len * self.fps)
        out['label'] = label
        # if self.get_clip_times:
        out['times'] = (start_time, end_time)
        video_name = video_path.split('/')[-1].replace('.mp4', '')
        # t_time = self.fails_data[video_name]['t']
        # out['t'] = t_time
        # out['rel_t'] = self.fails_data[video_name]['rel_t']
        out['video_idx'] = video_idx
        out['video_name'] = video_name
        return out
        # return video, label, (video_path, t_start, t_end, *other)

def get_crime_video_loader(args):
    # args = Namespace(**kwargs)
    args.fails_video_list = None
    if args.val:
        args.fails_path = os.path.join(args.fails_path, 'val')
        args.kinetics_path = os.path.join(args.kinetics_path, 'val')
    else:
        args.fails_path = os.path.join(args.fails_path, 'train')
        # args.kinetics_path = os.path.join(args.kinetics_path, 'train')
    # if args.fails_action_split:
    #     args.fails_path = None
    #     args.fails_video_list = torch.load(os.path.join(args.dataset_path, 'fails_action_split.pth'))[
    #         'val' if args.val else 'train']
    DEBUG = False
    datasets = []
    samplers = []
    for fps in args.fps_list:
        clips = None
        t_units = None
        args.fps = fps
        args.step_between_clips = round(args.step_between_clips_sec * fps)
        cache_path = os.path.join(args.dataset_path,
                                  '{0}-{1}-{2}_videoclips.pth'.format('val' if args.val else 'train', f'{args.fps}fps', f'{args.step_between_clips_sec}'))

        t_units_cache_path = os.path.join(args.dataset_path, 'time_units_%s.pth' % ('val' if args.val else 'train'))
        if args.cache_dataset and os.path.exists(cache_path):
            clips = torch.load(cache_path)
            t_units = torch.load(t_units_cache_path)
            if args.local_rank <= 0:
                print(f'Loaded dataset from {cache_path}')

        # args.transform = test_transform if args.val else train_transform

        dataset = UCFCrime(video_clips=clips, t_units=t_units, **vars(args))
        # if not args.val:
        print(f'Dataset contains {len(dataset)} items')
        if args.cache_dataset and args.local_rank <= 0 and clips is None:  # and not args.fails_only
            torch.save(dataset.video_clips, cache_path)
            torch.save(dataset.video_time_units, t_units_cache_path)
        if args.val:
            sampler = UniformClipSampler(dataset.video_clips,
                                         1000000 if args.sample_all_clips else args.clips_per_video)
        else:
            sampler = RandomClipSampler(dataset.video_clips, 1000000 if args.sample_all_clips else args.clips_per_video)
        datasets.append(dataset)
        samplers.append(sampler)
    if len(args.fps_list) > 1:
        dataset = ConcatDataset(datasets)
        sampler = ConcatSampler(samplers)
    else:
        dataset = datasets[0]
        sampler = samplers[0]
    if args.local_rank != -1:
        sampler = DistributedSampler(sampler)
    return data.DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        shuffle=False,
        # collate_fn=dataset.collate_fn,
        # sampler=sampler,
        pin_memory=True,
        drop_last=False
    )

if __name__ == '__main__':
    opt.dataset_path = '/BS/unintentional_actions/nobackup/ucf_crime'
    opt.val = False
    opt.frames_per_clip = 16
    opt.step_between_clips_sec = 0.25
    opt.fps_list = [30]
    opt.workers = 32
    opt.batch_size = 64
    video_loader = get_crime_video_loader(opt)
    for idx, data in enumerate(tqdm(video_loader)):
        videos = data['features']
        # print(data['label'])
        if torch.isnan(videos).any():
            print('Found Nan!!!')
        pass