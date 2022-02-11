
# @Author: Enea Duka
# @Date: 10/31/21

import torch
from torch.utils.data import Dataset
from os import path
import json
import numpy as np
import av
from tqdm import tqdm

class SwinOopsFeats(Dataset):
    def __init__(self, mode):
        super(SwinOopsFeats, self).__init__()
        '/BS/unintentional_actions/work/data/oops/swin_feats/train_new/Getting Away From Me Fails (November 2017 _ FailArmy28.mp4.npz'
        self.feats = []
        self.mode = mode
        self.feat_path = path.join('/BS/unintentional_actions/work/data/oops/swin_feats', mode)
        self.video_path = path.join('/BS/unintentional_actions/nobackup/oops/oops_dataset/oops_video', mode)
        with open("/BS/unintentional_actions/nobackup/oops/oops_dataset/annotations/transition_times.json") as f:
            self.fails_borders = json.load(f)
        pass_videos = 0
        zeros = 0
        ones = 0
        twos = 0
        for entry in tqdm(list(self.fails_borders.keys())):
            entry_path = path.join(self.feat_path, entry) + '.mp4.npz'
            t_fail = self.fails_borders[entry]['t']
            t_fail = [x for x in t_fail if x != -1]
            if len(t_fail) > 0:
                if path.isfile(entry_path):
                    # load the features
                    pass_videos += 1
                    features_array = np.load(entry_path)
                    features_array = features_array['arr_0']
                    features = torch.from_numpy(features_array)

                    key = entry_path.split('/')[-1].replace('.mp4.npz', '')
                    borders = self.fails_borders[key]

                    video_path = path.join(self.video_path, key) + '.mp4'
                    video_tunit, original_fps, all_frames = self._get_video_t_unit(video_path)
                    if video_tunit is None:
                        continue
                    t_fail = borders['t']
                    t_fail = [x for x in t_fail if x != -1]
                    try:
                        t_fail = t_fail[len(t_fail) // 2]
                    except Exception as e:
                        print('here')
                    fail_frame = int(t_fail * original_fps)

                    labels = []
                    feat_list = []
                    for idx, feat_clip in enumerate(features):
                        video_clip_start_frame = idx * 32
                        # if video_clip_start_frame > all_frames:
                        #     break
                        video_clip_end_frame = video_clip_start_frame + 32

                        feat_list.append(feat_clip)
                        label = 0
                        if video_clip_end_frame < fail_frame:
                            label = 0
                            zeros += 1
                        elif video_clip_start_frame <= fail_frame <= video_clip_end_frame:
                            label = 1
                            ones += 1
                        elif fail_frame < video_clip_start_frame:
                            label = 2
                            twos += 1
                        else:
                            print('error_creating_label!!')
                            raise

                        out = {}

                        out['features'] = feat_clip
                        out['labels'] = torch.tensor(label)

                        self.feats.append(out)
        print(pass_videos)
        print(zeros)
        print(ones)
        print(twos)
    def __len__(self):
        return len(self.feats)

    def _get_video_t_unit(self, video_path):
        try:
            with av.open(video_path, metadata_errors='ignore') as container:
                vid_mdata = container.streams.video[0]
                return vid_mdata.time_base, vid_mdata.average_rate, vid_mdata.frames
        except Exception:
            return None, None, None


    def __getitem__(self, idx):
        return self.feats[idx]
        # feat_path = self.feat_paths[idx]
        #
        # # load the features
        # features_array = np.load(feat_path)
        # features_array = features_array['arr_0']
        # features = torch.from_numpy(features_array)
        #
        # key = feat_path.split('/')[-1].replace('.npz', '')
        # borders = self.fails_borders[key]
        #
        # video_path = path.join(self.video_path, key) + '.mp4'
        # video_tunit, original_fps, all_frames = self._get_video_t_unit(video_path)
        # t_fail = borders['t']
        # t_fail = [x for x in t_fail if x != -1]
        # try:
        #     t_fail = t_fail[len(t_fail) // 2]
        # except Exception as e:
        #     print('here')
        # fail_frame = int(t_fail * original_fps)
        #
        # labels = []
        # feat_list = []
        # for idx, feat_clip in enumerate(features):
        #     video_clip_start_frame = idx * 32
        #     if video_clip_start_frame > all_frames:
        #         break
        #     video_clip_end_frame = video_clip_start_frame + 32
        #
        #     feat_list.append(feat_clip)
        #
        #     if video_clip_end_frame < fail_frame:
        #         labels.append(0)
        #     elif video_clip_start_frame <= fail_frame <= video_clip_end_frame:
        #         labels.append(1)
        #     elif fail_frame < video_clip_start_frame:
        #         labels.append(2)
        #     else:
        #         print('error_creating_label!!')
        #         raise
        #
        # out = {}
        #
        # out['features'] = torch.stack(feat_list, dim=0)
        # out['labels'] = torch.tensor(labels)
        #
        # return out



