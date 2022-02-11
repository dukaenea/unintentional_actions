
# @Author: Enea Duka
# @Date: 8/10/21

from glob import glob

import os
import bisect
import torch
import numpy as np
import torch.nn.functional as F
import scipy.io as scio
import torchvision
from torchvision.datasets.video_utils import VideoClips
from torchvision.datasets.vision import VisionDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from random import random, randint
from torch.utils.data.dataloader import default_collate
import math
from rep_learning.feat_transformer import fast_forward
import ffmpeg
import av
import utils.py12transforms as T
import glob
import cv2
import kornia as K
from dataloaders.dl_utils import Normalize
from dataloaders.dl_utils import tensor_to_zero_one
from pprint import pprint



normalize = T.Normalize(mean=[0.5, 0.5, 0.5],
                        std=[0.5, 0.5, 0.5])
unnormalize = T.Unnormalize(mean=[0.5, 0.5, 0.5],
                            std=[0.5, 0.5, 0.5])
train_transform = torchvision.transforms.Compose([
    T.ToFloatTensorInZeroOne(),
    T.Resize((128 * 2, 171 * 2)),
    T.RandomHorizontalFlip(),
    T.RandomRotate(),
    normalize,
    T.RandomCrop((224, 224))
])
test_transform = torchvision.transforms.Compose([
    T.ToFloatTensorInZeroOne(),
    T.Resize(230),
    normalize,
    T.CenterCrop((224, 224))
])



class AnomalyDataset():
    def __init__(self, videos_path, frames_per_clip, step_between_clips, fps,
                 val=False, clip_interval_factor=None, video_clips=None, dataset=None,
                 load_videos=False, load_frames=False):

        self.clip_len = frames_per_clip / fps
        self.clip_step = int(step_between_clips * fps)
        self.clip_interval_factor = clip_interval_factor
        self.fps = fps
        self.dataset = dataset
        self.val = val
        self.load_videos = load_videos
        self.load_frames = load_frames
        self.speeds = [0, 1, 2, 3]
        self.fpc = frames_per_clip
        self.norm = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])


        self.video_clips = None
        self.videos_path = videos_path

        if video_clips:
            self.video_clips = video_clips
        else:
            assert videos_path is not None
            if 'pedestrian' in self.dataset:
                video_list = sorted(glob.glob(os.path.join(videos_path, '**', '*.npz'), recursive=True))
                if dataset == 'pedestrian1':
                    self.video_list = [v_name for v_name in video_list if '1' in v_name.split('_')[-1]]
                if dataset == 'pedestrian2':
                    self.video_list = [v_name for v_name in video_list if '2' in v_name.split('_')[-1]]
            else:
                if self.load_frames:
                    data_folders = '/BS/unintentional_actions/nobackup/avenue/avenue/%sing' % ('test' if self.val else 'train')
                    self.video_list = []
                    for folder in sorted(glob.glob(f'{data_folders}/*')):
                        all_imgs = glob.glob(f'{folder}/*.jpg')
                        all_imgs.sort()
                        self.video_list.append(all_imgs)
                else:
                    self.video_list = sorted(glob.glob(os.path.join(videos_path, '**', ('*.avi' if self.load_frames else '*.npz')), recursive=True))
                print("Loaded %d videos for the Avenue dataset." % len(self.video_list))
            self.video_clips, self.cumulative_sizes = self._calculate_clips(self.video_list)

        #  we have labels only for the test split of the dataset
        if val:
            self.labels = []
            labels = []
            if self.dataset == 'avenue':
                label_loader = AvenueLabelLoader(
                    sorted([x[0] for x in os.walk('/BS/unintentional_actions/nobackup/avenue/avenue/testing')][1:])
                )
                self.init_labels = label_loader.load_ucsd_avenue()

                for label in self.init_labels:
                    num_frames = int(len(label) * self.fps / 25)
                    step = 25 / self.fps
                    if step.is_integer():
                        step = int(step)
                        idxs = slice(None, None, step)
                    else:
                        idxs = torch.arange(num_frames, dtype=torch.float32) * step
                        idxs = idxs.floor().to(torch.int64)
                    labels.append(torch.tensor(label))
                for idx, clip in enumerate(self.video_clips):
                    video_idx, _ = self._get_clip_location(idx)
                    label = labels[video_idx]
                    try:
                        sampled_label = label[clip]
                    except Exception as e:
                        raise
                    self.labels.append(sampled_label[-1])
                    # zeros = (sampled_label == 0).sum(dim=0)
                    # if zeros == sampled_label.shape[0]:
                    #     self.labels.append(0)
                    # else:
                    #     self.labels.append(1)
                    # else:
                    #     self.labels.append(2)
                    #     # self.labels.append(1)
            if 'pedestrian' in self.dataset:
                with open(
                        '/BS/unintentional_actions/nobackup/ucsd/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Test/labels.txt') as f:
                    p1_labels = [line.rstrip() for line in f]
                with open(
                        '/BS/unintentional_actions/nobackup/ucsd/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Test/labels.txt') as f:
                    p2_labels = [line.rstrip() for line in f]
                count = 0
                if self.dataset == 'pedestrian1':
                    labels = p1_labels
                elif self.dataset == 'pedestrian2':
                    labels = p2_labels
                else:
                    while count < len(p2_labels):
                        labels.append(p1_labels[count])
                        labels.append(p2_labels[count])
                        count += 1
                    labels = labels + p1_labels[len(p2_labels):]
                # self.labels = p1_labels
                for idx, clip in enumerate(self.video_clips):
                    # if idx == 1557:
                    #     print('Halt!')
                    video_idx, _ = self._get_clip_location(idx)
                    try:
                        if ',' in labels[video_idx]:
                            segments = labels[video_idx].split(',')
                        else:
                            segments = [labels[video_idx]]
                        appended = False
                        for segment in segments:
                            segment = segment.split(':')
                            # if int(segment[0]) <= clip[-1] <= int(segment[1]):
                            #     self.labels.append(1)
                            #     appended = True
                            # if (clip[-1] - int(segment[0]) > 0 and int(segment[0]) > clip[0]) or\
                            #     (int(segment[1]) - clip[0] > 0 and int(segment[1]) < clip[-1]) or\
                            #     (clip[0] >= int(segment[0]) and clip[-1] <= int(segment[1])):
                            if int(segment[0]) <= clip[-1] <= int(segment[1]):
                                self.labels.append(1)
                                appended = True
                                if video_idx == 0:
                                    cpv = self.cumulative_sizes[0]
                                else:
                                    cpv = self.cumulative_sizes[video_idx] - self.cumulative_sizes[video_idx - 1]
                                print('%d ---- %s --- %d --- 1' % (clip[-1], '--'.join(segment), cpv))
                        if not appended:
                            self.labels.append(0)
                            if video_idx == 0:
                                cpv = self.cumulative_sizes[0]
                            else:
                                end = self.cumulative_sizes[video_idx]
                                start = self.cumulative_sizes[video_idx - 1]
                                cpv = end - start
                            print('%d ---- %s --- %d --- 0' % (clip[-1], segments, cpv))
                    except Exception as e:
                        print(e)
                        raise
            zeros = self.labels.count(0)
            ones  = self.labels.count(1)
            twos  = self.labels.count(2)
            print(zeros)
            print(ones)
            print(twos)
        if load_frames and dataset=='avenue':
            self._load_videos()


    def __len__(self):
        return len(self.video_list) if self.load_videos else len(self.video_clips)

    def __getitem__(self, idx):
        if self.load_videos:
            out={}
            video = self._load_video(idx)
            out['features'] = video
            out['pure_nr_frames'] = video.shape[0]
            if self.val:
                label = self.init_labels[idx]
                out['label'] = label
        else:
            video_idx, clip_idx = self._get_clip_location(idx)
            video_clip, next_frame = self._load_clip_frames(video_idx, idx) if self.load_frames else self._load_clip(video_idx, idx)
            out = {}
            out['features'] = video_clip
            if self.load_frames:
                video_name = self.video_list[video_idx][0].split('/')[-2]
            else:
                video_name = self.video_list[video_idx].split('/')[-1]
            out['video_name'] = video_name
            out['pure_nr_frames'] = out['features'].shape[0]
            out['next_frame'] = next_frame
            if self.val:
                label = self.labels[idx]
                out['label'] = label
        return out

    def _get_centers(self, tensor, n_frames):
        w = tensor.shape[2]
        h = tensor.shape[3]
        return torch.FloatTensor([w // 2, h // 2]).repeat(n_frames, 1)

    def rotate(self, tensor, rot_angle, p=1.0):
        # tensor = tensor.permute(1, 0, 2, 3)
        f, c, w, h = tensor.shape
        # tensor = tensor.reshape(f*c, w, h)
        rnd = torch.rand((1,))
        rot_angle_tensor = ((rnd * rot_angle * 2) - rnd * rot_angle).repeat(f, 1).squeeze()
        center_tensor = self._get_centers(tensor, f)
        tensor = K.geometry.transform.rotate(tensor, rot_angle_tensor, center_tensor)
        # tensor = tensor.permute(1, 0, 2, 3)
        return tensor

    def rotate_det(self, tensor, rot_angle, p=1.0):
        # tensor = tensor.permute(1, 0, 2, 3)
        f, c, w, h = tensor.shape
        # tensor = tensor.reshape(f*c, w, h)
        rot_angle_tensor = torch.tensor(rot_angle).repeat(f, 1).squeeze()
        center_tensor = self._get_centers(tensor, f)
        tensor = K.geometry.transform.rotate(tensor, rot_angle_tensor, center_tensor)
        # tensor = tensor.permute(1, 0, 2, 3)
        return tensor

    def _load_video_from_frame_folder(self, video_folder):
        def np_load_frame(filename, resize_h, resize_w, gray_scale=False):
            img = cv2.imread(filename)
            if gray_scale:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = cv2.resize(img, (resize_w, resize_h)).astype('float32')
            # image_resized = (image_resized / 127.5) - 1.0  # to -1 ~ 1
            if not gray_scale:
                img = np.transpose(img, [2, 0, 1])  # to (C, W, H)
            return img

        video = []
        for image_path in video_folder:
            frame = np_load_frame(image_path, 224, 224)
            video.append(frame)
        video = np.array(video).reshape((-1, 3, 224, 224))
        video = torch.from_numpy(video)
        video = tensor_to_zero_one(video)
        video = self.norm(video)
        #video = test_transform(video)
        # video = video.permute(1, 0, 2, 3)
        return video

    def _load_videos(self):
        videos = []
        for video_folder in tqdm(self.video_list):
            videos.append(self._load_video_from_frame_folder(video_folder))

        self.videos = videos

    def _get_video_dim(self, video_path):
        try:
            probe = ffmpeg.probe(video_path)
            video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
            return int(video_stream['height']), int(video_stream['width'])
        except Exception as e:
            print(e)
            print(video_path)
            return None, None

    def _get_raw_video(self, video_path):
        height, width = self._get_video_dim(video_path)
        if width is None and height is None:
            return None
        container = av.open(video_path)
        cmd = (ffmpeg.input(video_path))
        frames = container.streams.video[0].average_rate
        video_fps = 25    # av.open(video_path).streams.video[0].average_rate
        cmd = cmd.filter('fps', fps=video_fps)

        out, _ = (
            cmd.output('pipe:', format='rawvideo', pix_fmt='rgb24')
                .run(capture_stdout=True, quiet=True)
        )

        video = np.frombuffer(out, np.uint8).reshape([-1, width, height, 3])
        video = torch.from_numpy(video)

        return video

    def _load_clip_frames(self, video_idx, clip_idx):
        try:
            video = self.videos[video_idx]
        except Exception:
            video = self._load_video_from_frame_folder(self.video_list[video_idx])
        # video = video.permute(1, 0, 2, 3)
        # video = self._make_16_fps(video)
        clip = self.video_clips[clip_idx]
        next_frame = video[clip[-1]]
        return video[clip], next_frame

    def _random_crop_vector(self, vector, crop_size):
        try:
            max_start_pos = (vector.shape[0] - (crop_size + 2))
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
        inds_foba = torch.cat([idxs[:-1], torch.flip(idxs, dims=(0,))], 0)
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
        # vec_len = vector.shape[0]
        # min_start_idx = (vec_len // 2) - (crop_len - 2)
        # max_start_idx = (vec_len // 2) - 2

        start_idx = randint(crop_len//2, crop_len//2 + crop_len + 1)
        return vector[crop_len//2: crop_len//2 + crop_len + 1]

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

    def _rot_lrn_collate_fn(self, batch):
        new_batch = []
        for _ in range(1):
            for idx, data in enumerate(batch):
                video = data['features']
                video = self._crop_video_temporally(video, 20)

                nine_video = torch.rot90(video, 1, [2, 3])
                oeight_video = torch.rot90(nine_video, 1, [2, 3])
                tseven_video = torch.rot90(oeight_video, 1, [2, 3])

                frames = torch.cat([video, nine_video, oeight_video, tseven_video], dim=0)
                labels = torch.tensor([0]*20 + [1]*20 + [2]*20 + [3]*20)

                perm_idxs = torch.randperm(20*4)
                frames = frames[perm_idxs]
                labels = labels[perm_idxs]

                for i in range(frames.shape[0]):
                    new_batch.append({'features ': frames[i], 'label': labels[i], 'pure_nr_frames': 1})

        return default_collate(new_batch)

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
                if video.shape[0] // 4 > min_seq_len:
                    subsample_idxs = torch.arange(0, video.shape[0], step=4)
                    video = video[subsample_idxs]
                p_speed = 1
                p_motion = 1
                # dset = 'oops'
                dset = 'avenue'
                if video.shape[0] > min_seq_len:

                    # first modify the speed of the video
                    s_video, speed_label, eff_skip, max_speed = self._speed_up_video(video, min_seq_len)
                    nrm_cropped_video = self._crop_video_temporally(video, min_seq_len)

                    new_batch.append({'features': s_video, 'label': speed_label, 'pure_nr_frames': s_video.shape[0],
                                      'org_vid_idx': -1, 'dset': dset})

                    new_batch.append(
                        {'features': nrm_cropped_video, 'label': 0, 'pure_nr_frames': nrm_cropped_video.shape[0],
                         'org_vid_idx': -1, 'dset': dset})
                    # else:
                    # if trn == 4:
                    sh_video = self._shuffle_video(video, min_seq_len)
                    # if random() < p_motion:
                    new_batch.append({'features': sh_video, 'label': 7, 'pure_nr_frames': sh_video.shape[0],
                                      'org_vid_idx': -1, 'dset': dset})

                    # then do e foba
                    # if trn == 5:
                    foba_video = self._foba_video(video, eff_skip, min_seq_len)
                    # if random() < p_motion:
                    new_batch.append({'features': foba_video, 'label': 4, 'pure_nr_frames': foba_video.shape[0],
                                      'org_vid_idx': -1, 'dset': dset})

                    # if trn == 6:
                    stitched_video = self._stitch_videos(nrm_cropped_video, s_video, min_seq_len)
                    new_batch.append({'features': stitched_video, 'label': 6, 'pure_nr_frames': stitched_video.shape[0],
                                      'org_vid_idx':  -1, 'dset': dset})
                        # and at the end warp the time
                    # if trn == 7:
                    twarp_video = self._twarp_video(video, max_speed, min_seq_len)
                    # if random() < p_motion:
                    new_batch.append({'features': twarp_video, 'label': 5, 'pure_nr_frames': twarp_video.shape[0],
                                      'org_vid_idx': -1, 'dset': dset})

                    #     ridx = randint(0, 3)
                    # new_batch.append(nn_videos[ridx])

        if len(new_batch) > 0:
            max_len = max([s['features'].shape[0] for s in new_batch])
            for data in new_batch:
                if data['features'].shape[0] < max_len:
                    data['features'] = _zeropad(data['features'], max_len)
        try:
            return default_collate(new_batch)
        except Exception:
            return

    def _load_video(self, video_idx):
        if self.load_frames:
            # video = self.videos[video_idx]
            video = self._load_video_from_frame_folder(self.video_list[video_idx])
            # if not self.val:
            #     if random() < 0.5:
            #         video = torch.flip(video, dims=(-1,))
            #     if random() < 0.5:
            #         video = self.rotate(video, 30)
        else:
            video_path = self.video_list[video_idx]
            compressed_file = np.load(video_path)
            video = compressed_file['arr_0']
            video = torch.from_numpy(video)
        return video

    def _load_clip(self, video_idx, clip_idx):
        video_path = self.video_list[video_idx]
        compressed_file = np.load(video_path)
        video = compressed_file['arr_0']
        video = torch.from_numpy(video)
        # video = self._make_16_fps(video)
        clip  = self.video_clips[clip_idx]
        next_frame =  video[clip[-1]]
        return video[clip], next_frame

    def _calculate_clips(self, video_list):
        # iterate over all the videos
        all_clips = []
        cum_sum = []
        for video_path in tqdm(video_list):
            # clips = []
            start = 0 if self.fpc > 1 else 31
            end = start + self.fpc
            clip_count = 0
            # first load the video from the .npz file
            if self.load_frames:
                # video = self._get_raw_video(video_path)
                video = self._load_video_from_frame_folder(video_path)
            else:
                compressed_file = np.load(video_path)
                video = compressed_file['arr_0']
                video = torch.from_numpy(video)
            # video = self._make_16_fps(video)
            while end <= video.shape[0]-self.clip_step:
                all_clips.append(torch.arange(start, end))
                start += self.clip_step
                end += self.clip_step
                clip_count += 1
            if len(cum_sum) == 0:
                cum_sum.append(clip_count)
            else:
                cum_sum.append(cum_sum[-1] + clip_count)

        return all_clips, cum_sum

    def _get_clip_location(self, idx):
        video_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if video_idx == 0:
            clip_idx = idx
        else:
            clip_idx = idx - self.cumulative_sizes[video_idx - 1]
        return video_idx, clip_idx


    def _make_16_fps(self, video):
        # depending on the dataset we might need to sub or over sample in time
        if self.dataset == 'avenue':
            # we need to subsample from 25 to 16 fps
            num_frames = int(video.shape[0] * self.fps / 25)
            video_idxs = self._resample_video_idx(num_frames, 25, self.fps)
            video = video[video_idxs]
        elif self.dataset == 'pedestrian':
            # we need to upsample from 10 to 16
            num_frames = int(video.shape[0] * self.fps / 10)
            # take the temporal axis to the end so we can interpolate
            video = video.permute(1, 0)
            video = video.unsqueeze(0)
            video = F.interpolate(video, size=num_frames)
            video = video.squeeze()
            video = video.permute(1, 0)

        return video


    def _resample_video_idx(self, num_frames, original_fps, new_fps):
        step = original_fps / new_fps
        if step.is_integer():
            step = int(step)
            return slice(None, None, step)
        idxs = torch.arange(num_frames, dtype=torch.float32) * step
        idxs = idxs.floor().to(torch.int64)
        return idxs


class AvenueLabelLoader:
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


def get_anomaly_loader(dataset, frames_per_clip, step_between_clips, fps,
                        val=False, clip_interval_factor=None, video_clips=None,
                       load_videos=False, load_frames=False):

    videos_path = None
    if dataset == 'avenue':
        if load_frames:
            videos_path = '/BS/unintentional_actions/nobackup/avenue/Avenue Dataset/%sing_videos' % ('test' if val else 'train')
        else:
            videos_path = '/BS/unintentional_actions/nobackup/avenue/avenue/vit_features/%s' % ('test' if val else 'train')
    if 'pedestrian' in dataset:
        videos_path = '/BS/unintentional_actions/nobackup/ucsd/UCSD_Anomaly_Dataset.v1p2/vit_features/%s' % ('Test' if val else 'Train')


    dataset = AnomalyDataset(videos_path, frames_per_clip,
                             step_between_clips, fps, val=val,
                             dataset=dataset, load_videos=load_videos,
                             load_frames=load_frames)

    # return dataset

    if val:
        return DataLoader(dataset,
                          num_workers=32,
                          batch_size=4 if load_videos else 32,
                          shuffle=False,
                          drop_last=False,
                          # collate_fn=dataset._rep_lrn_collate_fn,
                          pin_memory=False)
    else:
        return DataLoader(dataset,
                          num_workers=32,
                          batch_size=4 if load_videos else 32,
                          shuffle=True,
                          drop_last=False,
                          # collate_fn=dataset._rep_lrn_collate_fn,
                          pin_memory=False)

if __name__ == '__main__':
    loader = get_anomaly_loader('avenue', 16, 0.25, 16, load_videos=False, val=True)
    for idx, data in enumerate(loader):
        print(data['features'].shape)

    print("Len: %d" % len(loader))
