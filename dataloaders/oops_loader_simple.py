
# @Author: Enea Duka
# @Date: 6/7/21

from torch.utils.data import Dataset
from dataloaders.dl_utils import Normalize, tensor_to_zero_one
from torch.utils.data.dataloader import default_collate
# from utils.logging_setup import logger
from random import sample, shuffle
import pandas as pd
import os.path as ops
import numpy as np
import ffmpeg
import torch
from torchvision.io import read_video_timestamps
import av
import math
from tqdm import tqdm
import bisect
from utils.arg_parse import opt
from torch.utils.data import DataLoader
from dataloaders.oops_loader import get_video_loader_frames
# from utils.logging_setup import setup_logger_path
# from models.vit import create_vit_model
# from utils.logging_setup import logger
import json
import torchvision
from torchvision import transforms
from PIL import Image


def _convert_image_to_rgb(image):
    return image.convert("RGB")

clip_transform = torchvision.transforms.Compose([
    transforms.Resize(224, interpolation=Image.BICUBIC),
    transforms.CenterCrop(224),
    _convert_image_to_rgb,
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
])

class SimpleOopsDataset(Dataset):
    def __init__(self, mode, fps, feature_level, spat_crop, norm_statistics,
                 spat_scale=False, balance=False, loc_class=False):
        super(SimpleOopsDataset, self).__init__()

        self.feature_level = feature_level
        self.loc_class = loc_class
        self.balance = balance
        if feature_level == 'frames':
            if opt.task == 'classification':
                self.base_video_path = '/BS/unintentional_actions/nobackup/oops/oops_dataset/oops_video/%s' % mode
            elif opt.task == 'regression':
                self.base_video_path = '/BS/unintentional_actions/work/data/oops/vit_features/%s_normalised' % mode
        else:
            # self.base_video_path = '/BS/feat_augm/nobackup/oops/vit_features/%s' % mode
            self.base_video_path = '/BS/unintentional_actions/work/data/oops/vit_features/%s_normalised' % mode
        self.base_video_path_frames = '/BS/unintentional_actions/nobackup/oops/oops_dataset/oops_video/%s' % mode
        csv_path = '/BS/unintentional_actions/work/data/oops/splits/%s_0.csv' % mode
        self.csv = pd.read_csv(csv_path)
        self.feature_level = feature_level
        self.spat_crop = spat_crop
        self.size = 112
        # self.norm = Normalize(mean=norm_statistics['mean'], std=norm_statistics['std'])
        self.norm = Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
        self.spat_scale = spat_scale
        self.fps = fps
        self.training = True if mode == 'train' else False
        self.mode = mode
        self.fpc = 32
        self.clip_step = 1

        self.clip_step_frame = int(self.fps * self.clip_step)
        if feature_level == 'features':
            self._compute_clips()

        with open("/BS/unintentional_actions/nobackup/oops/oops_dataset/annotations/transition_times.json") as f:
            self.fails_borders = json.load(f)

    def __len__(self):
        if self.feature_level == 'features' and not self.loc_class:
            return self.cumulative_sizes[-1]
        else:
            return len(self.csv)


    def _compute_clips(self):
        if self.loc_class:
            clips_meta_path = '/BS/unintentional_actions/work/data/oops/vit_features/%s_clip_meta_full_loc_class.npz' % self.mode
        else:
            clips_meta_path = '/BS/unintentional_actions/work/data/oops/vit_features/%s_clip_meta_full.npz' % self.mode # (self.mode if self.mode == 'val' else (self.mode+'_unbalanced'))
        if ops.isfile(clips_meta_path):
            compressed_file = np.load(clips_meta_path, allow_pickle=True)
            array = compressed_file['arr_0']
            clips = array[0]
            labels = array[1]
            cumulative_sizes = array[2]
            if self.loc_class:
                clip_time_borders = array[3]
        else:
            clips = []
            labels = []
            clip_time_borders = []
            for idx in tqdm(range(len(self.csv))):
                video_path = ops.join(self.base_video_path, self.csv.iloc[idx]['filename'])
                video = self._get_features_video(video_path)
                if self.loc_class:
                    vid_clips, labs, ctb = self._get_clip_idx(video, self.csv.iloc[idx]['t'], video_path)
                else:
                    vid_clips, labs = self._get_clip_idx(video, self.csv.iloc[idx]['t'], video_path)
                clips.append(vid_clips)
                labels.append(labs)
                if self.loc_class:
                    clip_time_borders.append(ctb)
                # if idx == 500:
                #     break
            clip_lengths = torch.as_tensor([len(v) for v in clips])
            cumulative_sizes = clip_lengths.cumsum(0).tolist()
            meta = []
            meta.append(clips)
            meta.append(labels)
            meta.append(cumulative_sizes)
            if self.loc_class:
                meta.append(clip_time_borders)
            meta = np.array(meta)
            np.savez(clips_meta_path, meta)

        self.clips = clips
        self.labels = labels
        self.cumulative_sizes = cumulative_sizes
        if self.loc_class:
            self.clip_time_borders = clip_time_borders

    def _get_clip_idx(self, video, transition_time, video_path):
        indexes = []
        labels = []
        clip_time_borders = []
        transition_frame, video_fps, video_tunit, vid_timestamps = self._get_transition_frame(video_path, transition_time)

        # subsample the video
        num_frames = video.shape[0] * (float(self.fps) / video_fps)
        resampling_idx = self._resample_video_idx(num_frames, video_fps, self.fps)
        video = video[resampling_idx]
        # the transition frame is scaled back since we subsample the video
        transition_frame = int(transition_frame / (float(video_fps) / self.fps))

        p1 = 0
        p2 = p1 + self.fps  # + round(float(video_fps)) if self.loc_class else self.fpc # get 1 sec clips when doing localization by classification
        while p2 <= video.shape[0] - 1:
            indexes.append((p1, p2))

            label = 0
            if p1 <= transition_frame <= p2:
                label = 1
            if transition_frame < p1:
                label = 2
            labels.append(label)

            start_time = float(vid_timestamps[p1] * video_tunit)
            end_time = float(vid_timestamps[p2] * video_tunit)
            clip_time_borders.append((start_time, end_time))

            p1 += self.clip_step_frame
            p2 += self.clip_step_frame

        if p2 < video.shape[0] - 1 + self.clip_step_frame:
            indexes.append((video.shape[0]-1-self.fpc, video.shape[0]-1))
            p1 = video.shape[0]-1-self.fpc
            p2 = video.shape[0]-1

            label = 0
            if p1 <= transition_frame <= p2:
                label = 1
            if transition_frame < p1:
                label = 2
            labels.append(label)

            start_time = float(vid_timestamps[p1] * video_tunit)
            end_time = float(vid_timestamps[p2] * video_tunit)
            clip_time_borders.append((start_time, end_time))

        if self.balance:
            indexes, labels = self._balance_video(indexes, labels)
            return indexes, labels
        elif not self.balance and self.loc_class:
            return indexes, labels, clip_time_borders
        return indexes, labels

    def _get_transition_frame(self, video_path, transition_time):
        video_name = video_path.split('/')[-1]
        video_path = ops.join(self.base_video_path_frames, video_name)
        with av.open(video_path, metadata_errors='ignore') as container:
            video_fps = container.streams.video[0].average_rate
            video_tunit = container.streams.video[0].time_base
            vid_timestamps = read_video_timestamps(video_path)
            transition_pts = int(transition_time / video_tunit)
            pts_step = vid_timestamps[0][1] - vid_timestamps[0][0]
            transition_frame = (transition_pts // pts_step) + 1

            transition_frame_fps = math.ceil(transition_time * video_fps)

            if transition_frame != transition_frame_fps:
                print('Wrong by %d frames' % (transition_frame_fps - transition_frame))

            return transition_frame, video_fps, video_tunit, vid_timestamps[0]

    def _balance_video(self, indexes, labels):
        balanced_idxs = []
        counts = (labels.count(0), labels.count(1), labels.count(2))
        offsets = torch.LongTensor([0] + list(counts)).cumsum(0)[:-1].tolist()
        ratios = (1, 0.93, 1 / 0.93)
        labs = (0, 1, 2)
        lbl_mode = max(labs, key=lambda i: counts[i])
        for i in labs:
            if i != lbl_mode and counts[i] > 0:
                n_to_add = round(counts[i] * ((counts[lbl_mode] * ratios[i] / counts[i]) - 1))
                tmp = list(range(offsets[i], counts[i] + offsets[i]))
                shuffle(tmp)
                tmp_bal_idx = []
                while len(tmp_bal_idx) < n_to_add:
                    tmp_bal_idx += tmp
                tmp_bal_idx = tmp_bal_idx[:n_to_add]
                balanced_idxs += tmp_bal_idx

        indexes += [indexes[i] for i in balanced_idxs]
        labels += [labels[i] for i in balanced_idxs]

        return indexes, labels

    def _get_video_idx_from_clip(self, idx):
        video_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if video_idx == 0:
            clip_idx = idx
        else:
            clip_idx = idx - self.cumulative_sizes[video_idx - 1]
        return video_idx, clip_idx

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


    def _get_video_clips(self, video):
        def _zeropad(tensor):
            n = self.fps - tensor.shape[0] % self.fps
            z = torch.zeros((n, tensor.shape[1]))
            return torch.cat((tensor, z), dim=0)
        video = _zeropad(video)
        return torch.stack(list(video.split(self.fps, dim=0)) ,dim=0)

    def pad_videos_collate_clip(self, batch):
        def _zeropad(tensor, size):
            n = size - tensor.shape[0] % size
            z = torch.zeros((n, tensor.shape[1]))
            return torch.cat((tensor, z), dim=0)

        new_batch = []
        for data in batch:
            video = data['features']
            video = self._get_video_clips(video)
            tmp_pre_un = []
            tmp_trn_un = []
            tmp_post_un = []
            for idx, clip in enumerate(video):
                label = 0
                if idx <= data['t'] < idx + 1:
                    label = 1
                    if self.mode == 'train':
                        tmp_trn_un.append({'features': clip, 'label': label, 'pure_nr_frames': self.fps})
                    else:
                        new_batch.append({'features': clip, 'label': label, 'pure_nr_frames': self.fps})
                elif data['t'] <= idx + 1:
                    label = 2
                    if self.mode == 'train':
                        tmp_post_un.append({'features': clip, 'label': label, 'pure_nr_frames': self.fps})
                    else:
                        new_batch.append({'features': clip, 'label': label, 'pure_nr_frames': self.fps})
                if self.mode == 'train':
                    tmp_pre_un.append({'features': clip, 'label': label, 'pure_nr_frames': self.fps})
                else:
                    new_batch.append({'features': clip, 'label': label, 'pure_nr_frames': self.fps})
            if self.mode == 'train':
                num_trn = len(tmp_trn_un)
                new_batch.extend(tmp_trn_un)

                if len(tmp_pre_un) >= num_trn:
                    new_batch.extend(sample(tmp_pre_un, num_trn))
                elif len(tmp_pre_un) > 0:
                    new_batch.extend(tmp_pre_un)

                if len(tmp_post_un) >= num_trn:
                    new_batch.extend(sample(tmp_post_un, num_trn))
                elif len(tmp_post_un) > 0:
                    new_batch.extend(tmp_post_un)

        shuffle(new_batch)
        # logger.debug("SampledNrm: %d == SampledUNrm: %d" % ())
        return default_collate(new_batch)

    def pad_videos_collate_fn(self, batch):
        def _zeropad(tensor, size):
            n = size - tensor.shape[0] % size
            z = torch.zeros((n, tensor.shape[1]))
            return torch.cat((tensor, z), dim=0)
        new_batch = []
        sampled_nrm = 0
        sampled_unrm = 0
        max_batch_len = max([s['features'].shape[0] for s in batch])
        num_nrm = sum([1 for s in batch if s['t'] == -1])
        for data in batch:
            # if data['t'] != -1:
            #     if sampled_unrm == num_nrm:
            #         continue
            #     sampled_unrm += 1
            # else:
            #     sampled_nrm += 1
            video = data['features']
            # video = merge_along_time(video).squeeze()
            data['pure_nr_frames'] = video.shape[0]
            data['label'] = 1 if data['t'] != -1 else 0
            if video.shape[0] < max_batch_len:
                video = _zeropad(video, max_batch_len)
            data['features'] = video
            new_batch.append(data)
        # logger.debug("SampledNrm: %d == SampledUNrm: %d" % ())
        return default_collate(new_batch)

    def _get_video_dim(self, video_path):
        probe = ffmpeg.probe(video_path)
        video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        return int(video_stream['height']), int(video_stream['width'])

    def _get_output_dim(self, h, w):
        # if self.spat_scale and self.training:
        #     size = np.random.randint(self.size + 2, max(h, w, self.size) * 2)
        # else:
        #     size = self.size
        size = self.size * 2
        if h >= w:
            return int(h * size / w), size
        else:
            return size, int(w * size / h)

    def _preprocess_video(self, tensor):
        tensor = tensor_to_zero_one(tensor)
        tensor = self.norm(tensor)
        return tensor

    def _preprocess_video_clip(self, tensor):
        new_images = []
        for image in tensor:
            try:
                image = Image.fromarray(image)
                new_images.append(clip_transform(image))
            except Exception as e:
                print(e)
        return torch.stack(new_images, dim=0)

    def _get_raw_video(self, video_path):
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
            # if not self.training:
            x = int((width - self.size) / 2.0)
            y = int((height - self.size) / 2.0)
            cmd = cmd.crop(x, y, self.size * 2, self.size * 2)
            height, width = self.size * 2, self.size * 2
            # else:
            #     if (width - self.size) // 2 <= 0:
            #         x = 0
            #     else:
            #         x = np.random.randint(0, (width - self.size) // 2)
            #
            #     if (height - self.size) // 2 <= 0:
            #         y = 0
            #     else:
            #         y = np.random.randint(0, (height - self.size) // 2)
            #
            # cmd = cmd.crop(x, y, height, width)
            # height, width = self.size * 2, self.size * 2

        # if self.hflip:
        #     if np.random.rand() > 0.5:
        #         cmd = cmd.hflip()

        out, _ = (
            cmd.output('pipe:', format='rawvideo', pix_fmt='rgb24')
                .run(capture_stdout=True, quiet=True, capture_stderr=True)
        )
        try:
            video = np.frombuffer(out, np.uint8).reshape([-1, height, width, 3])
        except Exception as e:
            # print(e)
            pass
        video = torch.from_numpy(video.astype('float32'))
        video = video.permute(0, 3, 1, 2)

        video = self._preprocess_video(video)
        return video

    def _get_features_video(self, video_path, clip=None):
        video_path += '.npz'
        compressed_file = np.load(video_path)
        array = compressed_file['arr_0']
        video = torch.from_numpy(array)
        if self.loc_class and clip is not None:
            clips = []
            for c in clip:
                clips.append(video[c[0]:c[1]])
            return torch.stack(clips)
        else:
            video_name = video_path.split('/')[-1].replace('.npz', '')
            with av.open(self.base_video_path_frames+'/'+video_name, metadata_errors='ignore') as container:
                video_fps = container.streams.video[0].average_rate
                num_frames = video.shape[0] * (float(self.fps) / video_fps)
                resampling_idx = self._resample_video_idx(num_frames, video_fps, self.fps)
                video = video[resampling_idx]
                if self.mode == 'train':
                    noise = torch.randn_like(video)*0.01
                    video += noise
                return video[clip[0]:clip[1]] if clip is not None else video

    def vid_to_32_frame_clips(self, batch):

        def _zeropad(tensor, len):
            n = len - tensor.shape[0] % len
            z = torch.zeros((n, tensor.shape[1], tensor.shape[2], tensor.shape[3]))
            return torch.cat((tensor, z), dim=0)

        def _get_video_t_unit(video_path):
            try:
                with av.open(video_path, metadata_errors='ignore') as container:
                    vid_mdata = container.streams.video[0]
                    return vid_mdata.time_base, vid_mdata.average_rate, vid_mdata.frames
            except Exception:
                return None, None, None

        try:
            video = batch[0]['features']
            if video.shape[0] % 32 != 0:
                video = _zeropad(video, math.ceil(video.shape[0] / 32) * 32)

            clips = torch.stack(list(torch.split(video, 32)), dim=0)
            batch[0]['features'] = clips

            video_name = batch[0]['filename'].replace('.mp4', '')
            borders = self.fails_borders[video_name]
            video_path = ops.join(self.base_video_path, video_name) + '.mp4'
            video_tunit, original_fps, all_frames = _get_video_t_unit(video_path)
            if video_tunit is None:
                return {}
            t_fail = borders['t']
            t_fail = [x for x in t_fail if x != -1]
            try:
                t_fail = t_fail[len(t_fail) // 2]
            except Exception as e:
                print('here')
            fail_frame = int(t_fail * original_fps)
            labels = []

            for idx, feat_clip in enumerate(clips):
                video_clip_start_frame = idx * 32
                # if video_clip_start_frame > all_frames:
                #     break
                video_clip_end_frame = video_clip_start_frame + 32

                label = 0
                if video_clip_end_frame < fail_frame:
                    label = 0
                elif video_clip_start_frame <= fail_frame <= video_clip_end_frame:
                    label = 1
                elif fail_frame < video_clip_start_frame:
                    label = 2
                else:
                    print('error_creating_label!!')
                    raise
                labels.append(label)
            batch[0]['labels'] = torch.tensor(labels)

            return default_collate(batch)
        except Exception:
            print('Problem when loading video!')
            return default_collate(batch)

    def __getitem__(self, idx):
        output = {}
        try:
            if self.feature_level == 'features':
                video_idx, clip_idx = self._get_video_idx_from_clip(idx)
                video_path = ops.join(self.base_video_path, self.csv.iloc[video_idx]['filename'])
                if video_path[-4:] != '.mp4' and 'unlabeled' in self.csv_path:
                    video_path += '.mp4'
                    output['features'] = self._get_raw_video(
                    video_path) if self.feature_level == 'frames' else self._get_features_video(video_path)
                    output['filename'] = self.csv.iloc[idx]['filename']
                    return output
                if self.loc_class:
                    clip = self.clips[video_idx]
                else:
                    clip = self.clips[video_idx][clip_idx]
                output['features'] = self._get_features_video(video_path, clip)
                output['len'] = self.csv.iloc[video_idx]['len']
                output['t'] = self.csv.iloc[video_idx]['t']
                output['rel_t'] = self.csv.iloc[video_idx]['t_rel']
                output['filename'] = self.csv.iloc[video_idx]['filename']

                if self.loc_class:
                    output['label'] = torch.tensor(self.labels[video_idx])
                    output['pure_nr_frames'] = torch.tensor([output['features'].shape[1]] * (output['features'].shape[0]))
                    output['clip_time_boarders'] = torch.tensor(self.clip_time_borders[video_idx])
                else:
                    output['pure_nr_frames'] = output['features'].shape[0]
                    output['label'] = self.labels[video_idx][clip_idx]
            else:
                video_path = ops.join(self.base_video_path, self.csv.iloc[idx]['filename'])
                # print(video_path)
                if video_path[-4:] != '.mp4' and opt.task == 'classification':
                    video_path += '.mp4'
                if opt.task == 'classification':
                    output['features'] = self._get_raw_video(video_path)
                else:
                    output['features'] = self._get_features_video(video_path)
                # output['len'] = self.csv.iloc[idx]['len']
                # output['t'] = self.csv.iloc[idx]['t']
                # output['rel_t'] = self.csv.iloc[idx]['t_rel']
                output['filename'] = self.csv.iloc[idx]['filename']
        except Exception as e:
            # vid_clip_len = len(self.clips[video_idx])
            # vid_lbl_len = len(self.labels[video_idx])
            print(e)
            return {'filename': self.csv.iloc[idx]['filename']}

        return output


if __name__ == '__main__':

    # model = create_vit_model()
    # model.cuda()
    # model.eval()

    train_set = SimpleOopsDataset('train', 16, 'features' if opt.task == 'classification' else 'frames', True,
                                  {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}, balance=True)
    val_set = SimpleOopsDataset('val', 16, 'features' if opt.task == 'classification' else 'frames', True,
                                {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]})

    train_loader = DataLoader(val_set,
                              num_workers=32,
                              batch_size=256,
                              shuffle=True,
                              drop_last=True)

    opt.batch_size = 1
    opt.workers = 0
    opt.balance_fails_only = True
    opt.all_fail_videos = True
    opt.selfsup_loss = 'fps'
    train_loader = get_video_loader_frames(opt)
    opt.val = True
    opt.fails_path = '/BS/unintentional_actions/nobackup/oops/oops_dataset/oops_video'
    # val_loader = get_video_loader(opt)

    zeros = 0
    ones = 0
    twos = 0

    for idx, data in enumerate(tqdm(train_loader)):
        # labels = data['label'].tolist()
        # # clip = data[0]
        # # clip_feats = model(clip.squeeze().permute(1, 0, 2, 3))
        # zeros += labels.count(0)
        # ones += labels.count(1)
        # twos += labels.count(2)
        pass
    print(zeros)
    print(ones)
    print(twos)

