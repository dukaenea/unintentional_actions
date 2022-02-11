
# @Author: Enea Duka
# @Date: 6/13/21

from torch.utils.data import Dataset, WeightedRandomSampler
from os import walk
from random import random, randint
import random as rnd
from rep_learning.feat_transformer import fast_forward
from torch.utils.data.dataloader import default_collate
from utils.arg_parse import opt

import pandas as pd
import os.path as pth
import numpy as np
import math
import torch
from torch.utils.data import DataLoader
import utils.py12transforms as T
import torchvision
import av
import ffmpeg
from dataloaders.dl_utils import Normalize, tensor_to_zero_one
import math
from utils.logging_setup import logger


class ROKDataset(Dataset):
    def __init__(self, mode, spat_scale, size, spat_crop, load_frames=False):
        super(ROKDataset, self).__init__()

        self.mode = mode
        self.spat_scale = spat_scale
        self.size = size
        self.spat_crop = spat_crop
        self.dset_to_idx = {'kinetics': 0, 'oops': 1, 'rareact': 2}
        self.load_frames = load_frames
        self.norm = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        # self.csv_path = '/BS/unintentional_actions/work/data/kinetics/splits/%s_rep_lrn.csv' % mode
        # self.csv_path = '/BS/unintentional_actions/work/data/rok/data_splits/%s_rep_lrn_ok.csv' % mode
        # if opt.run_on_mpg:
        #     self.csv_path = '/u/eneaduka/data/oops/splits/%s_all_org.csv' % mode
        # else:
        self.csv_path = '/BS/unintentional_actions/work/data/oops/splits/%s_all_org.csv' % mode
        # self.csv_path = '/BS/unintentional_actions/work/data/rok/data_splits/%s_rep_lrn_kin.csv' % mode

        if load_frames:
            # if opt.run_on_mpg:
            #     self.oops_video_path = '/u/eneaduka/datasets/oops/oops_dataset/oops_video/%s' % mode
            # else:
            self.oops_video_path = '/BS/unintentional_actions/nobackup/oops/oops_dataset/oops_video/%s_downsize' % mode
        else:
            self.kinetics_video_path = '/BS/unintentional_actions/work/data/kinetics/vit_features/%s' % mode
            # if opt.run_on_mpg:
            #     self.oops_video_path = '/u/eneaduka/datasets/oops/vit_features/%s_all' % mode
            #     self.oops_video_path_labeld = '/u/eneaduka/data/oops/vit_features/%s_normalised' % mode
            # else:
            self.oops_video_path = '/BS/unintentional_actions/work/data/oops/vit_features/%s_all' % mode
            self.oops_video_path_labeld = '/BS/unintentional_actions/work/data/oops/vit_features/%s_normalised' % mode
            self.rareact_video_path = '//BS/unintentional_actions/work/data/rareact/vit_features/positive_negative/%s' % mode

        self.csv = pd.read_csv(self.csv_path)

        if opt.backbone == 'r3d_18':
            normalize = T.Normalize(mean=[0.43216, 0.394666, 0.37645],
                                    std=[0.22803, 0.22145, 0.216989])
            unnormalize = T.Unnormalize(mean=[0.43216, 0.394666, 0.37645],
                                        std=[0.22803, 0.22145, 0.216989])

            self.train_transform = torchvision.transforms.Compose([
                T.ToFloatTensorInZeroOne(),
                T.Resize((128, 171)),
                T.RandomHorizontalFlip(),
                normalize,
                T.RandomCrop((112, 112))
            ])
            self.test_transform = torchvision.transforms.Compose([
                T.ToFloatTensorInZeroOne(),
                T.Resize((128, 171)),
                normalize,
                T.CenterCrop((112, 112))
            ])

        elif opt.backbone == 'vit_longformer':
            normalize = T.Normalize(mean=[0.5, 0.5, 0.5],
                                    std=[0.5, 0.5, 0.5])
            unnormalize = T.Unnormalize(mean=[0.5, 0.5, 0.5],
                                        std=[0.5, 0.5, 0.5])
            self.train_transform = torchvision.transforms.Compose([
                T.ToFloatTensorInZeroOne(),
                normalize
            ])
            self.test_transform = torchvision.transforms.Compose([
                T.ToFloatTensorInZeroOne(),
                T.Resize((int(224 * 4))),
                T.CenterCrop((224, 224)),
                normalize,
            ])

        if 'normal' not in opt.transformations_list:
            self.speeds = [1, 2, 3]
        elif 'speedx2' not in opt.transformations_list:
            self.speeds = [0, 2, 3]
        elif 'speedx3' not in opt.transformations_list:
            self.speeds = [0, 1, 3]
        elif 'speedx4' not in opt.transformations_list:
            self.speeds = [0, 1, 2]
        else:
            self.speeds = [0, 1, 2, 3]

        return
        # self.kin_filename_to_path = self._get_kin_dict()

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):
        # if idx < 3800:
        #     return {'features': torch.rand((128, 3, 112, 112)), 'dataset': 'oops'}
        # dset = self.csv['dataset'][idx]
        dset = 'oops'
        filename = self.csv['filename'][idx]
        # if dset == 'kinetics':
        #     video = self._load_video_features(self.kin_filename_to_path[filename])
        # else:
        vid_path = pth.join(self.oops_video_path, filename) + ('.npz' if not self.load_frames else '.mp4')
        if not pth.isfile(vid_path):
            return None
            vid_path = pth.join(self.oops_video_path_labeld, filename) + ('.mp4.npz' if not self.load_frames else '.mp4')

        # if not pth.isfile(vid_path):
        #     print('not_file')

        video = self._load_video_features(vid_path) if not self.load_frames else self._load_video_frames(vid_path)
        if video is None:
            idx = randint(0, len(self.csv))
            filename = self.csv['filename'][idx]

            vid_path = pth.join(self.oops_video_path, filename) + ('.npz' if not self.load_frames else '.mp4')
            if not pth.isfile(vid_path):
                vid_path = pth.join(self.oops_video_path_labeld, filename) + (
                    '.mp4.npz' if not self.load_frames else '.mp4')
            if not pth.isfile(vid_path):
                print('here')
            video = self._load_video_features(vid_path) if not self.load_frames else self._load_video_frames(vid_path)
        # video = self._load_video_features(pth.join(vid_path if dset == 'oops' else self.rareact_video_path, filename)+'.npz')

        # if video is None:
        #     i = 0
        #     while i < 10:
        #         idx = randint(0, len(self.csv))
        #         i += 1
        #         # dset = self.csv['dataset'][idx]
        #         dset = 'kinetics'
        #         filename = self.csv['filename'][idx]
        #         if dset == 'kinetics':
        #             video = self._load_video_features(self.kin_filename_to_path[filename])
        #         else:
        #             video = self._load_video_features(
        #                 pth.join(self.oops_video_path if dset == 'oops' else self.rareact_video_path,
        #                          filename) + '.npz')
        #
        if opt.backbone == 'r3d_18':
            if self.mode == 'train':
                video = self.train_transform(video)
            elif self.mode == 'val':
                video = self.test_transform(video)

            video = video.permute(1, 0, 2, 3)

        return {'features': video, 'dataset': dset}

    def _get_video_dim(self, video_path):
        try:
            probe = ffmpeg.probe(video_path)
            video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
            return int(video_stream['height']), int(video_stream['width'])
        except Exception as e:
            print(e)
            print(video_path)
            return None, None

    def _get_output_dim(self, h, w):
        # if self.spat_scale and self.mode == 'train':
        #     size = np.random.randint(self.size + 2, max(h, w, self.size))
        # else:
        #     size = self.size
        size = 224
        if h >= w:
            return int(h * size / w), size
        else:
            return size, int(w * size / h)

    def random_gray(self, vid, p):
        if random() < p:
            # first select a channel
            channel = randint(0, 2)
            vid = vid[:, channel, :, :].unsqueeze(1)
            vid = vid.repeat(1, 3, 1, 1)
        return vid

    def _get_raw_video(self, video_path):
        # height, width = self._get_video_dim(video_path)
        h, w = self._get_video_dim(video_path)
        if w is None and h is None:
            logger.debug("Found corrupted video")
            return None
        max_angle = math.pi / 16
        angle = (random() * (max_angle * 2)) - max_angle

        height, width = self._get_output_dim(h, w)
        cmd = (
            ffmpeg
                .input(video_path)
                .filter('scale', width, height)
        )

        rt_rnd = random()
        # if self.mode == 'train' and rt_rnd < 0.8:
        #     cmd = cmd.filter('rotate', angle)

        # if self.spat_crop:
        #     if not self.mode == 'train':
        #         x = int((width - self.size) / 2.0)
        #         y = int((height - self.size) / 2.0)
        #         cmd = cmd.crop(x, y, self.size, self.size)
        #         height, width = self.size, self.size
        #     else:
        #         if (width - self.size) // 2 <= 0:
        #             x = 0
        #         else:
        #             x = np.random.randint(0, (width - self.size) // 2)
        #
        #         if (height - self.size) // 2 <= 0:
        #             y = 0
        #         else:
        #             y = np.random.randint(0, (height - self.size) // 2)
        #
        #         cmd = cmd.crop(x, y, self.size, self.size)
        #         height, width = self.size, self.size
        #
        # flip_prb = random()
        # if self.mode == 'train' and flip_prb < 0.5:
        #     cmd = cmd.hflip()
        #

        out, _ = (
            cmd.output('pipe:', format='rawvideo', pix_fmt='rgb24')
                .run(capture_stdout=True, quiet=True)
        )


        video = np.frombuffer(out, np.uint8).reshape([-1, width, height, 3])
        video = torch.from_numpy(video.astype('float32'))
        video = video.permute(0, 3, 1, 2)

        # if self.mode == 'train':
        #     video = self.random_gray(video, 0.3)

        # video = self._preprocess_video(video)
        return video

    def _preprocess_video(self, tensor):
        tensor = tensor_to_zero_one(tensor)
        tensor = self.norm(tensor)
        return tensor

    def _load_video_frames(self, video_path):
        video = self._get_raw_video(video_path)
        # video = video.permute(1, 0, 2, 3)
        if video is None:
            return None
        # video = train_transform(video) if self.mode == 'train' else test_transform(video)
        return video

    def _load_video_features(self, video_path):
        try:
            compressed_file = np.load(video_path)
            array = compressed_file['arr_0']
            video = torch.from_numpy(array)
        except Exception as e:
            print(e)
            return None
            # raise
        # if '.mp4' in video_path:
            # print('Got labeled')
        return video

    def _get_kin_dict(self):
        base, label_dirs, _ = next(walk(self.kinetics_video_path))
        kin_dict = {}
        for dir in label_dirs:
            _, _, files = next(walk(pth.join(base, dir)))
            for file in files:
                kin_dict[file[:-4]] = pth.join(base, dir, file)

        return kin_dict

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
        sub_len = min(vid_len, min_seq_len) #if eff_skip == 1 else (min_seq_len - 1) * eff_skip)
        crop_idx = self._random_crop_vector(vid_idx, sub_len)
        video = video[crop_idx]
        idxs = torch.arange(0, video.shape[0]) # torch.arange(0, (min_seq_len - 2) * eff_skip, step=eff_skip)
        start = 0 # 1 if eff_skip > 1 else 0
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

        start_idx = randint(min_start_idx, max_start_idx+1)
        return vector[start_idx: start_idx + crop_len]


    # def _crop_video_temporally_controlled(self, video, min_seq_len):
    #     video_len = video.shape[0]
    #     video_idx = torch.arange(0, video_len)
    #     max_start_pos = (video_idx.shape[0] - min_seq_len) // 2
    #     start_pos = randint(0, max_start_pos)


    def _stitch_videos(self, nrm_video, abnormal_video, min_seq_len):
        stitched_video = torch.cat([nrm_video, abnormal_video], dim=0)
        quarter_len = nrm_video.shape[0] // 4
        start_idx = randint(quarter_len, quarter_len*3)
        s_vid = stitched_video[start_idx: start_idx+min_seq_len]
        return s_vid


    def stitch_half_speed_video(self, video, fpc, speed_up_factor):
        remainder = video.shape[0] % fpc
        if remainder > 0:
            video = video[:-remainder]
        bisect_idxs = list(range(0, video.shape[0], fpc))
        bisect_idx = bisect_idxs[len(bisect_idxs)//2]
        sc1 = video[:bisect_idx]
        sc2 = video[bisect_idx:]
        sc1 = self.split_video_in_clips(sc1, fpc)
        sc2 = self.speed_up_video_clip_level(sc2, fpc, speed_up_factor)
        stitched_video = torch.cat((sc1, sc2), dim=0)
        return stitched_video

    def speed_up_video_clip_level(self, video, fpc, speed_up_factor):
        remainder = video.shape[0] % fpc
        if remainder > 0:
            video = video[:-remainder]
        resampling_idx = list(range(0, video.shape[0], speed_up_factor))
        video = video[resampling_idx]
        video = self.split_video_in_clips(video, fpc)
        return video

    def bisect_and_swap_video(self, video, fpc):
        remainder = video.shape[0] % fpc
        if remainder > 0:
            video = video[:-remainder]
        bisect_idxs = list(range(int(video.shape[0]*0.25), int(video.shape[0]*0.75), fpc))
        bisect_idx = rnd.choice(bisect_idxs)
        sc1 = video[:bisect_idx]
        sc2 = video[bisect_idx:]
        swapped_video = torch.cat((sc2, sc1), dim=0)
        swapped_video = self.split_video_in_clips(swapped_video, fpc)
        return swapped_video

    def shuffle_video_clips(self, video, fpc):
        split_video = self.split_video_in_clips(video, fpc)
        shuffl_idx = torch.randperm(split_video.shape[0])
        split_video = split_video[shuffl_idx]
        return split_video

    def foba_video_in_clips(self, video, fpc):
        remainder = video.shape[0] % fpc
        if remainder > 0:
            video = video[:-remainder]
        split_idxs = list(range(int(video.shape[0]*0.25), int(video.shape[0]*0.75), fpc))
        split_idx = rnd.choice(split_idxs)
        fw = video[:split_idx]
        bw = torch.flip(video[split_idx:], dims=[0])
        foba_video = torch.cat((fw, bw), dim=0)
        foba_video = self.split_video_in_clips(foba_video, fpc)
        return foba_video

    def split_video_in_clips(self, video, fpc):
        remainder = video.shape[0] % fpc
        if remainder > 0:
            video = video[:-remainder]
        return torch.stack(list(torch.split(video, fpc)), dim=0)


    def create_temp_disrupt(self, video):
        crop_length = 80
        video_len = video.shape[0]
        min_idx = crop_length // 2
        max_idx = video_len - (crop_length // 2)
        crop_idx = randint(min_idx, max_idx)
        video = video[crop_idx - (crop_length//2):crop_idx + (crop_length//2)]
        # get a random split index
        split_portion = rnd.choice([0.2, 0.5, 0.8])
        split_idx = int(split_portion * crop_length)
        fw = video[:split_idx]
        bw = torch.flip(video[split_idx:], dims=[0])
        td_video = torch.cat((fw, bw), dim=0)
        label = split_portion

        return td_video, label


    def reg_rep_learning(self, batch):
        def _zeropad(tensor, size):
            n = size - tensor.shape[0] % size
            z = -torch.ones((n, tensor.shape[1]))
            return torch.cat((tensor, z), dim=0)
        new_batch = []

        for idx, data in enumerate(batch):
            video = data['features']
            if video is None or video.shape[0] < 80:
                continue
            td_video, label = self.create_temp_disrupt(video)
            new_batch.append({'features': td_video, 'label': label, 'pure_nr_frames': td_video.shape[0]})

        max_len = max([s['features'].shape[0] for s in new_batch])
        for data in new_batch:
            if data['features'].shape[0] < max_len:
                data['features'] = _zeropad(data['features'], max_len)

        return default_collate(new_batch)


    def video_level_speed_and_motion_collate_fn(self, batch):
        def _zeropad(tensor, size):
            n = size - tensor.shape[0] % size
            if len(tensor.shape) == 5:
                z = -torch.ones((n, tensor.shape[1], tensor.shape[2], tensor.shape[3], tensor.shape[4]))
            else:
                z = -torch.ones((n, tensor.shape[1], tensor.shape[2]))
            return torch.cat((tensor, z), dim=0)

        new_batch = []
        fpc = 16
        for idx, data in enumerate(batch):
            if data is None:
                continue
            video = data['features']
            if video is None: #  or video.shape[0] < fpc * 3:
                continue

            dset = data['dataset']

            speed_up_factor = randint(2, 3)
            trn_idx = randint(0, 6)
            selected = False

            if 'speed' in opt.transformation_groups:
                # if trn_idx in [0, 1]:
                if video.shape[0] > (fpc * speed_up_factor * 4):
                    s_video = self.speed_up_video_clip_level(video, fpc, speed_up_factor)
                    new_batch.append({'features': s_video,
                                      'label': (4 if 'motion' in opt.transformation_groups else 0) + (
                                                  speed_up_factor - 2), 'pure_nr_frames': s_video.shape[0],
                                      'dset': dset})
                        # selected = True

                # if not selected:
                #     trn_idx = randint(2, 6)

                nrm_video = self.split_video_in_clips(video, fpc)
                new_batch.append({'features': nrm_video, 'label': 2 if 'motion' in opt.transformation_groups else 2,
                                  'pure_nr_frames': nrm_video.shape[0], 'dset': dset})

            if 'motion' in opt.transformation_groups:

                # if trn_idx == 3:
                class_idx = 3
                if 'random_point_speedup' in opt.transformations_list:
                    class_idx += 1
                    if video.shape[0] > (fpc * speed_up_factor * 4):
                        st_video = self.stitch_half_speed_video(video, fpc, speed_up_factor)
                        new_batch.append({'features': st_video, 'label': class_idx, 'pure_nr_frames': st_video.shape[0], 'dset': dset})

                if 'double_flip' in opt.transformations_list:
                    foba_video = self.foba_video_in_clips(video, fpc)
                    new_batch.append({'features': foba_video, 'label': class_idx, 'pure_nr_frames': foba_video.shape[0], 'dset': dset})
                    class_idx += 1

                if 'shuffle' in opt.transformations_list:
                    sc_video = self.shuffle_video_clips(video, fpc)
                    new_batch.append({'features': sc_video, 'label': class_idx, 'pure_nr_frames': sc_video.shape[0], 'dset': dset})
                    class_idx += 1

                if 'warp' in opt.transformations_list:
                    sw_video = self.bisect_and_swap_video(video, fpc)
                    new_batch.append({'features': sw_video, 'label': class_idx, 'pure_nr_frames': sw_video.shape[0], 'dset': dset})


        max_len = max([s['features'].shape[0] for s in new_batch])
        for data in new_batch:
            if data['features'].shape[0] < max_len:
                data['features'] = _zeropad(data['features'], max_len)


        return default_collate(new_batch)


    def rot_collate_fn(self, batch):
        new_batch = []

        for idx, data in enumerate(batch):
            video = data['features']
            video = self._crop_video_temporally(video, 20)

            nine_video = torch.rot90(video, 1, [2, 3])
            oeight_video = torch.rot90(nine_video, 1, [2, 3])
            tseven_video = torch.rot90(oeight_video, 1, [2, 3])

            frames = torch.cat([video, nine_video, oeight_video, tseven_video], dim=0)
            labels = torch.tensor([0] * 20 + [1] * 20 + [2] * 20 + [3] * 20)

            perm_idxs = torch.randperm(20 * 4)
            frames = frames[perm_idxs]
            labels = labels[perm_idxs]

            for i in range(frames.shape[0]):
                new_batch.append({'features ': frames[i], 'label': labels[i], 'pure_nr_frames': 1})

        return default_collate(new_batch)

    def speed_and_motion_collate_fn(self, batch):
        def _zeropad(tensor, size):
            n = size - tensor.shape[0] % size
            if len(tensor.shape) == 2:
                z = -torch.ones((n, tensor.shape[1]))
            elif len(tensor.shape) == 4:
                z = -torch.ones((n, tensor.shape[1], tensor.shape[2], tensor.shape[2]))
            return torch.cat((tensor, z), dim=0)
        new_batch = []
        if opt.consist_lrn:
            new_nrm_batch = []
        min_seq_len = 20
        for idx, data in enumerate(batch):
            if data is None:
                continue
            video = data['features']
            # if self.load_frames:
            #     video = video.permute(1, 0, 2, 3)
            if video is None:
                continue
            p_speed = 1
            p_motion = 1
            # dset = 'oops'
            try:
                dset = data['dataset']
            except KeyError:
                dset = 'avenue'
            if video.shape[0] > min_seq_len:
                if opt.consist_lrn:
                    new_nrm_batch.append(
                        {'features': video, 'label': 0, 'pure_nr_frames': video.shape[0], 'org_vid_idx': -1, 'dset': dset})
                # first modify the speed of the video
                trn = randint(0, 78)
                s_video, speed_label, eff_skip, max_speed = self._speed_up_video(video, min_seq_len)
                nrm_cropped_video = self._crop_video_temporally(video, min_seq_len)
                if 'speed' in opt.transformation_groups:
                    new_batch.append({'features': s_video, 'label': speed_label, 'pure_nr_frames': s_video.shape[0],
                                      'org_vid_idx': idx if opt.consist_lrn else -1, 'dset': dset})
                    # then shuffle the video
                    new_batch.append({'features': nrm_cropped_video, 'label': 0, 'pure_nr_frames': nrm_cropped_video.shape[0],
                                      'org_vid_idx': idx if opt.consist_lrn else -1, 'dset': dset})

                if 'motion' in opt.transformation_groups:
                    speed_list = ['normal', 'speedx2', 'speedx3', 'speedx4']
                    class_idx = 4 if set(speed_list).issubset(opt.transformations_list) else 3
                    if 'shuffle' in opt.transformations_list:
                        sh_video = self._shuffle_video(video, min_seq_len)
                        new_batch.append({'features': sh_video, 'label': class_idx if 'speed' in opt.transformation_groups else 0, 'pure_nr_frames': sh_video.shape[0],
                                          'org_vid_idx': idx if opt.consist_lrn else -1, 'dset': dset})
                        class_idx += 1

                    if 'double_flip' in opt.transformations_list:
                    # then do e foba
                        foba_video = self._foba_video(video, eff_skip, min_seq_len)
                        new_batch.append({'features': foba_video, 'label': class_idx if 'speed' in opt.transformation_groups else 1, 'pure_nr_frames': foba_video.shape[0],
                                          'org_vid_idx': idx if opt.consist_lrn else -1, 'dset': dset})
                        class_idx += 1

                    if 'random_point_speedup' in opt.transformations_list:
                        stitched_video = self._stitch_videos(nrm_cropped_video, s_video, min_seq_len)
                        new_batch.append({'features': stitched_video, 'label': class_idx if 'speed' in opt.transformation_groups else 2, 'pure_nr_frames': stitched_video.shape[0],
                                          'org_vid_idx': idx if opt.consist_lrn else -1, 'dset': dset})
                        class_idx += 1

                    if 'warp' in opt.transformations_list:
                        # # and at the end warp the time
                        twarp_video = self._twarp_video(video, max_speed, min_seq_len)
                        new_batch.append({'features': twarp_video, 'label': class_idx if 'speed' in opt.transformation_groups else 3, 'pure_nr_frames': twarp_video.shape[0],
                                          'org_vid_idx': idx if opt.consist_lrn else -1, 'dset': dset})
                        class_idx += 1


        if opt.consist_lrn:
            new_batch += new_nrm_batch

        if len(new_batch) > 0:
            if self.load_frames:
                min_len = min([s['features'].shape[0] for s in new_batch])
                for data in new_batch:
                    video = data['features']
                    if data['features'].shape[0] > min_len:
                        data['features'] = video[:min_len]
                    # video = video.permute(0, 2, 3, 1)
                    # data['features'] = train_transform(video).permute(1, 0, 2, 3)
            else:
                max_len = max([s['features'].shape[0] for s in new_batch])
                for data in new_batch:
                    if data['features'].shape[0] < max_len:
                        data['features'] = _zeropad(data['features'], max_len)

        return default_collate(new_batch)

    def artificial_actions_collate_fn(self, batch):
        def _zeropad(tensor, size):
            n = size - tensor.shape[0] % size
            z = -torch.ones((n, tensor.shape[1]))
            return torch.cat((tensor, z), dim=0)

        new_batch = []
        min_seq_len = 20
        for idx, data in enumerate(batch):
            video = data['features']
            dset = data['dataset']
            # dset = 'oops'
            if video.shape[0] > min_seq_len:
                # first modify the speed of the video
                nrm_cropped_video = self._crop_video_temporally(video, min_seq_len)
                new_batch.append({'features': nrm_cropped_video, 'label': 0, 'pure_nr_frames': nrm_cropped_video.shape[0],
                                  'org_vid_idx': idx if opt.consist_lrn else -1, 'dset': dset})

                s_video, speed_label, eff_skip, max_speed = self._speed_up_video(video, min_seq_len)
                new_batch.append({'features': s_video, 'label': 2, 'pure_nr_frames': s_video.shape[0],
                                  'org_vid_idx': idx if opt.consist_lrn else -1, 'dset': dset})

                sh_video = self._shuffle_video(video, min_seq_len)
                new_batch.append({'features': sh_video, 'label': 2, 'pure_nr_frames': sh_video.shape[0],
                                  'org_vid_idx': idx if opt.consist_lrn else -1, 'dset': dset})

                # and at the end warp the time
                twarp_video = self._twarp_video(video, max_speed, min_seq_len)
                new_batch.append({'features': twarp_video, 'label': 2, 'pure_nr_frames': twarp_video.shape[0],
                                  'org_vid_idx': idx if opt.consist_lrn else -1, 'dset': dset})

                # then do e foba
                foba_video = self._foba_video(video, eff_skip, min_seq_len)
                new_batch.append({'features': foba_video, 'label': 1, 'pure_nr_frames': foba_video.shape[0],
                                  'org_vid_idx': idx if opt.consist_lrn else -1, 'dset': dset})

                stitched_video = self._stitch_videos(nrm_cropped_video, s_video, min_seq_len)
                new_batch.append({'features': stitched_video, 'label': 1, 'pure_nr_frames': stitched_video.shape[0],
                                  'org_vid_idx': idx if opt.consist_lrn else -1, 'dset': dset})

        if len(new_batch) > 0:
            max_len = max([s['features'].shape[0] for s in new_batch])
            for data in new_batch:
                if data['features'].shape[0] < max_len:
                    data['features'] = _zeropad(data['features'], max_len)

        return default_collate(new_batch)

    def get_rok_sampler(self):
        # count samples for each class
        dsets = list(self.csv['dataset'])
        dset_sample_count = np.unique(dsets, return_counts=True)[1]
        num_samples = sum(dset_sample_count)
        kin_ratio = 1./(dset_sample_count[0]/num_samples)
        oops_ratio = 1./(dset_sample_count[1]/num_samples)
        rareact_ratio = 1./(dset_sample_count[2]/num_samples)
        weights = [kin_ratio if ds=='kinetics' else (oops_ratio if ds=='oops' else rareact_ratio) for ds in dsets]
        return WeightedRandomSampler(weights, len(dsets))

if __name__ == '__main__':
    dataset = ROKDataset('val', load_frames=True)
    dloader = DataLoader(dataset,
                         num_workers=32,
                         batch_size=4,
                         collate_fn=dataset.speed_and_motion_collate_fn)
    for idx, data in enumerate(dloader):
        print(idx)