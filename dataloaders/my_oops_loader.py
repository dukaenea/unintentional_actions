# @Author: Enea Duka
# @Date: 6/16/21

import json
import os
import random
import statistics
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
import time

import utils.py12transforms as T
from utils.sampler import (
    DistributedSampler,
    UniformClipSampler,
    RandomClipSampler,
    ConcatSampler,
)


normalize = T.Normalize(
    mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]
)
unnormalize = T.Unnormalize(
    mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]
)
normalize = T.Normalize(
    mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]
)
unnormalize = T.Unnormalize(
    mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]
)
train_transform = torchvision.transforms.Compose(
    [
        T.ToFloatTensorInZeroOne(),
        T.Resize((128, 171)),
        T.RandomHorizontalFlip(),
        normalize,
        T.RandomCrop((112, 112)),
    ]
)
test_transform = torchvision.transforms.Compose(
    [
        T.ToFloatTensorInZeroOne(),
        T.Resize((128, 171)),
        normalize,
        T.CenterCrop((112, 112)),
    ]
)


class KineticsAndFails(VisionDataset):
    def __init__(
        self,
        fails_path,
        frames_per_clip,
        step_between_clips,
        fps,
        transform=None,
        extensions=(".mp4",),
        video_clips=None,
        fails_only=False,
        val=False,
        balance_fails_only=False,
        get_clip_times=False,
        fails_video_list=None,
        fns_to_remove=None,
        all_fail_videos=True,
        selfsup_loss=None,
        clip_interval_factor=None,
        labeled_fails=True,
        debug_dataset=False,
        anticipate_label=0,
        data_proportion=1,
        t_units=None,
        base_features_path=None,
        **kwargs,
    ):
        self.clip_len = frames_per_clip / fps
        self.clip_step = step_between_clips / fps
        self.clip_interval_factor = clip_interval_factor
        self.fps = fps
        self.t = transform
        self.video_clips = None
        self.fails_path = fails_path
        self.selfsup_loss = selfsup_loss
        self.get_clip_times = get_clip_times
        self.anticipate_label = anticipate_label
        data_proportion = 1 if val else data_proportion
        if base_features_path is None:
            raise ValueError("Invalid value for base features path.")
        self.base_features_path = base_features_path
        self.video_time_units = t_units

        if video_clips:
            self.video_clips = video_clips  # use the provided video clips
        else:
            # load the clips from storage
            assert fails_path is None or fails_video_list is None
            video_list = fails_video_list or glob(
                os.path.join(fails_path, "**", "*.mp4"), recursive=True
            )
            # if not fails_only:
            #     kinetics_cls = torch.load("PATH/TO/kinetics_classes.pt")
            #     kinetics_dist = torch.load("PATH/TO/dist.pt")
            #     s = len(video_list)
            #     for i, n in kinetics_dist.items():
            #         n *= s
            #         video_list += sorted(
            #             glob(os.path.join(kinetics_path, '**', kinetics_cls[i], '*.mp4'), recursive=True))[
            #         :round(n)]
            self.video_clips = VideoClips(
                video_list, frames_per_clip, step_between_clips, fps, num_workers=16
            )
        with open(
            "../resources/metadata/oops/annotations/heldout_transition_times.json"
        ) as f:
            self.fails_borders = json.load(f)
        with open("../resources/metadata/oops/annotations/transition_times.json") as f:
            self.fails_data = json.load(f)
        self.fails_only = fails_only
        # get start and end time of the clip
        self.t_from_clip_idx = lambda idx: (
            (step_between_clips * idx) / fps,
            (step_between_clips * idx + frames_per_clip) / fps,
        )
        if (
            not balance_fails_only
        ):  # no support for recompute clips after balance calc yet
            self.video_clips.compute_clips(frames_per_clip, step_between_clips, fps)
        if video_clips is None and fails_only and labeled_fails:
            # if True:
            if not all_fail_videos:
                idxs = []
                for i, video_path in enumerate(self.video_clips.video_paths):
                    video_path = os.path.splitext(os.path.basename(video_path))[0]
                    if video_path in self.fails_data:
                        idxs.append(i)
                self.video_clips = self.video_clips.subset(idxs)
            # if not val and balance_fails_only:  # balance dataset
            # ratios = {0: 0.3764, 1: 0.0989, 2: 0.5247}
            self.video_clips.labels = []
            self.video_clips.compute_clips(frames_per_clip, step_between_clips, fps)
            self.video_time_units = []
            for video_idx, vid_clips in tqdm(
                enumerate(self.video_clips.clips), total=len(self.video_clips.clips)
            ):
                video_path = self.video_clips.video_paths[video_idx]
                if (
                    all_fail_videos
                    and os.path.splitext(os.path.basename(video_path))[0]
                    not in self.fails_data
                ):
                    self.video_clips.labels.append([-1 for _ in vid_clips])
                    continue
                try:
                    t_unit = (
                        av.open(video_path, metadata_errors="ignore")
                        .streams[0]
                        .time_base
                    )
                    self.video_time_units.append(t_unit)
                except av.AVError:
                    print("Encountered av error...continuing")
                    pass
                t_fail = sorted(
                    self.fails_data[os.path.splitext(os.path.basename(video_path))[0]][
                        "t"
                    ]
                )
                t_fail = t_fail[len(t_fail) // 2]
                if (
                    t_fail < 0
                    or not 0.01
                    <= statistics.median(
                        self.fails_data[
                            os.path.splitext(os.path.basename(video_path))[0]
                        ]["rel_t"]
                    )
                    <= 0.99
                    or self.fails_data[
                        os.path.splitext(os.path.basename(video_path))[0]
                    ]["len"]
                    < 3.2
                    or self.fails_data[
                        os.path.splitext(os.path.basename(video_path))[0]
                    ]["len"]
                    > 30
                ):
                    self.video_clips.clips[video_idx] = torch.Tensor()
                    # self.video_clips.clips.pop(video_idx)
                    self.video_clips.resampling_idxs[video_idx] = torch.Tensor()
                    # self.video_clips.resampling_idxs.pop(video_idx)
                    self.video_clips.labels.append([])
                    continue
                prev_label = 0
                first_one_idx = len(vid_clips)
                first_two_idx = len(vid_clips)
                for clip_idx, clip in enumerate(vid_clips):
                    start_pts = clip[0].item()
                    end_pts = clip[-1].item()
                    t_start = float(t_unit * start_pts)
                    t_end = float(t_unit * end_pts)
                    label = 0
                    if t_start <= t_fail <= t_end:
                        label = 1
                    elif t_start > t_fail:
                        label = 2
                    if label == 1 and prev_label == 0:
                        first_one_idx = clip_idx
                    elif label == 2 and prev_label == 1:
                        first_two_idx = clip_idx
                        break
                    prev_label = label
                self.video_clips.labels.append(
                    [0 for i in range(first_one_idx)]
                    + [1 for i in range(first_one_idx, first_two_idx)]
                    + [2 for i in range(first_two_idx, len(vid_clips))]
                )
                if balance_fails_only and not val:
                    balance_idxs = []
                    counts = (
                        first_one_idx,
                        first_two_idx - first_one_idx,
                        len(vid_clips) - first_two_idx,
                    )
                    offsets = (
                        torch.LongTensor([0] + list(counts)).cumsum(0)[:-1].tolist()
                    )
                    ratios = (1, 0.93, 1 / 0.93)
                    labels = (0, 1, 2)
                    lbl_mode = max(labels, key=lambda i: counts[i])
                    for i in labels:
                        if i != lbl_mode and counts[i] > 0:
                            n_to_add = round(
                                counts[i]
                                * ((counts[lbl_mode] * ratios[i] / counts[i]) - 1)
                            )
                            tmp = list(range(offsets[i], counts[i] + offsets[i]))
                            random.shuffle(tmp)
                            tmp_bal_idxs = []
                            while len(tmp_bal_idxs) < n_to_add:
                                tmp_bal_idxs += tmp
                            tmp_bal_idxs = tmp_bal_idxs[:n_to_add]
                            balance_idxs += tmp_bal_idxs
                    if not balance_idxs:
                        continue
                    t = torch.cat(
                        (vid_clips, torch.stack([vid_clips[i] for i in balance_idxs]))
                    )
                    self.video_clips.clips[video_idx] = t
                    vid_resampling_idxs = self.video_clips.resampling_idxs[video_idx]
                    try:
                        t = torch.cat(
                            (
                                vid_resampling_idxs,
                                torch.stack(
                                    [vid_resampling_idxs[i] for i in balance_idxs]
                                ),
                            )
                        )
                        self.video_clips.resampling_idxs[video_idx] = t
                    except IndexError:
                        pass
                    self.video_clips.labels[-1] += [
                        self.video_clips.labels[-1][i] for i in balance_idxs
                    ]
            clip_lengths = torch.as_tensor([len(v) for v in self.video_clips.clips])
            self.video_clips.cumulative_sizes = clip_lengths.cumsum(0).tolist()
            labels_m1 = 0
            labels_0 = 0
            labels_1 = 0
            labels_2 = 0
            for label in self.video_clips.labels:
                if len(label) != 0:
                    if label[0] == -1:
                        labels_m1 += 1
                    if label[0] == 0:
                        labels_0 += 1
                    if label[0] == 1:
                        labels_1 += 1
                    if label[0] == 2:
                        labels_2 += 1
            print(labels_m1)
            print(labels_0)
            print(labels_1)
            print(labels_2)

        fns_removed = 0
        if fns_to_remove and not val:
            for i, video_path in enumerate(self.video_clips.video_paths):
                if fns_removed > len(self.video_clips.video_paths) // 4:
                    break
                video_path = os.path.splitext(os.path.basename(video_path))[0]
                if video_path in fns_to_remove:
                    fns_removed += 1
                    self.video_clips.clips[i] = torch.Tensor()
                    self.video_clips.resampling_idxs[i] = torch.Tensor()
                    self.video_clips.labels[i] = []
            clip_lengths = torch.as_tensor([len(v) for v in self.video_clips.clips])
            self.video_clips.cumulative_sizes = clip_lengths.cumsum(0).tolist()
            if kwargs["local_rank"] <= 0:
                print(
                    f"removed videos from {fns_removed} out of {len(self.video_clips.video_paths)} files"
                )
        # if not fails_path.startswith("PATH/TO/scenes"):
        for i, p in enumerate(self.video_clips.video_paths):
            self.video_clips.video_paths[i] = p.replace(
                "PATH/TO/scenes", os.path.dirname(fails_path)
            )
        self.debug_dataset = debug_dataset
        if debug_dataset:
            # self.video_clips = self.video_clips.subset([0])
            pass
        if data_proportion < 1:
            rng = random.Random()
            rng.seed(23719)
            lbls = self.video_clips.labels
            subset_idxs = rng.sample(
                range(len(self.video_clips.video_paths)),
                int(len(self.video_clips.video_paths) * data_proportion),
            )
            self.video_clips = self.video_clips.subset(subset_idxs)
            self.video_clips.labels = [lbls[i] for i in subset_idxs]
        self.sampled_clips = [False] * len(self.video_clips.clips)

    def trim_borders(self, img, fn):
        l, r = self.fails_borders[os.path.splitext(os.path.basename(fn))[0]]
        w = img.shape[2]  # THWC
        if l > 0 and r > 0:
            img = img[:, :, round(w * l) : round(w * r)]
        return img

    def __len__(self):
        return self.video_clips.num_clips()

    def compute_clip_times(self, video_idx, clip_idx):
        video_path = self.video_clips.video_paths[video_idx]
        video_path = os.path.join(
            self.fails_path, os.path.sep.join(video_path.rsplit(os.path.sep, 2)[-2:])
        )
        clip_pts = self.video_clips.clips[video_idx][clip_idx]
        start_pts = clip_pts[0].item()
        end_pts = clip_pts[-1].item()
        try:
            t_unit = av.open(video_path, metadata_errors="ignore").streams[0].time_base
        except Exception as e:
            print("error")
            print(e)
        t_start = float(t_unit * start_pts)
        t_end = float(t_unit * end_pts)
        return t_start, t_end

    def _get_video_t_unit(self, video_idx):
        video_path = self.video_clips.video_paths[video_idx]
        # video_path = os.path.join(self.fails_path, os.path.sep.join(video_path.rsplit(os.path.sep, 2)[-2:]))
        with av.open(video_path, metadata_errors="ignore") as container:
            return (
                container.streams.video[0].time_base,
                container.streams.video[0].average_rate,
            )

    def _compute_clip_times(self, video_idx, start_pts, end_pts, terminal_pts):
        video_t_unit, original_fps = self._get_video_t_unit(video_idx)
        original_fps = float(original_fps)
        video_fps = self.fps  # self.video_clips.video_fps[video_idx]
        start_time = start_pts * float(video_t_unit)
        end_time = end_pts * float(video_t_unit)
        terminal_time = float(terminal_pts * video_t_unit)
        start_frame = int(start_time * float(original_fps))
        end_frame = math.ceil(end_time * float(original_fps))
        terminal_frame = int(terminal_time * video_fps)
        if end_frame - start_frame < video_fps:
            end_frame += video_fps - (end_frame - start_frame)
        # end_frame = int(start_frame + (self.clip_len * self.fps))
        # if ret == 'frames':
        return start_frame, end_frame, original_fps, start_time, end_time
        # elif ret == 'times':
        # return start_time, end_time, original_fps

    def _read_video_features(
        self, video_path, video_idx, clip_idx, start_pts, end_pts, terminal_pts
    ):
        start_time = time.time()
        compressed_file = np.load(video_path)
        array = compressed_file["arr_0"]
        video = torch.from_numpy(array)
        end_time = time.time()
        # print(end_time - start_time)
        (
            start_frame,
            end_frame,
            original_fps,
            start_time,
            end_time,
        ) = self._compute_clip_times(video_idx, start_pts, end_pts, terminal_pts)
        video = video[
            start_frame : end_frame + 1
        ]  # since torch slicing has exclusive target

        # num_frames = video.shape[0] * (float(self.fps) / float(original_fps))
        # resampling_idx = self._resample_video_idx(num_frames, original_fps, self.fps)
        resampling_idx = self.video_clips.resampling_idxs[video_idx][clip_idx]
        resampling_idx = resampling_idx - resampling_idx[0]
        video = video[resampling_idx]
        return video, start_time, end_time

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
        video_name = video_path.split("/")[-1] + ".npz"
        video_path = os.path.join(self.base_features_path, video_name)
        clip_pts = self.video_clips.clips[video_idx][clip_idx]

        start_pts = clip_pts[0].item()
        end_pts = clip_pts[-1].item()
        terminal_pts = self.video_clips.clips[video_idx][-1][-1].item()
        video_clip, start_time, end_time = self._read_video_features(
            video_path, video_idx, clip_idx, start_pts, end_pts, terminal_pts
        )
        return video_clip, start_time, end_time

    def __getitem__(self, idx):

        video_idx, clip_idx = self.video_clips.get_clip_location(idx)
        video, start_time, end_time = self._get_clip(idx)
        video_path = self.video_clips.video_paths[video_idx]
        # print(video_path)
        try:
            # labels = self.video_clips.labels[video_idx]
            # if len(labels) == 0:
            #     print('here')
            label = self.video_clips.labels[video_idx][clip_idx]
            if self.anticipate_label:
                video_path = self.video_clips.video_paths[video_idx]
                t_fail = statistics.median(
                    self.fails_data[os.path.splitext(os.path.basename(video_path))[0]][
                        "t"
                    ]
                )
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
        out["features"] = video
        out["pure_nr_frames"] = int(self.clip_len * self.fps)
        out["label"] = label
        # if self.get_clip_times:
        out["times"] = (start_time, end_time)
        video_name = video_path.split("/")[-1].replace(".mp4", "")
        t_time = self.fails_data[video_name]["t"]
        out["t"] = t_time
        out["rel_t"] = self.fails_data[video_name]["rel_t"]
        out["video_idx"] = video_idx
        out["video_name"] = video_name
        return out
        # return video, label, (video_path, t_start, t_end, *other)


def get_video_loader(args):
    # args = Namespace(**kwargs)
    args.fails_video_list = None
    if args.val:
        args.fails_path = os.path.join(args.fails_path, "val")
        args.kinetics_path = os.path.join(args.kinetics_path, "val")
    else:
        args.fails_path = os.path.join(args.fails_path, "train")
        # args.kinetics_path = os.path.join(args.kinetics_path, 'train')
    if args.fails_action_split:
        args.fails_path = None
        args.fails_video_list = torch.load(
            os.path.join(args.dataset_path, "fails_action_split.pth")
        )["val" if args.val else "train"]
    DEBUG = False
    datasets = []
    samplers = []
    for fps in args.fps_list:
        clips = None
        t_units = None
        args.fps = fps
        args.step_between_clips = round(args.step_between_clips_sec * fps)
        if args.val:
            cache_path = os.path.join(
                args.dataset_path,
                "{3}{2}{1}{0}{4}_videoclips_clean.pth".format(
                    "val" if args.val else "train",
                    f'fails_only_{"all_" if args.all_fail_videos else ""}'
                    if args.fails_only
                    else "",
                    "bal_" if (args.balance_fails_only and not DEBUG) else "",
                    "actions_" if args.fails_action_split else "",
                    f"{args.fps}fps",
                ),
            )
        else:
            cache_path = os.path.join(
                args.dataset_path,
                "{3}{2}{1}{0}{4}_videoclips.pth".format(
                    "val" if args.val else "train",
                    f'fails_only_{"all_" if args.all_fail_videos else ""}'
                    if args.fails_only
                    else "",
                    "bal_" if (args.balance_fails_only and not DEBUG) else "",
                    "actions_" if args.fails_action_split else "",
                    f"{args.fps}fps",
                ),
            )
        t_units_cache_path = os.path.join(
            args.dataset_path, "time_units_%s.pth" % ("val" if args.val else "train")
        )
        if args.cache_dataset and os.path.exists(cache_path):
            clips = torch.load(cache_path)
            t_units = torch.load(t_units_cache_path)
            if args.local_rank <= 0:
                print(f"Loaded dataset from {cache_path}")
        fns_to_remove = None
        if args.remove_fns == "action_based":
            fns_to_remove = torch.load("PATH/TO/fails_remove_fns.pth")["action_remove"]
        elif args.remove_fns == "random":
            fns_to_remove = torch.load("PATH/TO/fails_remove_fns.pth")["random_remove"]

        args.transform = test_transform if args.val else train_transform

        dataset = KineticsAndFails(
            video_clips=clips,
            fns_to_remove=fns_to_remove,
            t_units=t_units,
            **vars(args),
        )
        # if not args.val:
        print(f"Dataset contains {len(dataset)} items")
        if (
            args.cache_dataset and args.local_rank <= 0 and clips is None
        ):  # and not args.fails_only
            torch.save(dataset.video_clips, cache_path)
            torch.save(dataset.video_time_units, t_units_cache_path)
        if args.val:
            sampler = UniformClipSampler(
                dataset.video_clips,
                1000000 if args.sample_all_clips else args.clips_per_video,
            )
        else:
            sampler = RandomClipSampler(
                dataset.video_clips,
                1000000 if args.sample_all_clips else args.clips_per_video,
            )
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
        sampler=sampler,
        pin_memory=True,
        drop_last=False,
    )
