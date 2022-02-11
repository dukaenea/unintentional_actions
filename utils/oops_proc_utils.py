# @Author: Enea Duka
# @Date: 5/23/21

import json
from os import walk
import os.path as ops
from csv import DictReader, DictWriter


def filter_unlabeled():
    train_json_file = '/BS/unintentional_actions/nobackup/oops/oops_dataset/annotations/transition_times.json'
    train_video_dir = '/BS/unintentional_actions/nobackup/oops/oops_dataset/oops_video/train'
    val_video_dir = '/BS/unintentional_actions/nobackup/oops/oops_dataset/oops_video/val'
    base_split_path = '/BS/unintentional_actions/work/data/oops/splits'

    train_files = get_all_files_in_directory(train_video_dir)
    val_files = get_all_files_in_directory(val_video_dir)

    with open(train_json_file) as f:
        borders = json.load(f)
        train_labeled = 0
        val_labeled = 0

        train_videos = []
        val_videos = []

        for key, sample in borders.items():
            border = sample['t']
            rel_border = sample['rel_t']
            if border.count(-1) != len(border):
                valid_t = [ts for ts in border if ts != -1]
                valid_t_rel = [ts for ts in rel_border if ts >= 0]
                try:
                    t = sum(valid_t)/len(valid_t)
                    t_rel = sum(valid_t_rel)/len(valid_t_rel)
                except Exception:
                    print('here')
                vid_len = sample['len']
                file_name = key + '.mp4'
                if file_name in train_files:
                    train_videos.append({'filename': file_name})
                if file_name in val_files:
                    val_videos.append({'filename': file_name})

            # else:
            #     t = -1
            #     t_rel = -1
            # vid_len = sample['len']
            # file_name = key + '.mp4'
            # if file_name in train_files:
            #     train_videos.append({'filename': file_name, 't': t, 't_rel': t_rel, 'len': vid_len})
            # if file_name in val_files:
            #     val_videos.append({'filename': file_name, 't': t, 't_rel': t_rel, 'len': vid_len})


        with open(ops.join(base_split_path, 'train_all.csv'), 'w', newline='') as f:
            dwriter = DictWriter(f, train_videos[0].keys())
            dwriter.writeheader()
            dwriter.writerows(train_videos)

        with open(ops.join(base_split_path, 'val_all.csv'), 'w', newline='') as f:
            dwriter = DictWriter(f, val_videos[0].keys())
            dwriter.writeheader()
            dwriter.writerows(val_videos)


def get_all_files_in_directory(directory):
    _, _, files = next(walk(directory))
    return files

def get_unlabeled():
    train_json_file = '/BS/unintentional_actions/nobackup/oops/oops_dataset/annotations/transition_times.json'
    train_video_dir = '/BS/unintentional_actions/nobackup/oops/oops_dataset/oops_video/train'
    val_video_dir = '/BS/unintentional_actions/nobackup/oops/oops_dataset/oops_video/val'
    base_split_path = '/BS/unintentional_actions/work/data/oops/splits'

    with open(train_json_file) as f:
        borders = json.load(f)
        video_names = list(borders.keys())

        train_files = get_all_files_in_directory(train_video_dir)
        val_files = get_all_files_in_directory(val_video_dir)
        train_videos = []
        val_videos = []

        for train_file in train_files:
            # if train_file[:-4] not in video_names:
            train_videos.append({'filename': train_file[:-4]})
        for val_file in val_files:
            # if val_file[:-4] not in video_names:
            val_videos.append({'filename': val_file[:-4]})

        with open(ops.join(base_split_path, 'train_all.csv'), 'w', newline='') as f:
            dwriter = DictWriter(f, train_videos[0].keys())
            dwriter.writeheader()
            dwriter.writerows(train_videos)

        with open(ops.join(base_split_path, 'val_all.csv'), 'w', newline='') as f:
            dwriter = DictWriter(f, val_videos[0].keys())
            dwriter.writeheader()
            dwriter.writerows(val_videos)


if __name__ == '__main__':
    # filter_unlabeled()
    get_unlabeled()