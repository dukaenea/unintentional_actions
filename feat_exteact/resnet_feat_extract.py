# @Author: Enea Duka
# @Date: 4/22/21
import sys
sys.path.append('/BS/unintentional_actions/work/unintentional_actions')
import torch
from tqdm import tqdm
import os
import numpy as np
from dataloaders.kinetics_loader import KineticsDataset
from dataloaders.rareacts_loader import RareactsDataset
from dataloaders.oops_loader_simple import SimpleOopsDataset
from torch.utils.data import DataLoader

from models.resnet50 import build_resnet_50
from models.resnet50_21k import create_resnet50_21k


def extract_resnet_features(dataset_name, mode):
    # feat_path = '/BS/unintentional_actions/work/data/%s/resnet_feats/full_kinetics/%s/' % (dataset_name, mode)
    feat_path = '/BS/unintentional_actions/nobackup/oops/resnet_feats/%s_normalized/' % mode
    # feat_path = '/BS/unintentional_actions/work/data/oops/resnet_feats/unlabeled/%s/' % mode
    # feat_path = '/BS/unintentional_actions/work/data/rareact/resnet_feats/positive_negative/%s' % mode
    if dataset_name == 'kinetics':
        # use the ImageNet stats
        dataset = KineticsDataset(mode, 25, True, False,
                                  {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]},
                                  feat_ext=True)
    elif dataset_name == 'rareact':
        dataset = RareactsDataset(mode, 25, False, False,
                                  {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]})
    elif dataset_name == 'oops':
        dataset = SimpleOopsDataset(mode, 25, 'frames', True,
                                    {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]})


    dataloader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=0,
        shuffle=False
    )
    # model = build_resnet_50()
    model = create_resnet50_21k(pretrained=True)
    model.cuda()
    model.eval()

    with torch.no_grad():
        for idx, data in enumerate(tqdm(dataloader)):
            # frames = data['features'][0].permute(1, 0, 2, 3)
            if 'features' not in data.keys():
                continue
            frames = data['features'][0]
            # label_folder_path = os.path.join(feat_path, data['label_text'][0])
            # array_path = os.path.join(label_folder_path, data['video_name'][0]) + '.npz'
            array_path = os.path.join(feat_path, data['filename'][0].replace('mp4', 'npz'))
            if os.path.isfile(array_path):
                # print('Existing')
                continue
            # print('Saving: %d' % idx)
            # use the number of frames as the batch size to feed to resnet
            output = model(frames.cuda())
            # keep the spatial axes with a probability p
            p = .02
            sample = np.random.uniform(low=.0, high=1.0, size=None)
            # if idx % 10 != 0:
            #     # avg pool over the spatial axes
            #     output = output.mean(3).mean(2)
            output_np = output.cpu().detach().numpy()
            # float64 consumes a lot of space, using float32 to save space
            # output_np = output_np.astype(np.float16
            # if not os.path.exists(label_folder_path):
            #     os.makedirs(label_folder_path)


            np.savez_compressed(array_path, output_np)
            # torch.save(output, array_path)

            # if idx == 5:
            #     break



if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    extract_resnet_features('oops', 'train')
