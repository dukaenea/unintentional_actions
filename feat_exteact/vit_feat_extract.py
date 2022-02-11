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
from dataloaders.pedestrian_loader import PedestrianDataset
from dataloaders.avenue_loader import AvenueDataset
from torch.utils.data import DataLoader
from models.vit import create_vit_model
from utils.logging_setup import logger, setup_logger_path
from dataloaders.ucf_crime_video_loader import UCFCrimeVideoLoader
from utils.arg_parse import opt
from datetime import datetime
import warnings



def extract_resnet_features(dataset_name, mode):
    feat_path = '/BS/unintentional_actions/work/data/%s/vit_features/%s/' % (dataset_name, mode)
    # feat_path = '/BS/feat_augm/nobackup/oops/resnet_feats/%s/' % mode
    feat_path = '/BS/unintentional_actions/work/data/oops/vit_features/%s_unlabeled/' % mode
    feat_path = '/BS/unintentional_actions/work/data/oops/vit_features/%s_all' % mode
    # feat_path = '/BS/unintentional_actions/work/data/rareact/vit_features/positive_negative/%s' % mode
    # feat_path = '/BS/unintentional_actions/nobackup/ucf_crime/vit_features/%s' % ('Anomaly_Videos-200' if mode=='train' else 'Anomaly_Videos-200')
    if dataset_name == 'kinetics':
        # use the ImageNet stats
        dataset = KineticsDataset(mode, 25, True, False,
                                  {'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5]},
                                  feat_ext=True)
    elif dataset_name == 'rareact':
        dataset = RareactsDataset(mode, 25, True, False,
                                  {'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5]})
    elif dataset_name == 'oops':
        dataset = SimpleOopsDataset(mode, 25, 'frames', True,
                                    {'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5]})
    elif dataset_name == 'pedestrian':
        dataset = PedestrianDataset(mode, gray_scale=False)
        feat_path = '/BS/unintentional_actions/nobackup/ucsd/UCSD_Anomaly_Dataset.v1p2/vit_features/%s' % mode
    elif dataset_name == 'avenue':
        dataset = AvenueDataset(mode, (224, 224), '/BS/unintentional_actions/nobackup/avenue/avenue/training' if mode=='train' else '/BS/unintentional_actions/nobackup/avenue/avenue/testing')
        feat_path = '/BS/unintentional_actions/nobackup/avenue/avenue/vit_features/%s' % mode
    elif dataset_name == 'ucf_crime':
        dataset = UCFCrimeVideoLoader(mode, load_frames=True)

    opt.model_name = 'ViT_VideoLongformer_MLP'
    opt.sfx = str('unint_act.layers')
    opt.log_name = 'ln'
    opt.sfx = 'sfffx'
    opt.viz = False


    setup_logger_path()

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=0,                # for video_path in self.video_list:

        shuffle=False
    )
    model = create_vit_model()
    model.eval()

    total_memory, used_memory, free_memory = map(
        int, os.popen('free -t -m').readlines()[-1].split()[1:])

    # Memory usage
    logger.debug("RAM memory: %f " % total_memory)

    with torch.no_grad():
        for idx, data in enumerate(tqdm(dataloader)):
            # if idx < 870:
            #     continue
            # frames = data['features'][0].permute(1, 0, 2, 3)
            if 'features' not in data.keys():
                continue
            # total_memory, used_memory, free_memory = map(
            #     int, os.popen('free -t -m').readlines()[-1].split()[1:])

            # Memory usage
            # logger.debug("RAM memory used: %f " % round((used_memory / total_memory) * 100, 2))
            frames = data['features'][0]
            if torch.isnan(frames).any():
                print('Found Nan!!!')
            # print(frames.shape)
            # label_folder_path = os.path.join(feat_path, data['label_text'][0])
            # array_path = os.path.join(label_folder_path, data['video_name'][0]) + '.npz'
            # logger.debug(str(frames.shape))
            array_path = os.path.join(feat_path, data['filename'][0]+'.npz')
            if os.path.isfile(array_path):
                # print('Existing')
                continue
            # print('Saving: %d' % idx)
            # use the number of frames as the batch size to feed to resnet
            fpb = 128
            # print(frames.shape)
            # if frames.shape[0] > fpb:
            #     X = torch.stack(list(torch.split(frames, fpb))[:-1])
            #     output = []
            #     for _x in X:
            #         output.append(model(_x.cuda()).cpu())
            #         _x.cpu()
            #         torch.cuda.empty_cache()
            #     remainder = frames.shape[0] % fpb
            #     output.append(model(frames[-remainder:].cuda()).cpu())
            #     torch.cuda.empty_cache()
            #     output = torch.cat(output, dim=0)
            # else:
            output = model(frames)
            if torch.isnan(output).any():
                print('Found Nan!!!')
            # keep the spatial axes with a probability p
            # p = .02
            # sample = np.random.uniform(low=.0, high=1.0, size=None)
            # if idx % 10 != 0:
            #     # avg pool over the spatial axes
            #     output = output.mean(3).mean(2)
            output_np = output.cpu().detach().numpy()
            # float64 consumes a lot of space, using float32 to save space
            # output_np = output_np.astype(np.float16
            if not os.path.exists('/'.join(array_path.split('/')[:-1])):
                os.makedirs('/'.join(array_path.split('/')[:-1]))


            np.savez_compressed(array_path, output_np)
            # torch.save(output, array_path)

            # if idx == 5:
            #     break



if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    # print(IMAGENET_DEFAULT_STD)
    # print(IMAGENET_DEFAULT_MEAN)
    extract_resnet_features('oops', 'train')
