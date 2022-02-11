
# @Author: Enea Duka
# @Date: 10/30/21
import sys
import warnings
sys.path.append('/BS/unintentional_actions/work/unintentional_actions')
from utils.arg_parse import opt
from dataloaders.oops_loader_simple import SimpleOopsDataset
from torch.utils.data import DataLoader
from swin.swin_transformer import SwinTransformer3D
from os import path
from tqdm import tqdm
import torch.nn as nn
import numpy as np
import torch
from datetime import datetime
import pandas as pd
from utils.logging_setup import setup_logger_path
from swin.swin_feat_loader import SwinOopsFeats
from utils.util_functions import Meter
from utils.model_saver import ModelSaver
import os
from utils.logging_setup import logger
from utils.util_functions import Precision
import copy


base_swin_feat_path = '/BS/unintentional_actions/work/data/oops/swin_feats'
swin_ptr_path = '/BS/unintentional_actions/work/storage/models/swin/swin_base_patch244_window877_kinetics400_22k.pth'


class MLP(nn.Module):
    def __init__(self, in_channels, num_classes, dropout):
        super(MLP, self).__init__()
        self.mlp_head = nn.Sequential(
            nn.Linear(in_channels, in_channels//2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(in_channels//2, in_channels//4),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(in_channels//4, num_classes),
        )

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.mlp_head(x)
        return x




def ext_feats(mode):
    swin_feat_path = path.join(base_swin_feat_path, mode) + '_new'
    dataset = SimpleOopsDataset(mode, 30, 'frames', True, norm_statistics={'mean': [123.675, 116.28, 103.53],
                                                                                'std': [58.395, 57.12, 57.375]})

    avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

    dataloader = DataLoader(dataset=dataset,
                              num_workers=32,
                              batch_size=1,
                              collate_fn=dataset.vid_to_32_frame_clips)

    model = SwinTransformer3D(
        depths=(2, 2, 18, 2),
        embed_dim=128,
        num_heads=(4, 8, 16, 32),
        patch_size=(2, 4, 4),
        drop_path_rate=0.2,
        pretrained=swin_ptr_path
    )
    model.cuda()
    model = nn.DataParallel(model)
    model.eval()

    for idx, data in enumerate(tqdm(dataloader)):
        filename = data['filename']
        if 'features' not in list(data.keys()):
            continue
        video = data['features'][0]
        feat_path = path.join(swin_feat_path, filename[0]) + '.npz'
        # if path.isfile(feat_path):
        #     continue

        video = video.permute(0, 2, 1, 3, 4)
        split_clips = 5
        splits_in_video = video.shape[0] // split_clips
        clips_remaining = video.shape[0] - (split_clips * splits_in_video)
        all_swin_feats = []
        if splits_in_video > 0:
            video_splits = torch.stack(list(torch.split(video[:splits_in_video*split_clips], split_clips)), dim=0)
            for split in video_splits:
                all_swin_feats.append(model(split.cuda()).detach().cpu())
        if clips_remaining > 0:
            all_swin_feats.append(model(video[split_clips * splits_in_video:(split_clips * splits_in_video)+clips_remaining].cuda()).detach().cpu())

        # swin_feats = model(video)
        swin_feats = torch.cat(all_swin_feats, dim=0)
        swin_feats = avg_pool(swin_feats).squeeze()
        swin_feats = swin_feats.numpy()
        #
        # feat_path = path.join(swin_feat_path, filename[0]) + '.npz'
        # np.savez_compressed(feat_path, swin_feats)

def test(model, test_loader, loss_fn):
    model.eval()
    prec = Precision('val')
    meter = Meter(mode='val', name='loss')

    with torch.no_grad():
        for idx, data in enumerate(tqdm(test_loader)):
            features = data['features'].squeeze()
            labels = data['labels'].squeeze()
            # total += features.shape[0]
            # zeros += labels.count(0)
            # ones += labels.count(1)
            # twos += labels.count(2)
            out = model(features.cuda())
            loss = loss_fn(out, labels.cuda())

            meter.update(loss.item(), features.shape[0])
            prec.update_probs_sfx(out, labels.cuda())


    logger.debug('Val Loss: %f' % meter.avg)
    meter.reset()

    logger.debug('Val Acc: %f' % prec.top1(report_pca=False))
    # return {'top1': (1 / meter.avg)}
    return {'top1': prec.top1()}


def train(model, train_loader, test_loader, optimizer, loss_fn):

    loss_meter = Meter('train')
    model_saver = ModelSaver(path=os.path.join(opt.storage, 'models', opt.dataset, opt.model_name, opt.sfx, opt.log_name, 'val'))

    logger.debug('Starting training for %d epochs:' % opt.epochs)

    test(model, test_loader, loss_fn)

    for epoch in range(opt.epochs):
        model.train()

        for idx, data in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            features = data['features'].squeeze()
            labels = data['labels'].squeeze()
            # total += features.shape[0]
            # zeros += labels.count(0)
            # ones += labels.count(1)
            # twos += labels.count(2)
            out = model(features.cuda())
            try:
                loss = loss_fn(out, labels.cuda())
            except Exception as e:
                print(features.shape)
                print(out.shape)

            loss_meter.update(loss.item(), features.shape[0])
            loss.backward()
            optimizer.step()
        logger.debug('Train Loss: %f' % loss_meter.avg)
        loss_meter.reset()
        check_val = None

        if opt.test_freq and epoch % opt.test_freq == 0:
            check_val = test(model, test_loader, loss_fn)

        if opt.save_model and epoch % opt.save_model == 0 and check_val:
            if model_saver.check(check_val):
                save_dict = {'epoch': epoch,
                             'state_dict': copy.deepcopy(model.state_dict()),
                             'optimizer': copy.deepcopy(optimizer.state_dict().copy())}
                model_saver.update(check_val, save_dict, epoch)

            model_saver.save()

def unint_action_class():
    opt.dataset = 'oops_swin'
    opt.model_name = 'Swin_MLP'
    opt.viz = False
    opt.test = True
    opt.num_workers = 32
    opt.batch_size = 256
    opt.sfx = str('%s.unint_act' % (opt.dataset))
    opt.save_model = 1
    opt.test_val = True
    opt.epochs = 50
    opt.gpu_parallel = True
    opt.use_tqdm = True

    opt.optim = 'adam'
    opt.momentum = 0.9
    opt.lr = 1e-3
    opt.backbone_lr_factor = 1
    opt.cos_decay_lr_factor = 0.01
    opt.weight_decay = 1e-4
    opt.test_freq = 1
    opt.save_model = 1
    opt.pretrained = True
    opt.log_name = 'lr:%f~ep:%d~bs:%d~ptr:%s_cntr_loss' % (opt.lr, opt.epochs, opt.batch_size, str(opt.pretrained))
    opt.viz_env = '%s.%s%s_%s.' % (opt.model_name, opt.temp_learning_dataset_name, opt.env_pref, opt.sfx)
    opt.sfx = str('%s.unint_act.time%s_cntr_loss' % (opt.dataset,datetime.now().strftime('%Y%m%d-%H%M%S')))

    setup_logger_path()

    train_set = SwinOopsFeats('train')
    val_set = SwinOopsFeats('val')

    train_loader = DataLoader(dataset=train_set,
                              num_workers=opt.num_workers,
                              batch_size=opt.batch_size,
                              shuffle=True,
                              pin_memory=True,
                              drop_last=True)
    val_loader = DataLoader(dataset=val_set,
                            num_workers=opt.num_workers,
                            batch_size=opt.batch_size,
                            shuffle=False,
                            pin_memory=True)

    model = MLP(in_channels=1024, dropout=0.1, num_classes=3)
    model.cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor([25042/16916, 25042/5395, 25042/25042]).cuda())

    train(model, train_loader, val_loader, optimizer, loss_fn)



if __name__ == '__main__':
    # ext_feats('val')
    unint_action_class()