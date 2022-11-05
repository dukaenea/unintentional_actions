# @Author: Enea Duka
# @Date: 5/3/21
import sys

sys.path.append("/BS/unintentional_actions/work/unintentional_actions")
import os
import argparse
import time
from dataloaders.kinetics_loader import KineticsDataset
from dataloaders.rok_loader import ROKDataset
from dataloaders.rareacts_loader import RareactsDataset
from dataloaders.ucf_crime_video_loader import UCFCrimeVideoLoader
from torch.utils.data import DataLoader
from tqdm import tqdm

# from models.pm_vtn import create_model, get_froze_trn_optimizer
from models.pm_vtn import create_model
from models.res3d_18 import create_r3d
from transformer2x.vtn import create_model_trn_2x
from models.resnet50 import build_resnet_50
from rep_learning.train import train
from utils.logging_setup import setup_logger_path
from utils.arg_parse import opt
from dataloaders.my_oops_loader import get_video_loader
from dataloaders.trn_ad_loader import get_anomaly_loader
from torch.utils.data import ConcatDataset
import torch
from datetime import datetime
import warnings
from rep_learning.feat_transformer import fast_forward
from dataloaders.ucf_crime_video_loader import UCFCrimeVideoLoader
from models.pm_vtn import MILRegLoss
from dataloaders.ucf_crime_loader import get_crime_video_loader


def learn_representation():
    # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/all/VideoLongformer/all.rep_learning.tag:speed&motion.layers3.attn_win32.classes9.time20210713-012808/val/top1/VideoLongformer__all.rep_learning.tag:speed&motion.layers3.attn_win32.classes9.time20210713-012808_v0.5753_ep82.pth.tar'
    # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/all/VideoLongformer/all.rep_learning.tag:speed&motion.layers3.attn_win32.classes9.time20210805-084035/val/top1/VideoLongformer__all.rep_learning.tag:speed&motion.layers3.attn_win32.classes9.time20210805-084035_v0.5664_ep93.pth.tar'
    # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/all/VideoLongformer/all.rep_learning.tag:speed&motion.layers3.attn_win32.classes9.time20210713-012808/val/top1/VideoLongformer__all.rep_learning.tag:speed&motion.layers3.attn_win32.classes9.time20210713-012808_v0.5753_ep82.pth.tar'
    # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/all/VideoLongformer/all.rep_learning.tag:speed&motion.layers3.attn_win32.classes9.time20210817-010213/val/top1/VideoLongformer__all.rep_learning.tag:speed&motion.layers3.attn_win32.classes9.time20210817-010213_v0.4974_ep17.pth.tar'
    # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/all/VideoLongformer/all.rep_learning.tag:speed&motion.layers3.attn_win32.classes4.time20210817-153834/val/top1/VideoLongformer__all.rep_learning.tag:speed&motion.layers3.attn_win32.classes4.time20210817-153834_v0.7138_ep14.pth.tar'
    # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/all/VideoLongformer/all.rep_learning.tag:speed&motion.layers3.attn_win32.classes9.time20210713-012808/val/top1/VideoLongformer__all.rep_learning.tag:speed&motion.layers3.attn_win32.classes9.time20210713-012808_v0.5753_ep82.pth.tar'
    # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/all/VideoLongformer/all.rep_learning.tag:speed&motion.layers3.attn_win32.classes8.time20210822-192435/val/top1/VideoLongformer__all.rep_learning.tag:speed&motion.layers3.attn_win32.classes8.time20210822-192435_v0.6154_ep81.pth.tar'
    # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/all/VideoLongformer/all.rep_learning.tag:speed&motion.layers3.attn_win32.classes8.time20210919-181733/val/top1/VideoLongformer__all.rep_learning.tag:speed&motion.layers3.attn_win32.classes8.time20210919-181733_v0.4327_ep4.pth.tar'
    # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/all/VideoLongformer/all.rep_learning.tag:speed&motion.layers3.attn_win32.classes8.time20210920-140805/val/top1/VideoLongformer__all.rep_learning.tag:speed&motion.layers3.attn_win32.classes8.time20210920-140805_v0.4972_ep27.pth.tar'
    # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/all/VideoLongformer/all.rep_learning.tag:speed&motion.layers3.attn_win32.classes8.time20210928-173557/val/top1/VideoLongformer__all.rep_learning.tag:speed&motion.layers3.attn_win32.classes8.time20210928-173557_v0.2648_ep0.pth.tar'
    # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/all/VideoLongformer/all.rep_learning.tag:speed&motion.layers3.attn_win32.classes4.time20210930-215645/val/top1/VideoLongformer__all.rep_learning.tag:speed&motion.layers3.attn_win32.classes4.time20210930-215645_v1.0000_ep26.pth.tar'
    # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/all/VideoLongformer/all.rep_learning.tag:speed&motion.layers3.attn_win32.classes8.time20211001-205310/val/top1/VideoLongformer__all.rep_learning.tag:speed&motion.layers3.attn_win32.classes8.time20211001-205310_v0.8817_ep2.pth.tar'
    # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/all/VideoLongformer/all.rep_learning.tag:speed&motion.layers3.attn_win32.classes4.time20211002-125702/val/top1/VideoLongformer__all.rep_learning.tag:speed&motion.layers3.attn_win32.classes4.time20211002-125702_v1.0000_ep0.pth.tar'
    # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/all/VideoLongformer/all.rep_learning.tag:speed&motion.layers3.attn_win32.classes4.time20211002-162810/val/top1/VideoLongformer__all.rep_learning.tag:speed&motion.layers3.attn_win32.classes4.time20211002-162810_v0.5286_ep1.pth.tar'
    # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/all/VideoLongformer/all.rep_learning.tag:speed&motion.layers3.attn_win32.classes8.time20211003-203856/val/top1/VideoLongformer__all.rep_learning.tag:speed&motion.layers3.attn_win32.classes8.time20211003-203856_v0.4206_ep2.pth.tar'
    # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/oops/VideoLongformer_MLP/oops.unint_act.classification.layers3.attn_win32.time20210713-093910_cntr_loss/lr:0.000100~ep:40~bs:128~win:32~b_lr:1.000000~ptr:True_cntr_loss/val/top1/VideoLongformer_MLP__oops.unint_act.classification.layers3.attn_win32.time20210713-093910_cntr_loss_v0.6444_ep37.pth.tar'
    # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/all/VideoLongformer/all.rep_learning.tag:speed&motion.layers6.attn_win32.classes9.time20210803-113354/val/top1/VideoLongformer__all.rep_learning.tag:speed&motion.layers6.attn_win32.classes9.time20210803-113354_v0.5919_ep79.pth.tar'
    # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/all/VideoLongformer/all.rep_learning.tag:speed&motion.layers1.attn_win32.classes8.time20211030-084423/val/top1/VideoLongformer__all.rep_learning.tag:speed&motion.layers1.attn_win32.classes8.time20211030-084423_v0.4937_ep98.pth.tar'
    # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/all/VideoLongformer/all.rep_learning.tag:speed&motion.layers3.attn_win4.classes9.time20210803-142545/val/top1/VideoLongformer__all.rep_learning.tag:speed&motion.layers3.attn_win4.classes9.time20210803-142545_v0.5688_ep63.pth.tar'
    # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/all/VideoLongformer/all.rep_learning.tag:speed&motion.layers3.attn_win16.classes8.time20211103-145658/val/top1/VideoLongformer__all.rep_learning.tag:speed&motion.layers3.attn_win16.classes8.time20211103-145658_v0.5399_ep25.pth.tar'

    # opt.resnet_ptr_path = ''
    #
    # opt.run_on_mpg = False
    #
    # if opt.run_on_mpg:
    #     opt.fails_path = '/u/eneaduka/datasets/oops/oops_dataset/oops_video'
    #     opt.dataset_path = '/u/eneaduka/datasets/oops'
    #     opt.storage = '/u/eneaduka/storage'
    #     opt.log_save_dir = '/u/eneaduka/logs'
    #
    #
    # opt.dataset = 'all'
    # opt.rep_data_level = 'features'
    # opt.rep_backbone = 'vit'
    # opt.backbone = 'r3d_18'
    # opt.speed_and_motion = True
    # opt.num_classes = 8
    # opt.use_crf = False
    #
    # opt.embed_dim = 768
    # opt.intermediate_size = 3072
    # opt.hidden_dim = 512
    # # opt.mlp_dim = 300
    #
    # opt.tag = 'speed&motion'
    # opt.num_hidden_layers = 3
    # opt.attention_window = [32] * opt.num_hidden_layers
    #
    # # opt.attention_probs_dropout_prob = 0
    # # opt.hidden_dropout_prob = 0
    # # opt.mlp_dropout = 0.3
    #
    # opt.model_name = 'VideoLongformer'
    # opt.viz = False
    # opt.test = True
    # opt.num_workers = 16
    # opt.batch_size = 1
    opt.sfx = str(
        "%s.rep_learning.tag:%s.layers%d.attn_win%d.classes%d_rok_class_per_trn.task_%s"
        % (
            opt.dataset,
            opt.tag,
            opt.num_hidden_layers,
            opt.attention_window[0],
            opt.num_classes,
            opt.task,
        )
    )
    # opt.save_model = 1
    # opt.test_val = True
    # opt.epochs = 100
    # opt.gpu_parallel = True
    # opt.use_tqdm = True
    # opt.spat_temp = False
    # opt.contrastive_loss = False
    # opt.rep_learning = True
    # opt.pretrained = False
    # opt.transformation_groups = 'speed&motion'
    # opt.use_frame_encoder = False
    #
    # opt.multi_scale = False
    #
    # opt.optim = 'adam'
    # opt.momentum = 0.9
    # opt.lr = 1e-4
    # opt.backbone_lr_factor = 1
    # opt.cos_decay_lr_factor = 0.01
    # opt.weight_decay = 1e-4
    # opt.test_freq = 1
    # opt.save_model = 1
    opt.viz_env = "%s.%s%s_%s." % (
        opt.model_name,
        opt.temp_learning_dataset_name,
        opt.env_pref,
        opt.sfx,
    )
    opt.sfx = str(
        "%s.rep_learning.tag:%s.layers%d.attn_win%d.classes%d.time%s"
        % (
            opt.dataset,
            opt.tag,
            opt.num_hidden_layers,
            opt.attention_window[0],
            opt.num_classes,
            datetime.now().strftime("%Y%m%d-%H%M%S"),
        )
    )
    #
    # opt.debug = False
    #
    # if opt.debug:
    #     opt.num_workers = 0
    #     opt.batch_size = 5
    #     opt.save_model = False
    #     opt.epochs = 1
    #     opt.viz = False

    setup_logger_path()

    train_set, val_set, test_set = None, None, None
    if opt.dataset == "kinetics":
        train_set = KineticsDataset(
            "train",
            fps=25,
            fpc=32,
            spat_crop=True,
            hflip=False,
            norm_statistics={
                "mean": [0.43216, 0.394666, 0.37645],
                "std": [0.22803, 0.22145, 0.216989],
            },
            feat_ext=True,
            data_level=opt.rep_data_level,
            feat_set="%s_feats" % opt.rep_backbone,
        )
        val_set = KineticsDataset(
            "val",
            fps=25,
            fpc=32,
            spat_crop=True,
            hflip=False,
            norm_statistics={
                "mean": [0.43216, 0.394666, 0.37645],
                "std": [0.22803, 0.22145, 0.216989],
            },
            feat_ext=True,
            data_level=opt.rep_data_level,
            feat_set="%s_feats" % opt.rep_backbone,
        )
    elif opt.dataset == "rareact":
        pass
    elif opt.dataset == "oops":
        pass
    elif opt.dataset == "all":
        train_set = ROKDataset(
            "train",
            spat_scale=True,
            size=224,
            spat_crop=True,
            load_frames=True if opt.backbone == "r3d_18" else False,
        )
        val_set = ROKDataset(
            "val",
            spat_scale=True,
            size=224,
            spat_crop=True,
            load_frames=True if opt.backbone == "r3d_18" else False,
        )
    elif opt.dataset == "ucf_crime":
        train_set = UCFCrimeVideoLoader("train", load_frames=False)
        val_set = UCFCrimeVideoLoader("val", load_frames=False)

    if opt.dataset == "all":
        train_loader = DataLoader(
            train_set,
            num_workers=opt.num_workers,
            batch_size=opt.batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=train_set.speed_and_motion_collate_fn
            if not opt.multi_scale
            else train_set.video_level_speed_and_motion_collate_fn,
        )

        val_loader = DataLoader(
            val_set,
            num_workers=opt.num_workers,
            batch_size=opt.batch_size,
            shuffle=False,
            drop_last=True,
            collate_fn=val_set.speed_and_motion_collate_fn
            if not opt.multi_scale
            else train_set.video_level_speed_and_motion_collate_fn,
        )
    else:
        train_loader = DataLoader(
            train_set,
            num_workers=opt.num_workers,
            batch_size=opt.batch_size,
            shuffle=False if opt.debug else True,
            drop_last=True,
            collate_fn=train_set._rep_lrn_collate_fn,
        )

        val_loader = DataLoader(
            val_set,
            num_workers=opt.num_workers,
            batch_size=opt.batch_size,
            shuffle=False if opt.debug else True,
            drop_last=True,
            collate_fn=val_set._rep_lrn_collate_fn,
        )

    # train_loader = get_anomaly_loader('avenue', 32, 1/25, 25, load_videos=True, val=False, load_frames=True)
    # val_loader = get_anomaly_loader('avenue', 32, 1/25, 25, load_videos=True, val=True, load_frames=True)
    # train_set = UCFCrimeVideoLoader('train', load_frames=False)
    # val_set = UCFCrimeVideoLoader('val', load_frames=False)
    #
    #
    # # ts = ConcatDataset([train_set])
    # # vs = ConcatDataset([val_set])
    # #
    # train_loader = DataLoader(train_set,
    #                           num_workers=opt.num_workers,
    #                           batch_size=opt.batch_size,
    #                           shuffle=True,
    #                           drop_last=False,
    #                           collate_fn=train_set._rep_lrn_collate_fn)
    #
    # val_loader = DataLoader(val_set,
    #                         num_workers=opt.num_workers,
    #                         batch_size=opt.batch_size,
    #                         shuffle=False,
    #                         drop_last=False,
    #                         collate_fn=val_set._rep_lrn_collate_fn)
    if opt.multi_scale:
        model, optimizer, loss = create_model_trn_2x(
            opt.num_classes, pretrained=opt.pretrained, pretrain_scale="frame"
        )
    else:
        if opt.backbone == "vit_longformer":
            model, optimizer, loss = create_model(
                opt.num_classes, pretrained=opt.pretrained
            )
        elif opt.backbone == "r3d_18":
            model, optimizer, loss = create_r3d(pretrained=opt.pretrained)
    # feat_extractor = build_resnet_50()

    # lo = MILRegLoss(model)
    # y_pred = torch.randn((2, 32))
    # y_true = torch.cat((torch.ones((32, )), torch.zeros((32, ))), dim=0)
    #
    # l = lo(y_pred, y_true)

    epoch = 0

    # if opt.pretrained:
    #     saved_model = torch.load(opt.vtn_ptr_path)
    #     optimizer.load_state_dict(saved_model['optimizer'])
    #     epoch = saved_model['epoch'] + 1

    # opt.batch_size = 256
    # opt.workers = 32
    # opt.balance_fails_only = True
    # opt.all_fail_videos = False
    # # train_loader = get_video_loader(opt)
    # opt.val = True
    # opt.fails_path = '/BS/unintentional_actions/nobackup/oops/oops_dataset/oops_video'
    # val_loader_class = get_video_loader(opt)
    # opt.batch_size = 32

    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss=loss,
        test_freq=1,
        epochs=25,
        train_set=train_set,
        epoch=epoch,
    )

    return


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    learn_representation()
