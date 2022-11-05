# @Author: Enea Duka
# @Date: 5/3/21
import sys

sys.path.append('/BS/unintentional_actions/work/unintentional_actions')
import os
import argparse
import time
from dataloaders.kinetics_loader import KineticsDataset
from dataloaders.my_oops_loader import get_video_loader
from dataloaders.ucf_crime_loader import get_crime_video_loader
from dataloaders.rareacts_loader import RareactsDataset
from dataloaders.oops_loader_simple import SimpleOopsDataset
from dataloaders.trn_ad_loader import get_anomaly_loader
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.pm_vtn import create_model, get_froze_trn_optimizer
from models.resnet50 import build_resnet_50
from action_localization.train_regression import train
from action_localization.test_classification import test
from utils.logging_setup import setup_logger_path
from utils.arg_parse import opt
import torch
from datetime import datetime
from torch.utils.data.dataloader import default_collate



def learn_representation():
    opt.dataset = 'oops'
    opt.rep_data_level = 'features'
    opt.backbone = 'resnet'
    opt.rep_backbone = 'resnet'
    opt.lr_scheduler = 'step'
    # opt.task = 'regression'

    # opt.attention_probs_dropout_prob = 0
    # opt.hidden_dropout_prob = 0
    # opt.num_hidden_layers = 6
    # opt.attention_window = [16] * opt.num_hidden_layers
    opt.embed_dim = 768
    # opt.num_hidden_layers = 3
    # opt.attention_window = [4] * opt.num_hidden_layers
    # opt.attention_heads = 4
    opt.intermediate_size = 3072
    opt.hidden_dim = 768
    # opt.attention_probs_dropout_prob = 0.4
    # opt.hidden_dropout_prob = 0.4
    # opt.mlp_dropout = 0.4
    opt.num_classes = 3
    opt.num_classes_ptr = 3

    opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/all/VideoLongformer/all.rep_learning.tag:speed&motion.layers3.attn_win32.20210622-100001/val/top1/VideoLongformer__all.rep_learning.tag:speed&motion.layers3.attn_win32.20210622-100001_v0.5402_ep48.pth.tar' # best one so far 62.81% val acc
    opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/oops/VideoLongformer_MLP/oops.unint_act.classification.layers3.attn_win32_best_classification/val/top1/VideoLongformer_MLP__oops.unint_act.classification.layers3.attn_win32_best_classification_v0.6251_ep32.pth.tar'
    # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/all/VideoLongformer/all.rep_learning.tag:speed&motion.layers3.attn_win32.20210623-103212/val/top1/VideoLongformer__all.rep_learning.tag:speed&motion.layers3.attn_win32.20210623-103212_v0.6039_ep49.pth.tar'

    # these ones have prototypical memory
    # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/all/VideoLongformer/all.rep_learning.tag:speed&motion.layers3.attn_win32.20210627-121738/val/top1/VideoLongformer__all.rep_learning.tag:speed&motion.layers3.attn_win32.20210627-121738_v0.5692_ep48.pth.tar'

    # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/oops/VideoLongformer_MLP/oops.unint_act.classification.layers3.attn_win32.time20210704-192810/lr:0.000100~ep:100~bs:256~win:32~b_lr:0.050000~ptr:True/val/top1/VideoLongformer_MLP__oops.unint_act.classification.layers3.attn_win32.time20210704-192810_v0.6220_ep75.pth.tar' # gives 71.2% acc in loc within 1s
    # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/oops/VideoLongformer_MLP/oops.unint_act.classification.layers3.attn_win32.time20210704-095237/val/top1/VideoLongformer_MLP__oops.unint_act.classification.layers3.attn_win32.time20210704-095237_v0.6151_ep38.pth.tar'
    # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/oops/VideoLongformer_MLP/oops.unint_act.classification.layers3.attn_win32.time20210705-094422/lr:0.000100~ep:50~bs:32~win:32~b_lr:0.010000~ptr:True/val/top1/VideoLongformer_MLP__oops.unint_act.classification.layers3.attn_win32.time20210705-094422_v0.6049_ep26.pth.tar'
    # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/oops/VideoLongformer_MLP/oops.unint_act.classification.layers3.attn_win32.time20210705-034023/lr:0.000100~ep:50~bs:32~win:32~b_lr:1.000000~ptr:False/val/top1/VideoLongformer_MLP__oops.unint_act.classification.layers3.attn_win32.time20210705-034023_v5.4548_ep33.pth.tar'
    # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/oops/VideoLongformer_MLP/oops.unint_act.classification.layers3.attn_win32.time20210706-010455/lr:0.000100~ep:50~bs:256~win:32~b_lr:0.010000~ptr:False/val/top1/VideoLongformer_MLP__oops.unint_act.classification.layers3.attn_win32.time20210706-010455_v0.6158_ep48.pth.tar'
    opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/oops/VideoLongformer_MLP/oops.unint_act.classification.layers3.attn_win32.time20210706-230147/lr:0.000100~ep:50~bs:256~win:32~b_lr:1.000000~ptr:True/val/top1/VideoLongformer_MLP__oops.unint_act.classification.layers3.attn_win32.time20210706-230147_v0.6185_ep47.pth.tar'
    # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/oops/VideoLongformer_MLP/oops.unint_act.classification.layers3.attn_win32.time20210706-231444/lr:0.000100~ep:50~bs:256~win:32~b_lr:0.020000~ptr:True/val/top1/VideoLongformer_MLP__oops.unint_act.classification.layers3.attn_win32.time20210706-231444_v0.5918_ep48.pth.tar'
    # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/oops/VideoLongformer_MLP/oops.unint_act.classification.layers3.attn_win32.time20210705-094422/lr:0.000100~ep:50~bs:32~win:32~b_lr:0.010000~ptr:True/val/top1/VideoLongformer_MLP__oops.unint_act.classification.layers3.attn_win32.time20210705-094422_v0.6049_ep26.pth.tar'
    opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/oops/VideoLongformer_MLP/oops.unint_act.classification.layers3.attn_win32.time20210713-004035_cntr_loss/lr:0.000100~ep:40~bs:128~win:32~b_lr:1.000000~ptr:True_cntr_loss/val/top1/VideoLongformer_MLP__oops.unint_act.classification.layers3.attn_win32.time20210713-004035_cntr_loss_v0.6418_ep31.pth.tar'
    opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/oops/VideoLongformer_MLP/oops.unint_act.classification.layers3.attn_win32.time20210713-093910_cntr_loss/lr:0.000100~ep:40~bs:128~win:32~b_lr:1.000000~ptr:True_cntr_loss/val/top1/VideoLongformer_MLP__oops.unint_act.classification.layers3.attn_win32.time20210713-093910_cntr_loss_v0.6444_ep37.pth.tar'
    # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/oops/VideoLongformer_MLP/oops.unint_act.classification.layers3.attn_win32.time20211106-192104_cntr_loss/lr:0.000100~ep:50~bs:128~win:32~b_lr:1.000000~ptr:False_cntr_loss/val/top1/VideoLongformer_MLP__oops.unint_act.classification.layers3.attn_win32.time20211106-192104_cntr_loss_v0.6031_ep34.pth.tar'
    # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/oops/VideoLongformer_MLP/oops.unint_act.classification.layers3.attn_win32.time20211028-083718_cntr_loss/lr:0.000100~ep:30~bs:128~win:32~b_lr:1.000000~ptr:False_cntr_loss/val/top1/VideoLongformer_MLP__oops.unint_act.classification.layers3.attn_win32.time20211028-083718_cntr_loss_v0.6083_ep25.pth.tar'
    # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/oops/VideoLongformer_MLP/oops.unint_act.classification.layers3.attn_win32.time20211123-010540_trn_2x/lr:0.000100~ep:50~bs:128~win:32~b_lr:1.000000~ptr:True_cntr_loss/val/top1/VideoLongformer_MLP__oops.unint_act.classification.layers3.attn_win32.time20211123-010540_trn_2x_v0.7497_ep6.pth.tar'
    # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/oops/VideoLongformer_MLP/oops.unint_act.classification.layers3.attn_win32.time20211123-010540_trn_2x/lr:0.000100~ep:50~bs:128~win:32~b_lr:1.000000~ptr:True_cntr_loss/val/top1/VideoLongformer_MLP__oops.unint_act.classification.layers3.attn_win32.time20211123-010540_trn_2x_v0.7497_ep6.pth.tar'
    opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/oops/VideoLongformer_MLP/oops.unint_act.classification.layers3.attn_win32.time20210828-214755_cntr_loss/lr:0.000100~ep:20~bs:128~win:32~b_lr:1.000000~ptr:True_cntr_loss/val/top1/VideoLongformer_MLP__oops.unint_act.classification.layers3.attn_win32.time20210828-214755_cntr_loss_v0.6525_ep0.pth.tar'

    opt.model_name = 'VideoLongformer_MLP'
    opt.viz = False
    opt.test = True
    opt.num_workers = 32
    opt.batch_size = 1 if opt.task == 'classification' else 16
    opt.sfx = str('%s.unint_act_loc.%s.layers%d.attn_win%d.task%s' % (
    opt.dataset, opt.task, opt.num_hidden_layers, opt.attention_window[0], opt.task))
    opt.save_model = 1
    opt.test_val = True
    opt.epochs = 50
    opt.gpu_parallel = True
    opt.use_tqdm = True
    opt.spat_temp = False
    opt.use_memory = False

    opt.optim = 'adam'
    opt.momentum = 0.9
    opt.lr = 1e-5
    opt.backbone_lr_factor = 1
    opt.cos_decay_lr_factor = 1e-2
    opt.weight_decay = 1e-4
    opt.test_freq = 1
    opt.save_model = 1
    opt.viz_env = '%s.%s%s_%s.' % (opt.model_name, opt.temp_learning_dataset_name, opt.env_pref, opt.sfx)

    opt.debug = False

    if opt.debug:
        opt.num_workers = 0
        opt.batch_size = 2
        opt.save_model = False
        opt.epochs = 1
        opt.viz = False

    setup_logger_path()

    train_set, val_set, test_set = None, None, None
    if opt.dataset == 'oops':
        train_set = SimpleOopsDataset('train', 16, 'features' if opt.task == 'classification' else 'frames', True,
                                      {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}, balance=True)
        val_set = SimpleOopsDataset('val', 16, 'features' if opt.task == 'classification' else 'frames', True,
                                    {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]},
                                    loc_class=True if opt.task=='classification' else False)

    if opt.dataset == 'oops' and opt.task == 'classification':
        opt.batch_size = 128
        opt.sample_videos = True
        opt.workers = 32
        opt.balance_fails_only = True
        opt.all_fail_videos = False
        train_loader = get_video_loader(opt)
        opt.val = True
        opt.fails_path = '/BS/unintentional_actions/nobackup/oops/oops_dataset/oops_video'
        val_loader = get_video_loader(opt)

        # opt.dataset_path = '/BS/unintentional_actions/nobackup/ucf_crime'
        # opt.val = True
        # opt.frames_per_clip = 16
        # opt.step_between_clips_sec = 0.25
        # opt.fps_list = [30]
        # opt.workers = 32
        # opt.batch_size = 64
        # val_loader = get_crime_video_loader(opt)


    else:
        train_loader = DataLoader(train_set,
                                  num_workers=opt.num_workers,
                                  batch_size=opt.batch_size,
                                  shuffle=True,
                                  drop_last=True,
                                  collate_fn=train_set.pad_videos_collate_fn if opt.task=='regression' else default_collate)

        val_loader = DataLoader(val_set,
                                num_workers=opt.num_workers,
                                batch_size=opt.batch_size,
                                shuffle=True,
                                drop_last=True,
                                collate_fn=train_set.pad_videos_collate_fn if opt.task=='regression' else default_collate)

    # if opt.dataset == 'avenue':
    #     val_loader = get_anomaly_loader('avenue', 16, 0.25, 16, True)

    model, optimizer, loss = create_model(pretrained=True, num_classes=1 if opt.task=='regression' else None)
    # optimizer = get_froze_trn_optimizer(model)
    feat_extractor = build_resnet_50()
    if opt.task == 'classification':
        test(model=model,
             loss=loss,
             dataloader=val_loader,
             mode='val',
             epoch=-1,
             time=datetime.now().strftime('%Y%m%d-%H%M%S'))
    else:
        train(model=model,
              feat_extractor=feat_extractor,
              train_loader=train_loader,
              val_loader=val_loader,
              optimizer=optimizer,
              loss=loss,
              test_freq=1,
              epochs=25)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    learn_representation()
