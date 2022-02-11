# @Author: Enea Duka
# @Date: 9/1/21
import sys
sys.path.append('/BS/unintentional_actions/work/unintentional_actions')

import os
from tqdm import tqdm
from dataloaders.rok_loader import ROKDataset
from models.pm_vtn import create_model
from models.vit import create_vit_model
from utils.logging_setup import setup_logger_path
from utils.arg_parse import opt
from datetime import datetime
from rep_learning.end2end_train import train
from torch.utils.data import DataLoader



import torch
import warnings

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/oops/ViT_VideoLongformer_MLP/oops.unint_act.classification.layers3.attn_win32.time20210914-010223_cntr_loss/lr:0.000100~ep:20~bs:64~win:32~b_lr:1.000000~ptr:False_cntr_loss/val/top1/ViT_VideoLongformer_MLP__oops.unint_act.classification.layers3.attn_win32.time20210914-010223_cntr_loss_v0.3724_ep2.pth.tar'
    opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/oops/ViT_VideoLongformer_MLP/oops.unint_act.classification.layers3.attn_win32.time20210914-190408_cntr_loss/lr:0.000100~ep:20~bs:64~win:32~b_lr:1.000000~ptr:False_cntr_loss/val/top1/ViT_VideoLongformer_MLP__oops.unint_act.classification.layers3.attn_win32.time20210914-190408_cntr_loss_v0.1777_ep0.pth.tar'
    opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/oops/ViT_VideoLongformer_MLP/oops.unint_act.classification.layers3.attn_win32.time20210916-005536_cntr_loss/lr:0.000050~ep:20~bs:64~win:32~b_lr:1.000000~ptr:False_cntr_loss/val/top1/ViT_VideoLongformer_MLP__oops.unint_act.classification.layers3.attn_win32.time20210916-005536_cntr_loss_v0.2371_ep0.pth.tar'
    opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/oops/ViT_VideoLongformer_MLP/oops.unint_act.classification.layers3.attn_win32.time20210916-091641_cntr_loss/lr:0.000050~ep:20~bs:64~win:32~b_lr:1.000000~ptr:True_cntr_loss/val/top1/ViT_VideoLongformer_MLP__oops.unint_act.classification.layers3.attn_win32.time20210916-091641_cntr_loss_v0.2644_ep2.pth.tar'


    # hyperparams for vtn model
    opt.embed_dim = 768
    opt.intermediate_size = 3072
    opt.hidden_dim = 768
    opt.num_classes = 8

    # hyperparams for the dataloader
    opt.dataset = 'oops'
    opt.anticipate_label = 0
    opt.balance_fails_only = True
    opt.all_fail_videos = False

    # training hyperparams
    opt.pretrained = False
    opt.viz = False
    opt.test = True
    opt.workers = 64
    opt.batch_size = 64
    opt.save_model = 1
    opt.test_val = True
    opt.epochs = 50
    opt.gpu_parallel = True
    opt.use_tqdm = True
    opt.optim = 'adam'
    opt.momentum = 0.9
    opt.lr = 1e-4
    opt.backbone_lr_factor = 1
    opt.cos_decay_lr_factor = 0.01
    opt.weight_decay = 1e-4
    opt.test_freq = 1
    opt.save_model = 1

    # logging params
    opt.model_name = 'ViT_VideoLongformer_MLP'
    opt.sfx = str('unint_act.layers%d.attn_win%d_trn' % (opt.num_hidden_layers, opt.attention_window[0]))
    opt.log_name = 'lr:%f~ep:%d~bs:%d~win:%d~b_lr:%f~ptr:%s_cntr_loss' % (opt.lr, opt.epochs, opt.batch_size,
                                                                    opt.attention_window[0], opt.backbone_lr_factor,
                                                                    str(opt.pretrained))
    opt.viz_env = '%s.%s%s_%s.' % (opt.model_name, opt.temp_learning_dataset_name, opt.env_pref, opt.sfx)
    opt.sfx = str('%s.unint_act.%s.layers%d.attn_win%d.time%s_cntr_loss' % (
        opt.dataset, opt.task, opt.num_hidden_layers, opt.attention_window[0],
        datetime.now().strftime('%Y%m%d-%H%M%S')))

    setup_logger_path()

    train_set = ROKDataset('train', spat_scale=True, size=224, spat_crop=True, load_frames=True)
    val_set = ROKDataset('val', spat_scale=True, size=224, spat_crop=True, load_frames=True)

    train_loader = DataLoader(train_set,
                              num_workers=32,
                              batch_size=16,
                              pin_memory=True,
                              shuffle=True,
                              collate_fn=train_set.speed_and_motion_collate_fn)

    val_loader = DataLoader(val_set,
                              num_workers=32,
                              batch_size=16,
                              pin_memory=True,
                              shuffle=False,
                              collate_fn=train_set.speed_and_motion_collate_fn)




    model, optimizer, loss = create_model(num_classes=(None if opt.pretrained else 8), pretrained=opt.pretrained)
    backbone = create_vit_model(pretrain=opt.pretrained)

    # optimizer = torch.optim.AdamW([{'params': model.parameters(), 'lr': opt.lr}],
    #                               weight_decay=opt.weight_decay)
    epoch = 0

    if opt.pretrained:
        saved_model = torch.load(opt.vtn_ptr_path)
        optimizer.load_state_dict(saved_model['optimizer'])
        epoch = saved_model['epoch'] + 1

    # for idx, data in enumerate(tqdm(train_loader)):
    #     videos = data[0]
    #     labels = data[1]
    #     pnfs = data[2]
    #
    #     continue

    train(model=model,
         feat_extractor=backbone,
         train_loader=train_loader,
         val_loader=val_loader,
         optimizer=optimizer,
         loss=loss,
         test_freq=1,
         epoch=epoch,
         train_set=train_set)
