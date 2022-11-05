# @Author: Enea Duka
# @Date: 9/1/21
import sys

sys.path.append("/BS/unintentional_actions/work/unintentional_actions")

import os
from dataloaders.oops_loader import get_video_loader_frames
from tqdm import tqdm
from models.pm_vtn import create_model
from models.vit import create_vit_model
from utils.logging_setup import setup_logger_path
from utils.arg_parse import opt
from datetime import datetime
from action_classification.end2end_train import train

import torch
import warnings

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    # path for the weights of the pretrained model
    opt.vtn_ptr_path = "/BS/unintentional_actions/work/storage/models/all/VideoLongformer/all.rep_learning.tag:speed&motion.layers3.attn_win32.classes9.time20210713-012808/val/top1/VideoLongformer__all.rep_learning.tag:speed&motion.layers3.attn_win32.classes9.time20210713-012808_v0.5753_ep82.pth.tar"  # best one. gives 64.44% of acc
    opt.vtn_ptr_path = "/BS/unintentional_actions/work/storage/models/all/VideoLongformer/all.rep_learning.tag:speed&motion.layers3.attn_win32.classes8.time20210920-140805/val/top1/VideoLongformer__all.rep_learning.tag:speed&motion.layers3.attn_win32.classes8.time20210920-140805_v0.4972_ep27.pth.tar"

    # hyperparams for vtn model
    opt.embed_dim = 768
    opt.intermediate_size = 3072
    opt.hidden_dim = 768
    opt.num_classes = 8

    # hyperparams for the dataloader
    opt.dataset = "oops"
    opt.anticipate_label = 0
    opt.balance_fails_only = True
    opt.all_fail_videos = False

    # training hyperparams
    opt.pretrained = False
    opt.viz = False
    opt.test = True
    opt.workers = 0
    opt.batch_size = 2
    opt.save_model = 1
    opt.test_val = True
    opt.epochs = 20
    opt.gpu_parallel = True
    opt.use_tqdm = True
    opt.optim = "adam"
    opt.momentum = 0.9
    opt.lr = 1e-4
    opt.backbone_lr_factor = 1
    opt.cos_decay_lr_factor = 0.9
    opt.weight_decay = 1e-4
    opt.test_freq = 1
    opt.save_model = 1

    # logging params
    opt.model_name = "ViT_VideoLongformer_MLP"
    opt.sfx = str(
        "unint_act.layers%d.attn_win%d_trn"
        % (opt.num_hidden_layers, opt.attention_window[0])
    )
    opt.log_name = "lr:%f~ep:%d~bs:%d~win:%d~b_lr:%f~ptr:%s_cntr_loss" % (
        opt.lr,
        opt.epochs,
        opt.batch_size,
        opt.attention_window[0],
        opt.backbone_lr_factor,
        str(opt.pretrained),
    )
    opt.viz_env = "%s.%s%s_%s." % (
        opt.model_name,
        opt.temp_learning_dataset_name,
        opt.env_pref,
        opt.sfx,
    )
    opt.sfx = str(
        "%s.unint_act.%s.layers%d.attn_win%d.time%s_cntr_loss"
        % (
            opt.dataset,
            opt.task,
            opt.num_hidden_layers,
            opt.attention_window[0],
            datetime.now().strftime("%Y%m%d-%H%M%S"),
        )
    )

    setup_logger_path()

    train_loader = get_video_loader_frames(opt)
    opt.val = True
    opt.fails_path = "/BS/unintentional_actions/nobackup/oops/oops_dataset/oops_video"
    val_loader = get_video_loader_frames(opt)

    model, optimizer, loss = create_model(num_classes=3, pretrained=opt.pretrained)

    epoch = 0

    if opt.pretrained:
        saved_model = torch.load(opt.vtn_ptr_path)
        optimizer.load_state_dict(saved_model["optimizer"])
        epoch = saved_model["epoch"]
    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss=loss,
        test_freq=1,
        epoch=epoch,
    )
