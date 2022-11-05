# @Author: Enea Duka
# @Date: 5/3/21
import sys

sys.path.append("/BS/unintentional_actions/work/unintentional_actions")
import os
import argparse
import time
from dataloaders.kinetics_loader import KineticsDataset
from transformer2x.my_oops_loader import get_video_loader_trn_2x
from dataloaders.rareacts_loader import RareactsDataset
from dataloaders.oops_loader_simple import SimpleOopsDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformer2x.vtn import create_model_trn_2x, get_froze_trn_optimizer
from models.resnet50 import build_resnet_50
from models.mlp import create_mlp_model
from transformer2x.train import train
from utils.logging_setup import setup_logger_path
from utils.arg_parse import opt
import torch
from datetime import datetime
from torch.utils.data.dataloader import default_collate

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import splrep, splev
import cv2
from scipy.interpolate import interp1d


def learn_representation():
    opt.dataset = "oops"
    opt.rep_data_level = "features"
    opt.rep_backbone = "resnet"
    opt.lr_scheduler = "step"
    opt.backbone = "vit_longformer"

    # opt.attention_probs_dropout_prob = 0
    # opt.hidden_dropout_prob = 0
    # opt.num_hidden_layers = 6
    # opt.attention_window = [32] * opt.num_hidden_layers
    opt.embed_dim = 512
    # opt.num_hidden_layers = 3
    # opt.attention_window = [4] * opt.num_hidden_layers
    # opt.attention_heads = 4
    opt.intermediate_size = 3072
    opt.hidden_dim = 512
    opt.mlp_dim = 1024
    # opt.attention_probs_dropout_prob = 0.3
    # opt.hidden_dropout_prob = 0.3
    # opt.mlp_dropout = 0.3
    # opt.num_classes = 3

    # parameters for the reduced complexity model
    # opt.embed_dim = 768
    # opt.num_hidden_layers = 3
    # opt.attention_window = [32] * opt.num_hidden_layers
    # opt.num_attention_heads = 8
    # opt.intermediate_size = 128
    # opt.mlp_dim = 128
    # opt.hidden_dim = 768
    # opt.attention_probs_dropout_prob = 0.2
    # opt.hidden_dropout_prob = 0.2
    # opt.mlp_dropout = 0.2
    opt.num_classes = 7
    opt.num_classes_ptr = 7

    # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/kinetics/VideoLongformer/kinetics.rep_learning.layers3.attn_win18.20210601-121721/val/top1/VideoLongformer__kinetics.rep_learning.layers3.attn_win18.20210601-121721_v0.9236_ep63.pth.tar'
    # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/kinetics/VideoLongformer/kinetics.rep_learning.layers3.attn_win18.20210601-112717/val/top1/VideoLongformer__kinetics.rep_learning.layers3.attn_win18.20210601-112717_v0.7035_ep21.pth.tar'
    # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/kinetics/VideoLongformer/kinetics.rep_learning.layers3.attn_win18.20210603-095339/val/top1/VideoLongformer__kinetics.rep_learning.layers3.attn_win18.20210603-095339_v0.7115_ep15.pth.tar'
    opt.vtn_ptr_path = "/BS/unintentional_actions/work/storage/models/kinetics/VideoLongformer/kinetics.rep_learning.layers3.attn_win18.20210603-155238/val/top1/VideoLongformer__kinetics.rep_learning.layers3.attn_win18.20210603-155238_v0.6898_ep65.pth.tar"
    # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/kinetics/VideoLongformer/kinetics.rep_learning.layers3.attn_win32.20210604-201505/val/top1/VideoLongformer__kinetics.rep_learning.layers3.attn_win32.20210604-201505_v0.4266_ep18.pth.tar'
    # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/kinetics/VideoLongformer/kinetics.rep_learning.layers1.attn_win32.20210604-221704/val/top1/VideoLongformer__kinetics.rep_learning.layers1.attn_win32.20210604-221704_v0.4427_ep24.pth.tar'
    # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/kinetics/VideoLongformer/kinetics.rep_learning.layers3.attn_win32.20210604-234944/val/top1/VideoLongformer__kinetics.rep_learning.layers3.attn_win32.20210604-234944_v0.4243_ep26.pth.tar'
    # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/kinetics/VideoLongformer/kinetics.rep_learning.layers3.attn_win32.20210605-133838/val/top1/VideoLongformer__kinetics.rep_learning.layers3.attn_win32.20210605-133838_v0.4343_ep36.pth.tar'
    opt.vtn_ptr_path = "/BS/unintentional_actions/work/storage/models/kinetics/VideoLongformer/kinetics.rep_learning.layers3.attn_win32.20210605-223927/val/top1/VideoLongformer__kinetics.rep_learning.layers3.attn_win32.20210605-223927_v0.7292_ep3.pth.tar"
    # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/kinetics/VideoLongformer/kinetics.rep_learning.tag:speed&motion.layers3.attn_win32.20210606-111009/val/top1/VideoLongformer__kinetics.rep_learning.tag:speed&motion.layers3.attn_win32.20210606-111009_v0.4243_ep40.pth.tar'
    # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/kinetics/VideoLongformer/kinetics.rep_learning.tag:speed&motion.layers3.attn_win32.20210606-171751/val/top1/VideoLongformer__kinetics.rep_learning.tag:speed&motion.layers3.attn_win32.20210606-171751_v0.5463_ep11.pth.tar'
    # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/all/VideoLongformer/all.rep_learning.tag:speed&motion.layers3.attn_win32.20210615-084757/val/top1/VideoLongformer__all.rep_learning.tag:speed&motion.layers3.attn_win32.20210615-084757_v0.4324_ep2.pth.tar'
    # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/all/VideoLongformer/all.rep_learning.tag:speed&motion.layers6.attn_win32.20210615-115251/val/top1/VideoLongformer__all.rep_learning.tag:speed&motion.layers6.attn_win32.20210615-115251_v0.2555_ep0.pth.tar'
    # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/kinetics/VideoLongformer/kinetics.rep_learning.tag:speed&motion.layers3.attn_win32.20210621-193149/val/top1/VideoLongformer__kinetics.rep_learning.tag:speed&motion.layers3.attn_win32.20210621-193149_v0.5419_ep36.pth.tar'
    opt.vtn_ptr_path = "/BS/unintentional_actions/work/storage/models/all/VideoLongformer/all.rep_learning.tag:speed&motion.layers3.attn_win32.20210622-100001/val/top1/VideoLongformer__all.rep_learning.tag:speed&motion.layers3.attn_win32.20210622-100001_v0.5402_ep48.pth.tar"  # best one so far 62.56% val acc
    # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/all/VideoLongformer/all.rep_learning.tag:speed&motion.layers3.attn_win4.20210701-023239_reduced_capacity_oops_only/val/top1/VideoLongformer__all.rep_learning.tag:speed&motion.layers3.attn_win4.20210701-023239_reduced_capacity_oops_only_v0.6510_ep19.pth.tar' # best one so far 62.56% val acc
    # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/all/VideoLongformer/all.rep_learning.tag:speed&motion.layers3.attn_win4.20210630-100936_reduced_capacity_oops_only/val/top1/VideoLongformer__all.rep_learning.tag:speed&motion.layers3.attn_win4.20210630-100936_reduced_capacity_oops_only_v0.6690_ep40.pth.tar'
    # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/all/VideoLongformer/all.rep_learning.tag:speed&motion.layers3.attn_win32.20210623-103212/val/top1/VideoLongformer__all.rep_learning.tag:speed&motion.layers3.attn_win32.20210623-103212_v0.6039_ep49.pth.tar'

    # these ones have prototypical memory
    # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/all/VideoLongformer/all.rep_learning.tag:speed&motion.layers3.attn_win32.20210627-121738/val/top1/VideoLongformer__all.rep_learning.tag:speed&motion.layers3.attn_win32.20210627-121738_v0.5692_ep48.pth.tar'

    # these are validated on unintentional action classification
    # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/all/VideoLongformer/all.rep_learning.tag:speed&motion.layers3.attn_win32_reduced_capacity_oops_only/val/top1/VideoLongformer__all.rep_learning.tag:speed&motion.layers3.attn_win32_reduced_capacity_oops_only_v0.5186_ep10.pth.tar'
    # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/all/VideoLongformer/all.rep_learning.tag:speed&motion.layers3.attn_win32.time20210704-112033/val/top1/VideoLongformer__all.rep_learning.tag:speed&motion.layers3.attn_win32.time20210704-112033_v0.5049_ep0.pth.tar'

    # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/all/VideoLongformer/all.rep_learning.tag:speed&motion.layers3.attn_win32.time20210704-195117/val/top1/VideoLongformer__all.rep_learning.tag:speed&motion.layers3.attn_win32.time20210704-195117_v0.4898_ep28.pth.tar'
    # # motion + normal video
    # # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/all/VideoLongformer/all.rep_learning.tag:speed&motion.layers3.attn_win32.time20210704-025158/val/top1/VideoLongformer__all.rep_learning.tag:speed&motion.layers3.attn_win32.time20210704-025158_v0.5797_ep23.pth.tar'
    # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/oops/VideoLongformer_MLP/oops.unint_act.classification.layers3.attn_win32.time20210705-034023/lr:0.000100~ep:50~bs:32~win:32~b_lr:1.000000~ptr:False/val/top1/VideoLongformer_MLP__oops.unint_act.classification.layers3.attn_win32.time20210705-034023_v5.4548_ep33.pth.tar'
    #
    # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/all/VideoLongformer/all.rep_learning.tag:speed&motion.layers3.attn_win32.classes9.time20210706-124744/val/top1/VideoLongformer__all.rep_learning.tag:speed&motion.layers3.attn_win32.classes9.time20210706-124744_v0.3890_ep27.pth.tar'  # oops only 9 classes
    # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/all/VideoLongformer/all.rep_learning.tag:speed&motion.layers3.attn_win32.classes3.time20210706-125717/val/top1/VideoLongformer__all.rep_learning.tag:speed&motion.layers3.attn_win32.classes3.time20210706-125717_v0.5681_ep25.pth.tar' # oops only 3 classes
    #
    # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/all/VideoLongformer/all.rep_learning.tag:speed&motion.layers3.attn_win32.classes3.time20210706-131207/val/top1/VideoLongformer__all.rep_learning.tag:speed&motion.layers3.attn_win32.classes3.time20210706-131207_v0.5302_ep17.pth.tar' # rok 3 classes

    # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/oops/VideoLongformer_MLP/oops.unint_act.classification.layers3.attn_win32.time20210706-230147/lr:0.000100~ep:50~bs:256~win:32~b_lr:1.000000~ptr:True/val/top1/VideoLongformer_MLP__oops.unint_act.classification.layers3.attn_win32.time20210706-230147_v0.6185_ep47.pth.tar'

    # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/oops/VideoLongformer_MLP/oops.unint_act.classification.layers3.attn_win32.time20210712-102951_cntr_loss/lr:0.000010~ep:50~bs:128~win:32~b_lr:1.000000~ptr:False_cntr_loss/val/top1/VideoLongformer_MLP__oops.unint_act.classification.layers3.attn_win32.time20210712-102951_cntr_loss_v1.5963_ep33.pth.tar'
    # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/all/VideoLongformer/all.rep_learning.tag:speed&motion.layers3.attn_win32.classes3.time20210712-124809/val/top1/VideoLongformer__all.rep_learning.tag:speed&motion.layers3.attn_win32.classes3.time20210712-124809_v0.7155_ep70.pth.tar'
    # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/all/VideoLongformer/all.rep_learning.tag:speed&motion.layers3.attn_win32.classes4.time20210712-222354/val/top1/VideoLongformer__all.rep_learning.tag:speed&motion.layers3.attn_win32.classes4.time20210712-222354_v0.5624_ep62.pth.tar'
    opt.vtn_ptr_path = "/BS/unintentional_actions/work/storage/models/all/VideoLongformer/all.rep_learning.tag:speed&motion.layers3.attn_win32.classes9.time20210712-230738/val/top1/VideoLongformer__all.rep_learning.tag:speed&motion.layers3.attn_win32.classes9.time20210712-230738_v0.5533_ep59.pth.tar"  # best one. gives 64.17% of acc
    # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/all/VideoLongformer/all.rep_learning.tag:speed&motion.layers3.attn_win32.classes9.time20210713-012808/val/top1/VideoLongformer__all.rep_learning.tag:speed&motion.layers3.attn_win32.classes9.time20210713-012808_v0.5753_ep82.pth.tar'  # best one. gives 64.44% of acc
    # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/all/VideoLongformer/all.rep_learning.tag:speed&motion.layers3.attn_win32.classes9.time20210713-124941/val/top1/VideoLongformer__all.rep_learning.tag:speed&motion.layers3.attn_win32.classes9.time20210713-124941_v0.5906_ep63.pth.tar'  #
    # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/all/VideoLongformer/all.rep_learning.tag:speed&motion.layers3.attn_win32.classes9.time20210713-094607/val/top1/VideoLongformer__all.rep_learning.tag:speed&motion.layers3.attn_win32.classes9.time20210713-094607_v0.5851_ep49.pth.tar'  #
    opt.vtn_ptr_path = "/BS/unintentional_actions/work/storage/models/all/VideoLongformer/all.rep_learning.tag:speed&motion.layers3.attn_win32.classes9.time20210716-101657/val/top1/VideoLongformer__all.rep_learning.tag:speed&motion.layers3.attn_win32.classes9.time20210716-101657_v0.4795_ep37.pth.tar"
    # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/oops/VideoLongformer_MLP/oops.unint_act.classification.layers3.attn_win32.time20210728-200615_cntr_loss/lr:0.000100~ep:20~bs:128~win:32~b_lr:1.000000~ptr:True_cntr_loss/val/top1/VideoLongformer_MLP__oops.unint_act.classification.layers3.attn_win32.time20210728-200615_cntr_loss_v0.7792_ep19.pth.tar'
    # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/all/VideoLongformer/all.rep_learning.tag:speed&motion.layers1.attn_win32.classes8.time20211030-120906/val/top1/VideoLongformer__all.rep_learning.tag:speed&motion.layers1.attn_win32.classes8.time20211030-120906_v0.4783_ep84.pth.tar'
    # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/all/VideoLongformer/all.rep_learning.tag:speed&motion.layers6.attn_win4.classes9.time20210803-230920/val/top1/VideoLongformer__all.rep_learning.tag:speed&motion.layers6.attn_win4.classes9.time20210803-230920_v0.5885_ep96.pth.tar'
    opt.vtn_ptr_path = "/BS/unintentional_actions/work/storage/models/oops/VideoLongformer_MLP/oops.unint_act.classification.layers3.attn_win32.time20211101-095037_cntr_loss/lr:0.000100~ep:50~bs:128~win:32~b_lr:1.000000~ptr:True_cntr_loss/val/top1/VideoLongformer_MLP__oops.unint_act.classification.layers3.attn_win32.time20211101-095037_cntr_loss_v0.7804_ep26.pth.tar"
    # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/oops/VideoLongformer_MLP/oops.unint_act.classification.layers3.attn_win32.time20211102-200314_trn_2x/lr:0.000100~ep:50~bs:128~win:32~b_lr:1.000000~ptr:True_cntr_loss/val/top1/VideoLongformer_MLP__oops.unint_act.classification.layers3.attn_win32.time20211102-200314_trn_2x_v0.7623_ep2.pth.tar'
    # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/all/VideoLongformer/all.rep_learning.tag:motion.layers3.attn_win32.classes4.time20211111-082743/val/top1/VideoLongformer__all.rep_learning.tag:motion.layers3.attn_win32.classes4.time20211111-082743_v0.6548_ep75.pth.tar'
    # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/all/VideoLongformer/all.rep_learning.tag:speed.layers3.attn_win32.classes3.time20211111-082723/val/top1/VideoLongformer__all.rep_learning.tag:speed.layers3.attn_win32.classes3.time20211111-082723_v0.7668_ep82.pth.tar'
    # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/all/VideoLongformer/all.rep_learning.tag:speed&motion.layers3.attn_win16.classes7.time20211118-200029/val/top1/VideoLongformer__all.rep_learning.tag:speed&motion.layers3.attn_win16.classes7.time20211118-200029_v0.4905_ep43.pth.tar'
    # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/all/VideoLongformer/all.rep_learning.tag:speed&motion.layers3.attn_win64.classes7.time20211119-051453/val/top1/VideoLongformer__all.rep_learning.tag:speed&motion.layers3.attn_win64.classes7.time20211119-051453_v0.4855_ep42.pth.tar'
    # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/all/VideoLongformer/all.rep_learning.tag:speed&motion.layers3.attn_win4.classes7.time20211118-171100/val/top1/VideoLongformer__all.rep_learning.tag:speed&motion.layers3.attn_win4.classes7.time20211118-171100_v0.5453_ep66.pth.tar'
    # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/all/VideoLongformer/all.rep_learning.tag:speed&motion.layers3.attn_win32.classes7.time20211119-224751/val/top1/VideoLongformer__all.rep_learning.tag:speed&motion.layers3.attn_win32.classes7.time20211119-224751_v0.4760_ep77.pth.tar'
    # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/all/VideoLongformer/all.rep_learning.tag:speed&motion.layers3.attn_win32.classes7.time20211120-103635/val/top1/VideoLongformer__all.rep_learning.tag:speed&motion.layers3.attn_win32.classes7.time20211120-103635_v0.4282_ep94.pth.tar'
    # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/oops/VideoLongformer_MLP/oops.unint_act.classification.layers3.attn_win32.time20211121-134407_trn_2x/lr:0.000100~ep:50~bs:128~win:32~b_lr:1.000000~ptr:True_cntr_loss/val/top1/VideoLongformer_MLP__oops.unint_act.classification.layers3.attn_win32.time20211121-134407_trn_2x_v0.7332_ep47.pth.tar'
    opt.vtn_ptr_path = "/BS/unintentional_actions/work/storage/models/oops/VideoLongformer_MLP/oops.unint_act.classification.layers3.attn_win32.time20211102-103443_cntr_loss/lr:0.000100~ep:50~bs:128~win:32~b_lr:1.000000~ptr:True_cntr_loss/val/top1/VideoLongformer_MLP__oops.unint_act.classification.layers3.attn_win32.time20211102-103443_cntr_loss_v0.7872_ep0.pth.tar"
    # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/oops/VideoLongformer_MLP/oops.unint_act.classification.layers3.attn_win32.time20210901-095708_cntr_loss/lr:0.000200~ep:20~bs:128~win:32~b_lr:1.000000~ptr:True_cntr_loss/val/top1/VideoLongformer_MLP__oops.unint_act.classification.layers3.attn_win32.time20210901-095708_cntr_loss_v0.7426_ep1.pth.tar'
    # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/oops/VideoLongformer_MLP/oops.unint_act.classification.layers3.attn_win32.time20211123-010540_trn_2x/lr:0.000100~ep:50~bs:128~win:32~b_lr:1.000000~ptr:True_cntr_loss/val/top1/VideoLongformer_MLP__oops.unint_act.classification.layers3.attn_win32.time20211123-010540_trn_2x_v0.7497_ep6.pth.tar'
    # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/oops/VideoLongformer_MLP/oops.unint_act.classification.layers3.attn_win32.time20211116-111427_trn_2x/lr:0.000100~ep:50~bs:128~win:32~b_lr:1.000000~ptr:False_cntr_loss/val/top1/VideoLongformer_MLP__oops.unint_act.classification.layers3.attn_win32.time20211116-111427_trn_2x_v0.6964_ep46.pth.tar'
    # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/all/Resnet3D18/all.rep_learning.tag:r3d_18.layers3.attn_win32.classes7.time20220130-032009/val/top1/Resnet3D18__all.rep_learning.tag:r3d_18.layers3.attn_win32.classes7.time20220130-032009_v0.5381_ep4.pth.tar'
    # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/all/Resnet3D18/all.rep_learning.tag:r3d_18.layers3.attn_win32.classes7.time20220130-222258/val/top1/Resnet3D18__all.rep_learning.tag:r3d_18.layers3.attn_win32.classes7.time20220130-222258_v0.5075_ep0.pth.tar'
    #
    # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/all/VideoLongformer/all.rep_learning.tag:warp.layers3.attn_win32.classes6.time20220205-211114/val/top1/VideoLongformer__all.rep_learning.tag:warp.layers3.attn_win32.classes6.time20220205-211114_v0.5955_ep63.pth.tar'
    # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/all/VideoLongformer/all.rep_learning.tag:shuffle_stage2.layers3.attn_win32.classes6.time20220208-092839/val/top1/VideoLongformer__all.rep_learning.tag:shuffle_stage2.layers3.attn_win32.classes6.time20220208-092839_v0.5198_ep97.pth.tar'
    # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/all/VideoLongformer/all.rep_learning.tag:random_point_speedup_stage2.layers3.attn_win32.classes6.time20220205-215811/val/top1/VideoLongformer__all.rep_learning.tag:random_point_speedup_stage2.layers3.attn_win32.classes6.time20220205-215811_v0.5131_ep97.pth.tar'
    # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/all/VideoLongformer/all.rep_learning.tag:double_flip_stage2.layers3.attn_win32.classes6.time20220206-145727/val/top1/VideoLongformer__all.rep_learning.tag:double_flip_stage2.layers3.attn_win32.classes6.time20220206-145727_v0.5979_ep59.pth.tar'
    opt.vtn_ptr_path = "/BS/unintentional_actions/work/storage/models/all/Resnet3D18/all.rep_learning.tag:r3d_18.layers3.attn_win32.classes7.time20220226-184715/val/top1/Resnet3D18__all.rep_learning.tag:r3d_18.layers3.attn_win32.classes7.time20220226-184715_v0.6551_ep13.pth.tar"
    opt.vtn_ptr_path = "/BS/unintentional_actions/work/storage/models/oops/VideoLongformer_MLP/oops.unint_act.classification.layers3.attn_win32.time20220301-231812_trn_2x/lr:0.000100~ep:50~bs:128~win:32~b_lr:1.000000~ptr:True_cntr_loss/val/top1/VideoLongformer_MLP__oops.unint_act.classification.layers3.attn_win32.time20220301-231812_trn_2x_v0.6633_ep0.pth.tar"
    opt.vtn_ptr_path = "/BS/unintentional_actions/work/storage/models/all/VideoLongformer/all.rep_learning.tag:speedx3.layers3.attn_win32.classes6.time20220307-113444/val/top1/VideoLongformer__all.rep_learning.tag:speedx3.layers3.attn_win32.classes6.time20220307-113444_v0.5592_ep55.pth.tar"
    opt.vtn_ptr_path = "/BS/unintentional_actions/work/storage/models/oops/VideoLongformer_MLP/oops.unint_act.classification.layers3.attn_win32.time20210828-214755_cntr_loss/lr:0.000100~ep:20~bs:128~win:32~b_lr:1.000000~ptr:True_cntr_loss/val/top1/VideoLongformer_MLP__oops.unint_act.classification.layers3.attn_win32.time20210828-214755_cntr_loss_v0.6525_ep0.pth.tar"

    # pretrained on contrastive loss
    opt.model_name = "VideoLongformer_MLP"
    opt.viz = False
    opt.test = True
    opt.num_workers = 32
    opt.batch_size = 128.0
    opt.sfx = str(
        "%s.unint_act.%s.layers%d.attn_win%d_2x_trn"
        % (opt.dataset, opt.task, opt.num_hidden_layers, opt.attention_window[0])
    )
    opt.save_model = 1
    opt.test_val = True
    opt.epochs = 50
    opt.gpu_parallel = True
    opt.use_tqdm = True
    opt.spat_temp = False
    opt.use_memory = False
    opt.use_bbone = False
    opt.mmargin_loss = False
    opt.use_crf = True
    opt.crf_margin_probs = False
    opt.use_frame_encoder = True

    opt.optim = "adam"
    opt.momentum = 0.9
    opt.lr = 1e-4
    opt.backbone_lr_factor = 1
    opt.cos_decay_lr_factor = 0.01
    opt.weight_decay = 1e-4
    opt.test_freq = 1
    opt.save_model = 1
    opt.multi_scale = False
    pretrained = True  #################################################################################################
    opt.log_name = "lr:%f~ep:%d~bs:%d~win:%d~b_lr:%f~ptr:%s_cntr_loss" % (
        opt.lr,
        opt.epochs,
        opt.batch_size,
        opt.attention_window[0],
        opt.backbone_lr_factor,
        str(pretrained),
    )
    opt.viz_env = "%s.%s%s_%s." % (
        opt.model_name,
        opt.temp_learning_dataset_name,
        opt.env_pref,
        opt.sfx,
    )
    opt.sfx = str(
        "%s.unint_act.%s.layers%d.attn_win%d.time%s_trn_2x"
        % (
            opt.dataset,
            opt.task,
            opt.num_hidden_layers,
            opt.attention_window[0],
            datetime.now().strftime("%Y%m%d-%H%M%S"),
        )
    )

    opt.debug = False

    if opt.debug:
        opt.num_workers = 0
        opt.batch_size = 2
        opt.save_model = False
        opt.epochs = 1

        opt.viz = False

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
        test_set = KineticsDataset(
            "test",
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
        train_set = SimpleOopsDataset(
            "train",
            16,
            "features" if opt.task == "classification" else "frames",
            True,
            {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
            balance=True,
        )
        val_set = SimpleOopsDataset(
            "val",
            16,
            "features" if opt.task == "classification" else "frames",
            True,
            {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        )
        pass
    elif opt.dataset == "all":
        pass

    if opt.dataset == "oops":
        opt.batch_size = 8
        opt.workers = 16
        opt.balance_fails_only = False
        opt.all_fail_videos = False
        opt.load_videos = True
        opt.load_frames = opt.backbone == "r3d_18"
        opt.step_between_clips_sec = 1
        # opt.anticipate_label = 0
        train_loader = get_video_loader_trn_2x(opt)
        opt.val = True
        opt.fails_path = (
            "/BS/unintentional_actions/nobackup/oops/oops_dataset/oops_video"
        )
        val_loader = get_video_loader_trn_2x(opt)

    else:
        train_loader = DataLoader(
            train_set,
            num_workers=opt.num_workers,
            batch_size=opt.batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=train_set.pad_videos_collate_fn
            if opt.task == "regression"
            else default_collate,
        )

        val_loader = DataLoader(
            val_set,
            num_workers=opt.num_workers,
            batch_size=opt.batch_size,
            shuffle=False,
            drop_last=True,
            collate_fn=train_set.pad_videos_collate_fn
            if opt.task == "regression"
            else default_collate,
        )

    model, optimizer, loss = create_model_trn_2x(
        num_classes=100 if opt.dataset == "kinetics" else 3, pretrained=pretrained
    )

    # if pretrained:
    #     optimizer = get_froze_trn_optimizer(model)
    feat_extractor = build_resnet_50()
    train(
        model=model,
        feat_extractor=feat_extractor,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss=loss,
        test_freq=1,
        epochs=25,
    )


def plot_qual_results(nrm_cnf, trn_cnf, unrm_crf, vid_path):
    def _frame_image(img, frame_width, frame_color):
        b = frame_width  # border size in pixel
        ny, nx = img.shape[0], img.shape[1]  # resolution / number of pixels in x and y
        if img.ndim == 3:  # rgb or rgba array
            framed_img = np.zeros((b + ny + b, b + nx + b, img.shape[2]))
            framed_img[:, :, 0] = frame_color[0]
            framed_img[:, :, 1] = frame_color[1]
            framed_img[:, :, 2] = frame_color[2]
        elif img.ndim == 2:  # grayscale image
            framed_img = np.zeros((b + ny + b, b + nx + b))
        framed_img[b:-b, b:-b] = img
        framed_img = framed_img.astype("int")
        return framed_img

    save_path = "/BS/unintentional_actions/work/unintentional_actions"
    x = list(range(len(nrm_cnf)))
    video = cv2.VideoCapture(vid_path)
    frame_width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = video.get(cv2.CAP_PROP_FPS)
    nr_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    nr_clips = nr_frames // 16
    idc = list(range(0, nr_frames, 16))
    idc = [elm + 8 for elm in idc]
    counter = 0
    curr_idc_idx = 0
    cols = len(idc)
    rows = 1

    fig, ax = plt.subplots(rows, cols, figsize=(10, 1))
    while True:
        ret, frame = video.read()
        if counter == idc[curr_idc_idx]:
            ax[curr_idc_idx].imshow(_frame_image(frame, int(100), [97, 196, 110]))
            ax[curr_idc_idx].axis("off")
            curr_idc_idx += 1
            if curr_idc_idx >= len(idc):
                break
        counter += 1
    plt.axis("off")
    # plt.show()
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.savefig(os.path.join(save_path, "fig.png"), dpi=1000)


if __name__ == "__main__":
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0 1 2 3'
    learn_representation()

    # nrm_cnf = [0.9, 0.85, 0.88, 0.97, 0.86, 0.01, 0.05, 0.09, 0.00, 0.02, 0.02, 0.01]
    # trn_cnf = [0.05, 0.1, 0.02, 0.01, 0.12, 0.97, 0.08, 0.08, 0.01, 0.06, 0.04, 0.02]
    # unrm_cnf = [0.05, 0.05, 0.1, 0.02, 0.02, 0.02, 0.06, 0.1, 0.99, 0.92, 0.94, 0.97]
    #
    # x = list(range(len(nrm_cnf)))
    # x_intp = np.linspace(min(x), max(x), 500)
    #
    # f_n = interp1d(x, nrm_cnf, kind='quadratic')
    # nrm_cnf = f_n(x_intp)
    # nrm_cnf[nrm_cnf < 0] = 0
    #
    # f_t = interp1d(x, trn_cnf, kind='quadratic')
    # trn_cnf = f_t(x_intp)
    # trn_cnf[trn_cnf < 0] = 0
    #
    # f_u = interp1d(x, unrm_cnf, kind='quadratic')
    # unrm_cnf = f_u(x_intp)
    # unrm_cnf[unrm_cnf < 0] = 0
    #
    #
    # plt.plot(x_intp, nrm_cnf, lw=1, color='#61c46e')
    # plt.fill_between(x_intp, 0, nrm_cnf, alpha=0.3, facecolor='#61c46e')
    # plt.plot(x_intp, trn_cnf, lw=1, color='#f6ed22')
    # plt.fill_between(x_intp, 0, trn_cnf, alpha=0.3, facecolor='#f6ed22')
    # plt.plot(x_intp, unrm_cnf, lw=1, color='#f15a24')
    # plt.fill_between(x_intp, 0, unrm_cnf, alpha=0.3, facecolor='#f15a24')
    # plt.xlabel('Time')
    # plt.ylabel('Prediction confidence')
    # plt.legend()
    # plt.show()
    #
    # green = [97, 196, 110]
    # yellow = [246, 237, 33]
    # red = [241, 90, 36]

    # vid_path = '/BS/unintentional_actions/nobackup/oops/oops_dataset/oops_video/val/34 Funny Kid Nominees - FailArmy Hall Of Fame (May 2017)11.mp4'
    # plot_qual_results(nrm_cnf, trn_cnf, unrm_cnf, vid_path)
