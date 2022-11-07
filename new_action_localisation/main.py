# @Author: Enea Duka
# @Date: 5/3/21
import warnings

import os
from dataloaders.kinetics_loader import KineticsDataset
from new_action_localisation.my_oops_loader import get_video_loader
from transformer2x.my_oops_loader import get_video_loader_trn_2x
from dataloaders.oops_loader_simple import SimpleOopsDataset
from torch.utils.data import DataLoader
from new_action_localisation.pm_vtn import create_model
from new_action_localisation.train import train
from utils.logging_setup import setup_logger_path
from utils.arg_parse import opt
from datetime import datetime
from torch.utils.data.dataloader import default_collate


def learn_representation():
    # opt.dataset = 'oops'
    # opt.rep_data_level = 'features'
    # opt.rep_backbone = 'resnet'
    # opt.lr_scheduler = 'step'
    #
    # # opt.attention_probs_dropout_prob = 0
    # # opt.hidden_dropout_prob = 0
    # # opt.num_hidden_layers = 6
    # # opt.attention_window = [4] * opt.num_hidden_layers
    # opt.embed_dim = 768
    # # opt.num_hidden_layers = 3
    # # opt.attention_window = [4] * opt.num_hidden_layers
    # # opt.attention_heads = 4
    # opt.intermediate_size = 3072
    # opt.hidden_dim = 768
    # # opt.attention_probs_dropout_prob = 0.3
    # # opt.hidden_dropout_prob = 0.3
    # # opt.mlp_dropout = 0.3
    # opt.num_classes_ptr = 7
    #
    # opt.anticipate_label = 0
    #
    # # parameters for the reduced complexity model
    # # opt.embed_dim = 768
    # # opt.num_hidden_layers = 3
    # # opt.attention_window = [4] * opt.num_hidden_layers
    # # opt.num_attention_heads = 8
    # # opt.intermediate_size = 128
    # # opt.mlp_dim = 128
    # # opt.hidden_dim = 768
    # # opt.attention_probs_dropout_prob = 0.2
    # # opt.hidden_dropout_prob = 0.2
    # # opt.mlp_dropout = 0.4
    # # opt.num_classes = 7
    #
    # # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/kinetics/VideoLongformer/kinetics.rep_learning.layers3.attn_win18.20210601-121721/val/top1/VideoLongformer__kinetics.rep_learning.layers3.attn_win18.20210601-121721_v0.9236_ep63.pth.tar'
    # # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/kinetics/VideoLongformer/kinetics.rep_learning.layers3.attn_win18.20210601-112717/val/top1/VideoLongformer__kinetics.rep_learning.layers3.attn_win18.20210601-112717_v0.7035_ep21.pth.tar'
    # # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/kinetics/VideoLongformer/kinetics.rep_learning.layers3.attn_win18.20210603-095339/val/top1/VideoLongformer__kinetics.rep_learning.layers3.attn_win18.20210603-095339_v0.7115_ep15.pth.tar'
    # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/kinetics/VideoLongformer/kinetics.rep_learning.layers3.attn_win18.20210603-155238/val/top1/VideoLongformer__kinetics.rep_learning.layers3.attn_win18.20210603-155238_v0.6898_ep65.pth.tar'
    # # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/kinetics/VideoLongformer/kinetics.rep_learning.layers3.attn_win32.20210604-201505/val/top1/VideoLongformer__kinetics.rep_learning.layers3.attn_win32.20210604-201505_v0.4266_ep18.pth.tar'
    # # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/kinetics/VideoLongformer/kinetics.rep_learning.layers1.attn_win32.20210604-221704/val/top1/VideoLongformer__kinetics.rep_learning.layers1.attn_win32.20210604-221704_v0.4427_ep24.pth.tar'
    # # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/kinetics/VideoLongformer/kinetics.rep_learning.layers3.attn_win32.20210604-234944/val/top1/VideoLongformer__kinetics.rep_learning.layers3.attn_win32.20210604-234944_v0.4243_ep26.pth.tar'
    # # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/kinetics/VideoLongformer/kinetics.rep_learning.layers3.attn_win32.20210605-133838/val/top1/VideoLongformer__kinetics.rep_learning.layers3.attn_win32.20210605-133838_v0.4343_ep36.pth.tar'
    # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/kinetics/VideoLongformer/kinetics.rep_learning.layers3.attn_win32.20210605-223927/val/top1/VideoLongformer__kinetics.rep_learning.layers3.attn_win32.20210605-223927_v0.7292_ep3.pth.tar'
    # # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/kinetics/VideoLongformer/kinetics.rep_learning.tag:speed&motion.layers3.attn_win32.20210606-111009/val/top1/VideoLongformer__kinetics.rep_learning.tag:speed&motion.layers3.attn_win32.20210606-111009_v0.4243_ep40.pth.tar'
    # # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/kinetics/VideoLongformer/kinetics.rep_learning.tag:speed&motion.layers3.attn_win32.20210606-171751/val/top1/VideoLongformer__kinetics.rep_learning.tag:speed&motion.layers3.attn_win32.20210606-171751_v0.5463_ep11.pth.tar'
    # # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/all/VideoLongformer/all.rep_learning.tag:speed&motion.layers3.attn_win32.20210615-084757/val/top1/VideoLongformer__all.rep_learning.tag:speed&motion.layers3.attn_win32.20210615-084757_v0.4324_ep2.pth.tar'
    # # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/all/VideoLongformer/all.rep_learning.tag:speed&motion.layers6.attn_win32.20210615-115251/val/top1/VideoLongformer__all.rep_learning.tag:speed&motion.layers6.attn_win32.20210615-115251_v0.2555_ep0.pth.tar'
    # # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/kinetics/VideoLongformer/kinetics.rep_learning.tag:speed&motion.layers3.attn_win32.20210621-193149/val/top1/VideoLongformer__kinetics.rep_learning.tag:speed&motion.layers3.attn_win32.20210621-193149_v0.5419_ep36.pth.tar'
    # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/all/VideoLongformer/all.rep_learning.tag:speed&motion.layers3.attn_win32.20210622-100001/val/top1/VideoLongformer__all.rep_learning.tag:speed&motion.layers3.attn_win32.20210622-100001_v0.5402_ep48.pth.tar'  # best one so far 62.56% val acc
    # # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/all/VideoLongformer/all.rep_learning.tag:speed&motion.layers3.attn_win4.20210701-023239_reduced_capacity_oops_only/val/top1/VideoLongformer__all.rep_learning.tag:speed&motion.layers3.attn_win4.20210701-023239_reduced_capacity_oops_only_v0.6510_ep19.pth.tar' # best one so far 62.56% val acc
    # # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/all/VideoLongformer/all.rep_learning.tag:speed&motion.layers3.attn_win4.20210630-100936_reduced_capacity_oops_only/val/top1/VideoLongformer__all.rep_learning.tag:speed&motion.layers3.attn_win4.20210630-100936_reduced_capacity_oops_only_v0.6690_ep40.pth.tar'
    # # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/all/VideoLongformer/all.rep_learning.tag:speed&motion.layers3.attn_win32.20210623-103212/val/top1/VideoLongformer__all.rep_learning.tag:speed&motion.layers3.attn_win32.20210623-103212_v0.6039_ep49.pth.tar'
    #
    # # these ones have prototypical memory
    # # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/all/VideoLongformer/all.rep_learning.tag:speed&motion.layers3.attn_win32.20210627-121738/val/top1/VideoLongformer__all.rep_learning.tag:speed&motion.layers3.attn_win32.20210627-121738_v0.5692_ep48.pth.tar'
    #
    # # these are validated on unintentional action classification
    # # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/all/VideoLongformer/all.rep_learning.tag:speed&motion.layers3.attn_win32_reduced_capacity_oops_only/val/top1/VideoLongformer__all.rep_learning.tag:speed&motion.layers3.attn_win32_reduced_capacity_oops_only_v0.5186_ep10.pth.tar'
    # # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/all/VideoLongformer/all.rep_learning.tag:speed&motion.layers3.attn_win32.time20210704-112033/val/top1/VideoLongformer__all.rep_learning.tag:speed&motion.layers3.attn_win32.time20210704-112033_v0.5049_ep0.pth.tar'
    #
    # # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/all/VideoLongformer/all.rep_learning.tag:speed&motion.layers3.attn_win32.time20210704-195117/val/top1/VideoLongformer__all.rep_learning.tag:speed&motion.layers3.attn_win32.time20210704-195117_v0.4898_ep28.pth.tar'
    # # # motion + normal video
    # # # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/all/VideoLongformer/all.rep_learning.tag:speed&motion.layers3.attn_win32.time20210704-025158/val/top1/VideoLongformer__all.rep_learning.tag:speed&motion.layers3.attn_win32.time20210704-025158_v0.5797_ep23.pth.tar'
    # # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/oops/VideoLongformer_MLP/oops.unint_act.classification.layers3.attn_win32.time20210705-034023/lr:0.000100~ep:50~bs:32~win:32~b_lr:1.000000~ptr:False/val/top1/VideoLongformer_MLP__oops.unint_act.classification.layers3.attn_win32.time20210705-034023_v5.4548_ep33.pth.tar'
    # #
    # # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/all/VideoLongformer/all.rep_learning.tag:speed&motion.layers3.attn_win32.classes9.time20210706-124744/val/top1/VideoLongformer__all.rep_learning.tag:speed&motion.layers3.attn_win32.classes9.time20210706-124744_v0.3890_ep27.pth.tar'  # oops only 9 classes
    # # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/all/VideoLongformer/all.rep_learning.tag:speed&motion.layers3.attn_win32.classes3.time20210706-125717/val/top1/VideoLongformer__all.rep_learning.tag:speed&motion.layers3.attn_win32.classes3.time20210706-125717_v0.5681_ep25.pth.tar' # oops only 3 classes
    # #
    # # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/all/VideoLongformer/all.rep_learning.tag:speed&motion.layers3.attn_win32.classes3.time20210706-131207/val/top1/VideoLongformer__all.rep_learning.tag:speed&motion.layers3.attn_win32.classes3.time20210706-131207_v0.5302_ep17.pth.tar' # rok 3 classes
    #
    # # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/oops/VideoLongformer_MLP/oops.unint_act.classification.layers3.attn_win32.time20210706-230147/lr:0.000100~ep:50~bs:256~win:32~b_lr:1.000000~ptr:True/val/top1/VideoLongformer_MLP__oops.unint_act.classification.layers3.attn_win32.time20210706-230147_v0.6185_ep47.pth.tar'
    #
    # # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/oops/VideoLongformer_MLP/oops.unint_act.classification.layers3.attn_win32.time20210712-102951_cntr_loss/lr:0.000010~ep:50~bs:128~win:32~b_lr:1.000000~ptr:False_cntr_loss/val/top1/VideoLongformer_MLP__oops.unint_act.classification.layers3.attn_win32.time20210712-102951_cntr_loss_v1.5963_ep33.pth.tar'
    # # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/all/VideoLongformer/all.rep_learning.tag:speed&motion.layers3.attn_win32.classes3.time20210712-124809/val/top1/VideoLongformer__all.rep_learning.tag:speed&motion.layers3.attn_win32.classes3.time20210712-124809_v0.7155_ep70.pth.tar'
    # # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/all/VideoLongformer/all.rep_learning.tag:speed&motion.layers3.attn_win32.classes4.time20210712-222354/val/top1/VideoLongformer__all.rep_learning.tag:speed&motion.layers3.attn_win32.classes4.time20210712-222354_v0.5624_ep62.pth.tar'
    # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/all/VideoLongformer/all.rep_learning.tag:speed&motion.layers3.attn_win32.classes9.time20210712-230738/val/top1/VideoLongformer__all.rep_learning.tag:speed&motion.layers3.attn_win32.classes9.time20210712-230738_v0.5533_ep59.pth.tar'  # best one. gives 64.17% of acc
    # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/all/VideoLongformer/all.rep_learning.tag:speed&motion.layers3.attn_win32.classes9.time20210713-012808/val/top1/VideoLongformer__all.rep_learning.tag:speed&motion.layers3.attn_win32.classes9.time20210713-012808_v0.5753_ep82.pth.tar'  # best one. gives 64.44% of acc
    # # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/all/VideoLongformer/all.rep_learning.tag:speed&motion.layers3.attn_win32.classes9.time20210713-124941/val/top1/VideoLongformer__all.rep_learning.tag:speed&motion.layers3.attn_win32.classes9.time20210713-124941_v0.5906_ep63.pth.tar'  #
    # # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/all/VideoLongformer/all.rep_learning.tag:speed&motion.layers3.attn_win32.classes9.time20210713-094607/val/top1/VideoLongformer__all.rep_learning.tag:speed&motion.layers3.attn_win32.classes9.time20210713-094607_v0.5851_ep49.pth.tar'  #
    # # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/all/VideoLongformer/all.rep_learning.tag:speed&motion.layers3.attn_win32.classes9.time20210716-101657/val/top1/VideoLongformer__all.rep_learning.tag:speed&motion.layers3.attn_win32.classes9.time20210716-101657_v0.4795_ep37.pth.tar'
    # # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/all/VideoLongformer/all.rep_learning.tag:speed&motion.layers3.attn_win32.classes9.time20210802-132037/val/top1/VideoLongformer__all.rep_learning.tag:speed&motion.layers3.attn_win32.classes9.time20210802-132037_v0.5736_ep98.pth.tar'
    # # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/all/VideoLongformer/all.rep_learning.tag:speed&motion.layers1.attn_win32.classes9.time20210803-113147/val/top1/VideoLongformer__all.rep_learning.tag:speed&motion.layers1.attn_win32.classes9.time20210803-113147_v0.4939_ep78.pth.tar'
    # # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/all/VideoLongformer/all.rep_learning.tag:speed&motion.layers6.attn_win32.classes9.time20210803-113354/val/top1/VideoLongformer__all.rep_learning.tag:speed&motion.layers6.attn_win32.classes9.time20210803-113354_v0.5919_ep79.pth.tar'
    # # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/all/VideoLongformer/all.rep_learning.tag:speed&motion.layers3.attn_win4.classes9.time20210803-142545/val/top1/VideoLongformer__all.rep_learning.tag:speed&motion.layers3.attn_win4.classes9.time20210803-142545_v0.5688_ep63.pth.tar'
    # # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/all/VideoLongformer/all.rep_learning.tag:speed&motion.layers3.attn_win16.classes9.time20210803-153143/val/top1/VideoLongformer__all.rep_learning.tag:speed&motion.layers3.attn_win16.classes9.time20210803-153143_v0.5479_ep98.pth.tar'
    # # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/all/VideoLongformer/all.rep_learning.tag:speed&motion.layers3.attn_win64.classes9.time20210803-230920/val/top1/VideoLongformer__all.rep_learning.tag:speed&motion.layers3.attn_win64.classes9.time20210803-230920_v0.5772_ep80.pth.tar'
    # # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/all/VideoLongformer/all.rep_learning.tag:speed&motion.layers6.attn_win4.classes9.time20210803-230920/val/top1/VideoLongformer__all.rep_learning.tag:speed&motion.layers6.attn_win4.classes9.time20210803-230920_v0.5885_ep96.pth.tar'
    # # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/all/VideoLongformer/all.rep_learning.tag:speed&motion.layers3.attn_win32.classes9.time20210804-112553/val/top1/VideoLongformer__all.rep_learning.tag:speed&motion.layers3.attn_win32.classes9.time20210804-112553_v0.5697_ep90.pth.tar'
    # # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/all/VideoLongformer/all.rep_learning.tag:speed&motion.layers3.attn_win32.classes9.time20210804-224329/val/top1/VideoLongformer__all.rep_learning.tag:speed&motion.layers3.attn_win32.classes9.time20210804-224329_v0.6402_ep33.pth.tar'
    # # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/all/VideoLongformer/all.rep_learning.tag:speed&motion.layers3.attn_win32.classes9.time20210805-110233/val/top1/VideoLongformer__all.rep_learning.tag:speed&motion.layers3.attn_win32.classes9.time20210805-110233_v0.5778_ep89.pth.tar'
    # # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/all/VideoLongformer/all.rep_learning.tag:speed&motion.layers3.attn_win32.classes9.time20210816-155402/val/top1/VideoLongformer__all.rep_learning.tag:speed&motion.layers3.attn_win32.classes9.time20210816-155402_v0.5731_ep86.pth.tar'
    # # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/oops/VideoLongformer_MLP/oops.unint_act.classification.layers3.attn_win32.time20210713-093910_cntr_loss/lr:0.000100~ep:40~bs:128~win:32~b_lr:1.000000~ptr:True_cntr_loss/val/top1/VideoLongformer_MLP__oops.unint_act.classification.layers3.attn_win32.time20210713-093910_cntr_loss_v0.6444_ep37.pth.tar'
    # # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/all/VideoLongformer/all.rep_learning.tag:speed&motion.layers3.attn_win32.classes9.time20210817-120258/val/top1/VideoLongformer__all.rep_learning.tag:speed&motion.layers3.attn_win32.classes9.time20210817-120258_v0.4052_ep9.pth.tar'
    # # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/oops/VideoLongformer_MLP/oops.unint_act.classification.layers3.attn_win32.time20210915-120919_cntr_loss/lr:0.000100~ep:10~bs:128~win:32~b_lr:1.000000~ptr:False_cntr_loss/val/top1/VideoLongformer_MLP__oops.unint_act.classification.layers3.attn_win32.time20210915-120919_cntr_loss_v0.5962_ep1.pt'
    # # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/all/VideoLongformer/all.rep_learning.tag:speed&motion.layers3.attn_win32.classes8.time20210920-140805/val/top1/VideoLongformer__all.rep_learning.tag:speed&motion.layers3.attn_win32.classes8.time20210920-140805_v0.4972_ep27.pth.tar'
    # # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/oops/VideoLongformer_MLP/oops.unint_act.classification.layers3.attn_win32.time20210925-190538_cntr_loss/lr:0.000100~ep:30~bs:128~win:32~b_lr:1.000000~ptr:True_cntr_loss/val/top1/VideoLongformer_MLP__oops.unint_act.classification.layers3.attn_win32.time20210925-190538_cntr_loss_v0.6146_ep11.pth.tar'
    # # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/oops/VideoLongformer_MLP/oops.unint_act.classification.layers3.attn_win32.time20210713-093910_cntr_loss/lr:0.000100~ep:40~bs:128~win:32~b_lr:1.000000~ptr:True_cntr_loss/val/top1/VideoLongformer_MLP__oops.unint_act.classification.layers3.attn_win32.time20210713-093910_cntr_loss_v0.6444_ep37.pth.tar'
    # # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/all/VideoLongformer/all.rep_learning.tag:speed.layers3.attn_win32.classes4.time20211107-025051/val/top1/VideoLongformer__all.rep_learning.tag:speed.layers3.attn_win32.classes4.time20211107-025051_v0.7514_ep67.pth.tar'
    # # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/all/VideoLongformer/all.rep_learning.tag:motion.layers3.attn_win32.classes4.time20211107-050235/val/top1/VideoLongformer__all.rep_learning.tag:motion.layers3.attn_win32.classes4.time20211107-050235_v0.7135_ep57.pth.tar'
    # opt.vtn_ptr_path = '/BS/feature_bank/work/enea/models/all/VideoLongformer/val/top1/VideoLongformer___v0.4157_ep63.pth.tar'
    #
    #
    # # pretrained on contrastive loss
    # opt.model_name = 'VideoLongformer_MLP'
    # opt.viz = False
    # opt.test = True
    # opt.num_workers = 32
    # opt.batch_size = 128
    # opt.sfx = str('%s.unint_act.%s.layers%d.attn_win%d_trn' % (
    #     opt.dataset, opt.task, opt.num_hidden_layers, opt.attention_window[0]))
    # opt.save_model = 1
    # opt.test_val = True
    # opt.epochs = 50
    # opt.gpu_parallel = True
    # opt.use_tqdm = True
    # opt.spat_temp = False
    # opt.use_memory = False
    # opt.use_bbone = False
    # opt.mmargin_loss = False
    #
    # opt.use_crf = False
    #
    # opt.optim = 'adam'
    # opt.momentum = 0.9
    # opt.lr = 1e-4
    # opt.backbone_lr_factor = 1
    # opt.cos_decay_lr_factor = 0.01
    # opt.weight_decay = 1e-4
    # opt.test_freq = 1
    # opt.save_model = 1
    # opt.pretrained = True  #################################################################################################
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

    # opt.debug = False
    #
    # if opt.debug:
    #     opt.num_workers = 0
    #     opt.batch_size = 2
    #     opt.save_model = False
    #     opt.epochs = 1
    #
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
        if opt.use_crf:
            # opt.batch_size = 16
            # opt.workers = 32
            opt.balance_fails_only = False
            opt.all_fail_videos = False
            opt.load_videos = True
            opt.step_between_clips_sec = 0.25
            train_loader = get_video_loader_trn_2x(opt)
            opt.val = True
            opt.fails_path = (
                "/BS/unintentional_actions/nobackup/oops/oops_dataset/oops_video"
            )
            val_loader = get_video_loader_trn_2x(opt)
        else:
            # opt.batch_size = 64
            # opt.workers = 32
            opt.balance_fails_only = True
            opt.all_fail_videos = False
            opt.load_videos = False
            # # opt.step_between_clips_sec = 1.0
            train_loader = get_video_loader(opt)
            opt.val = True
            opt.fails_path = (
                "/BS/unintentional_actions/nobackup/oops/oops_dataset/oops_video"
            )
            val_loader = get_video_loader(opt)

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
            shuffle=True,
            drop_last=True,
            collate_fn=train_set.pad_videos_collate_fn
            if opt.task == "regression"
            else default_collate,
        )

    # if opt.dataset == 'avenue':
    #     train_loader = get_anomaly_loader('avenue', 10, 1/25 , 25, load_videos=False, val=False)
    #     val_loader = get_anomaly_loader('avenue', 16, 1 / 25, 25, load_videos=False, val=True)

    model, optimizer, loss = create_model(
        num_classes=100 if opt.dataset == "kinetics" else opt.num_classes,
        pretrained=opt.pretrained,
    )

    epoch = 0

    # if opt.pretrained:
    #     saved_model = torch.load(opt.vtn_ptr_path)
    #     optimizer.load_state_dict(saved_model['optimizer'])
    #     epoch = saved_model['epoch'] + 1

    # if pretrained:
    #     optimizer = get_froze_trn_optimizer(model)
    # feat_extractor = build_resnet_50()
    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss=loss,
        test_freq=1,
        epochs=25,
        epoch=epoch,
    )


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    warnings.filterwarnings("ignore")

    learn_representation()
