
# @Author: Enea Duka
# @Date: 8/14/21
import sys

sys.path.append('/BS/unintentional_actions/work/unintentional_actions')
import os
import argparse
import time
from dataloaders.kinetics_loader import KineticsDataset
from dataloaders.my_oops_loader import get_video_loader
from dataloaders.rareacts_loader import RareactsDataset
from dataloaders.oops_loader_simple import SimpleOopsDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.pm_vtn import create_model, get_froze_trn_optimizer
from models.resnet50 import build_resnet_50
from models.mlp import create_mlp_model
from action_classification.train import train
from utils.logging_setup import setup_logger_path
from utils.arg_parse import opt
import torch
from datetime import datetime
from anomaly_detection.feat_extr import AnomalyFeatureExtractor
from dataloaders.trn_ad_loader import get_anomaly_loader
from torch.utils.data.dataloader import default_collate
from matplotlib import pyplot as plt
import numpy as np
from scipy.integrate import simps
from numpy import trapz
from sklearn import metrics
from utils.logging_setup import logger


def learn_representation():
    opt.dataset = 'avenue'
    opt.rep_data_level = 'features'
    opt.backbone = 'resnet'
    opt.rep_backbone = 'resnet'
    opt.lr_scheduler = 'step'

    opt.embed_dim = 768
    opt.intermediate_size = 3072
    opt.hidden_dim = 768
    opt.num_classes = 3

    opt.anticipate_label = 0

    opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/oops/VideoLongformer_MLP/oops.unint_act.classification.layers3.attn_win32.time20210713-093910_cntr_loss/lr:0.000100~ep:40~bs:128~win:32~b_lr:1.000000~ptr:True_cntr_loss/val/top1/VideoLongformer_MLP__oops.unint_act.classification.layers3.attn_win32.time20210713-093910_cntr_loss_v0.6444_ep37.pth.tar'
    # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/all/VideoLongformer/all.rep_learning.tag:speed&motion.layers3.attn_win32.classes9.time20210812-161813/val/top1/VideoLongformer__all.rep_learning.tag:speed&motion.layers3.attn_win32.classes9.time20210812-161813_v0.4921_ep70.pth.tar'
    # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/all/VideoLongformer/all.rep_learning.tag:speed&motion.layers3.attn_win32.classes9.time20210816-231902/val/top1/VideoLongformer__all.rep_learning.tag:speed&motion.layers3.attn_win32.classes9.time20210816-231902_v0.4286_ep78.pth.tar'
    # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/all/VideoLongformer/all.rep_learning.tag:speed&motion.layers3.attn_win32.classes9.time20210817-010213/val/top1/VideoLongformer__all.rep_learning.tag:speed&motion.layers3.attn_win32.classes9.time20210817-010213_v0.4974_ep17.pth.tar'
    # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/all/VideoLongformer/all.rep_learning.tag:speed&motion.layers3.attn_win32.classes4.time20210817-153834/val/top1/VideoLongformer__all.rep_learning.tag:speed&motion.layers3.attn_win32.classes4.time20210817-153834_v0.7138_ep14.pth.tar'
    # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/all/VideoLongformer/all.rep_learning.tag:speed&motion.layers3.attn_win32.classes9.time20210817-012516/val/top1/VideoLongformer__all.rep_learning.tag:speed&motion.layers3.attn_win32.classes9.time20210817-012516_v0.4127_ep8.pth.tar'
    # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/all/VideoLongformer/all.rep_learning.tag:speed&motion.layers3.attn_win32.classes9.time20210812-161459/val/top1/VideoLongformer__all.rep_learning.tag:speed&motion.layers3.attn_win32.classes9.time20210812-161459_v0.3500_ep1.pth.tar'
    # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/all/VideoLongformer/all.rep_learning.tag:speed&motion.layers3.attn_win32.classes9.time20210816-011209/val/top1/VideoLongformer__all.rep_learning.tag:speed&motion.layers3.attn_win32.classes9.time20210816-011209_v0.2698_ep48.pth.tar'
    # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/all/VideoLongformer/all.rep_learning.tag:speed&motion.layers3.attn_win32.classes9.time20210817-091521/val/top1/VideoLongformer__all.rep_learning.tag:speed&motion.layers3.attn_win32.classes9.time20210817-091521_v0.3333_ep9.pth.tar'
    # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/all/VideoLongformer/all.rep_learning.tag:speed&motion.layers3.attn_win32.classes9.time20210817-092831/val/top1/VideoLongformer__all.rep_learning.tag:speed&motion.layers3.attn_win32.classes9.time20210817-092831_v0.4701_ep8.pth.tar'
    # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/all/VideoLongformer/all.rep_learning.tag:speed&motion.layers3.attn_win32.classes9.time20210817-120258/val/top1/VideoLongformer__all.rep_learning.tag:speed&motion.layers3.attn_win32.classes9.time20210817-120258_v0.4052_ep9.pth.tar'
    # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/all/VideoLongformer/all.rep_learning.tag:speed&motion.layers3.attn_win32.classes2.time20210817-124254/val/top1/VideoLongformer__all.rep_learning.tag:speed&motion.layers3.attn_win32.classes2.time20210817-124254_v0.7844_ep6.pth.tar'
    # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/all/VideoLongformer/all.rep_learning.tag:speed&motion.layers3.attn_win32.classes2.time20210817-133246/val/top1/VideoLongformer__all.rep_learning.tag:speed&motion.layers3.attn_win32.classes2.time20210817-133246_v0.7670_ep5.pth.tar'
    # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/oops/VideoLongformer_MLP/oops.unint_act.classification.layers3.attn_win32.time20210817-140453_cntr_loss/lr:0.000100~ep:20~bs:128~win:32~b_lr:1.000000~ptr:False_cntr_loss/val/top1/VideoLongformer_MLP__oops.unint_act.classification.layers3.attn_win32.time20210817-140453_cntr_loss_v0.6218_ep5.pth.tar'
    # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/all/VideoLongformer/all.rep_learning.tag:speed&motion.layers3.attn_win32.classes2.time20210817-141355/val/top1/VideoLongformer__all.rep_learning.tag:speed&motion.layers3.attn_win32.classes2.time20210817-141355_v0.7339_ep9.pth.tar'
    # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/all/VideoLongformer/all.rep_learning.tag:speed&motion.layers3.attn_win32.classes4.time20210817-201133/val/top1/VideoLongformer__all.rep_learning.tag:speed&motion.layers3.attn_win32.classes4.time20210817-201133_v0.3810_ep4.pth.tar'
    # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/all/VideoLongformer/all.rep_learning.tag:speed&motion.layers3.attn_win32.classes4.time20210817-203003/val/top1/VideoLongformer__all.rep_learning.tag:speed&motion.layers3.attn_win32.classes4.time20210817-203003_v0.6667_ep61.pth.tar'
    # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/avenue/VideoLongformer_MLP/avenue.unint_act.classification.layers3.attn_win32.time20210819-000402_cntr_loss/lr:0.000100~ep:50~bs:128~win:32~b_lr:1.000000~ptr:False_cntr_loss/val/top1/VideoLongformer_MLP__avenue.unint_act.classification.layers3.attn_win32.time20210819-000402_cntr_loss_v0.7607_ep32.pth.tar'
    # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/avenue/VideoLongformer_MLP/avenue.unint_act.classification.layers3.attn_win32.time20210819-072552_cntr_loss/lr:0.000100~ep:50~bs:128~win:32~b_lr:1.000000~ptr:False_cntr_loss/val/top1/VideoLongformer_MLP__avenue.unint_act.classification.layers3.attn_win32.time20210819-072552_cntr_loss_v0.0000_ep33.pth.tar'
    opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/all/VideoLongformer/all.rep_learning.tag:speed&motion.layers3.attn_win32.classes9.time20210713-012808/val/top1/VideoLongformer__all.rep_learning.tag:speed&motion.layers3.attn_win32.classes9.time20210713-012808_v0.5753_ep82.pth.tar'  # best one. gives 64.44% of acc
    # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/all/VideoLongformer/all.rep_learning.tag:speed&motion.layers3.attn_win32.classes8.time20210819-104133/val/top1/VideoLongformer__all.rep_learning.tag:speed&motion.layers3.attn_win32.classes8.time20210819-104133_v0.3016_ep96.pth.tar'  # best one. gives 64.44% of acc
    # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/all/VideoLongformer/all.rep_learning.tag:speed&motion.layers3.attn_win32.classes9.time20210819-110958/val/top1/VideoLongformer__all.rep_learning.tag:speed&motion.layers3.attn_win32.classes9.time20210819-110958_v0.5397_ep44.pth.tar'  # best one. gives 64.44% of acc
    # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/all/VideoLongformer/all.rep_learning.tag:speed&motion.layers3.attn_win32.classes9.time20210822-172526/val/top1/VideoLongformer__all.rep_learning.tag:speed&motion.layers3.attn_win32.classes9.time20210822-172526_v0.6119_ep76.pth.tar'  # best one. gives 64.44% of acc
    #  opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/all/VideoLongformer/all.rep_learning.tag:speed&motion.layers3.attn_win32.classes8.time20210822-192435/val/top1/VideoLongformer__all.rep_learning.tag:speed&motion.layers3.attn_win32.classes8.time20210822-192435_v0.6154_ep81.pth.tar'  # best one. gives 64.44% of acc
    # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/all/VideoLongformer/all.rep_learning.tag:speed&motion.layers3.attn_win32.classes8.time20210823-005051/val/top1/VideoLongformer__all.rep_learning.tag:speed&motion.layers3.attn_win32.classes8.time20210823-005051_v0.4524_ep84.pth.tar'  # best one. gives 64.44% of acc
    # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/all/VideoLongformer/all.rep_learning.tag:speed&motion.layers3.attn_win32.classes8.time20210823-101737/val/top1/VideoLongformer__all.rep_learning.tag:speed&motion.layers3.attn_win32.classes8.time20210823-101737_v0.4992_ep48.pth.tar'  # mem size 256
    # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/all/VideoLongformer/all.rep_learning.tag:speed&motion.layers3.attn_win32.classes8.time20210826-133326/val/top1/VideoLongformer__all.rep_learning.tag:speed&motion.layers3.attn_win32.classes8.time20210826-133326_v0.5016_ep89.pth.tar'  # mem size 64
    # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/all/VideoLongformer/all.rep_learning.tag:speed&motion.layers3.attn_win32.classes8.time20210826-133443/val/top1/VideoLongformer__all.rep_learning.tag:speed&motion.layers3.attn_win32.classes8.time20210826-133443_v0.5016_ep80.pth.tar'  # mem size 128
    # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/all/VideoLongformer/all.rep_learning.tag:speed&motion.layers3.attn_win32.classes8.time20210826-134502/val/top1/VideoLongformer__all.rep_learning.tag:speed&motion.layers3.attn_win32.classes8.time20210826-134502_v0.4889_ep26.pth.tar'  # mem size 512
    # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/all/VideoLongformer/all.rep_learning.tag:speed&motion.layers3.attn_win32.classes8.time20210826-134623/val/top1/VideoLongformer__all.rep_learning.tag:speed&motion.layers3.attn_win32.classes8.time20210826-134623_v0.4944_ep42.pth.tar'  # mem size 1024
    # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/all/VideoLongformer/all.rep_learning.tag:speed&motion.layers3.attn_win32.classes8.time20210826-142555/val/top1/VideoLongformer__all.rep_learning.tag:speed&motion.layers3.attn_win32.classes8.time20210826-142555_v0.4952_ep48.pth.tar'  # mem size 256
    # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/all/VideoLongformer/all.rep_learning.tag:speed&motion.layers3.attn_win32.classes8.time20210826-162008/val/top1/VideoLongformer__all.rep_learning.tag:speed&motion.layers3.attn_win32.classes8.time20210826-162008_v0.4817_ep8.pth.tar '  # mem size 256


    # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/all/VideoLongformer/all.rep_learning.tag:speed&motion.layers3.attn_win32.classes9.time20210826-104032/val/top1/VideoLongformer__all.rep_learning.tag:speed&motion.layers3.attn_win32.classes9.time20210826-104032_v0.4841_ep11.pth.tar'  # best one. gives 64.44% of acc
    # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/all/VideoLongformer/all.rep_learning.tag:speed&motion.layers3.attn_win32.classes9.time20210826-104317/val/top1/VideoLongformer__all.rep_learning.tag:speed&motion.layers3.attn_win32.classes9.time20210826-104317_v0.2865_ep16.pth.tar'  # best one. gives 64.44% of acc
    # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/all/VideoLongformer/all.rep_learning.tag:speed&motion.layers3.attn_win32.classes8.time20210903-222318/val/top1/VideoLongformer__all.rep_learning.tag:speed&motion.layers3.attn_win32.classes8.time20210903-222318_v0.6073_ep68.pth.tar'  # best one. gives 64.44% of acc
    # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/oops/ViT_VideoLongformer_MLP/oops.unint_act.classification.layers3.attn_win32.time20210904-170702_cntr_loss/lr:0.000100~ep:20~bs:64~win:32~b_lr:1.000000~ptr:True_cntr_loss/val/top1/ViT_VideoLongformer_MLP__oops.unint_act.classification.layers3.attn_win32.time20210904-170702_cntr_loss_v0.6473_ep7.pth.tar'  # best one. gives 64.44% of acc
    # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/all/VideoLongformer/all.rep_learning.tag:speed&motion.layers3.attn_win32.classes8.time20210920-140805/val/top1/VideoLongformer__all.rep_learning.tag:speed&motion.layers3.attn_win32.classes8.time20210920-140805_v0.4972_ep27.pth.tar'
    # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/all/VideoLongformer/all.rep_learning.tag:speed&motion.layers3.attn_win32.classes8.time20210927-192213/val/top1/VideoLongformer__all.rep_learning.tag:speed&motion.layers3.attn_win32.classes8.time20210927-192213_v0.2421_ep34.pth.tar'
    # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/all/VideoLongformer/all.rep_learning.tag:speed&motion.layers3.attn_win32.classes8.time20210928-224230/val/top1/VideoLongformer__all.rep_learning.tag:speed&motion.layers3.attn_win32.classes8.time20210928-224230_v0.4603_ep23.pth.tar'
    # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/all/VideoLongformer/all.rep_learning.tag:speed&motion.layers3.attn_win32.classes4.time20211001-090507/val/top1/VideoLongformer__all.rep_learning.tag:speed&motion.layers3.attn_win32.classes4.time20211001-090507_v0.2540_ep12.pth.tar'
    # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/all/VideoLongformer/all.rep_learning.tag:speed&motion.layers3.attn_win32.classes4.time20211002-125702/val/top1/VideoLongformer__all.rep_learning.tag:speed&motion.layers3.attn_win32.classes4.time20211002-125702_v1.0000_ep0.pth.tar'
    # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/all/VideoLongformer/all.rep_learning.tag:speed&motion.layers3.attn_win32.classes4.time20211002-172216/val/top1/VideoLongformer__all.rep_learning.tag:speed&motion.layers3.attn_win32.classes4.time20211002-172216_v0.2698_ep98.pth.tar'
    # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/all/VideoLongformer/all.rep_learning.tag:speed&motion.layers3.attn_win32.classes4.time20211002-162810/val/top1/VideoLongformer__all.rep_learning.tag:speed&motion.layers3.attn_win32.classes4.time20211002-162810_v0.5286_ep1.pth.tar'
    # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/all/VideoLongformer/all.rep_learning.tag:speed&motion.layers3.attn_win32.classes4.time20211002-162810/val/top1/VideoLongformer__all.rep_learning.tag:speed&motion.layers3.attn_win32.classes4.time20211002-162810_v0.5286_ep1.pth.tar'


    # pretrained on contrastive loss
    opt.model_name = 'VideoLongformer_MLP'
    opt.viz = False
    opt.test = True
    opt.num_workers = 32
    opt.batch_size = 32
    opt.sfx = str('%s.unint_act.%s.layers%d.attn_win%d_2x_trn' % (
        opt.dataset, opt.task, opt.num_hidden_layers, opt.attention_window[0]))
    opt.save_model = 1
    opt.test_val = True
    opt.epochs = 1
    opt.gpu_parallel = True
    opt.use_tqdm = True
    opt.spat_temp = False
    opt.use_memory = False
    opt.use_bbone = False
    opt.mmargin_loss = False

    opt.optim = 'adam'
    opt.momentum = 0.9
    opt.lr = 1e-4
    opt.backbone_lr_factor = 1
    opt.cos_decay_lr_factor = 0.1
    opt.weight_decay = 1e-4
    opt.test_freq = 1
    opt.save_model = 100
    opt.pretrained = False  #################################################################################################
    opt.log_name = 'lr:%f~ep:%d~bs:%d~win:%d~b_lr:%f~ptr:%s_cntr_loss' % (opt.lr, opt.epochs, opt.batch_size,
                                                                opt.attention_window[0], opt.backbone_lr_factor,
                                                                str(opt.pretrained))
    opt.viz_env = '%s.%s%s_%s.' % (opt.model_name, opt.temp_learning_dataset_name, opt.env_pref, opt.sfx)
    opt.sfx = str('%s.unint_act.%s.layers%d.attn_win%d.time%s_cntr_loss' % (
        opt.dataset, opt.task, opt.num_hidden_layers, opt.attention_window[0],
        datetime.now().strftime('%Y%m%d-%H%M%S')))

    opt.debug = False 

    if opt.debug:
        opt.num_workers = 0
        opt.batch_size = 2
        opt.save_model = False
        opt.epochs = 1

        opt.viz = False

    # if opt.dataset == 'avenue':
    fpc = 1
    logger.debug('Temporal extent: %d' % fpc)
    frame_train_loader = get_anomaly_loader(opt.dataset, fpc, 1/25, 25, load_videos=False, load_frames=False)
    frame_val_loader = get_anomaly_loader(opt.dataset, fpc, 1/25, 25, load_videos=False, val=True, load_frames=False)

    fpc = 32
    logger.debug('Temporal extent: %d' % fpc)
    clip_train_loader = get_anomaly_loader(opt.dataset, fpc, 1 / 25, 25, load_videos=False, load_frames=False)
    clip_val_loader = get_anomaly_loader(opt.dataset, fpc, 1 / 25, 25, load_videos=False, val=True, load_frames=False)


    setup_logger_path()
    feature_save_path = '/BS/unintentional_actions/work/data/avenue/train_features/trained_opps_no_finetune'
    feat_extractor = AnomalyFeatureExtractor(feature_save_path)
    # feat_extractor.train_ae(train_loader, val_loader)
    feat_extractor.extract_feats(clip_train_loader)
    feat_extractor.do_outlier_detection(clip_val_loader)

    # for idx, data in enumerate(tqdm(clip_val_loader)):
    #     pass
    #
    # for idx, data in enumerate(tqdm(clip_train_loader)):
    #     pass

    # feat_extractor.train_svm(frame_train_loader, clip_train_loader)
    # feat_extractor.eval_svm(frame_val_loader, clip_val_loader)

    tprs = []
    fprs = []
    thresholds = [0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2, 3, 4, 5, 6, 7, 8]
    thresholds.reverse()
    data_vector = None
    for t in thresholds:
        feat_extractor.update_threshold(t)
        tpr, fpr = feat_extractor.do_outlier_detection(clip_val_loader)
        tprs.append(tpr)
        fprs.append(fpr)
    print(tprs)

    tprs.reverse()
    fprs.reverse()
    area = metrics.auc(fprs, tprs)
    logger.debug("AUC: %f" % area)

    tprs.reverse()
    fprs.reverse()
    x_pos = 0
    y_pos = 0
    plt.text(x_pos, y_pos, "AUC: %f" % area)
    plt.plot(fprs, tprs, )
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.show()

    # fnrs = 1 - np.asarray(tprs)
    # err_1 = fprs[np.nanargmin(np.absolute((fnrs - np.asarray(fprs))))]
    # err_2 = fnrs[np.nanargmin(np.absolute((fnrs - np.asarray(fprs))))]
    #
    # print("EER: %f" % ((err_1+err_2)/2))





if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # x = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    # y = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    # plt.plot(x, y)
    # plt.show()
    # print(metrics.auc(x, y))
    learn_representation()
