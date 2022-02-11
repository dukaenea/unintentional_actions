
# @Author: Enea Duka
# @Date: 10/20/21
import sys
sys.path.append('/BS/unintentional_actions/work/unintentional_actions')
import os

from torch.utils.data import DataLoader
import torch.optim as optim
import torch
from crime_detection_i3d.rftm_model import Model
from crime_detection_i3d.i3d_crime_dataset import Dataset
from crime_detection_i3d.train_org import train
from crime_detection_i3d.test import test
import option
from tqdm import tqdm
from crime_detection_i3d.config import *
from utils.arg_parse import opt
from datetime import datetime
from utils.logging_setup import setup_logger_path
import torch.nn as nn
from utils.util_functions import Meter
from utils.util_functions import logger
from utils.model_saver import ModelSaver
import copy
from utils.util_functions import lr_func_cosine




if __name__ == '__main__':

    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    opt.dataset = 'crime'
    opt.rep_data_level = 'features'
    opt.backbone = 'resnet'
    opt.rep_backbone = 'resnet'
    opt.lr_scheduler = 'step'

    opt.embed_dim = 2048
    opt.intermediate_size = 3072
    opt.hidden_dim = 2048
    opt.num_classes = 8

    # opt.num_attention_heads = 2
    opt.attention_probs_dropout_prob = 0.1
    opt.hidden_dropout_prob = 0.1
    opt.num_hidden_layers = 1
    opt.attention_window = [8]

    # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/ucf_crime/VideoLongformer/ucf_crime.rep_learning.tag:speed&motion.layers3.attn_win32.classes3.time20211013-072446/val/top1/VideoLongformer__ucf_crime.rep_learning.tag:speed&motion.layers3.attn_win32.classes3.time20211013-072446_v0.4517_ep11.pth.tar'
    # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/crime/VideoLongformer_Crime/crime.crime_det.classification.layers3.attn_win32.taskclassification.time20211013-213552/val/top1/VideoLongformer_Crime__crime.crime_det.classification.layers3.attn_win32.taskclassification.time20211013-213552_v0.7236_ep1.pth.tar'
    # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/ucf_crime/VideoLongformer/ucf_crime.rep_learning.tag:speed&motion.layers3.attn_win32.classes8.time20211013-204741/val/top1/VideoLongformer__ucf_crime.rep_learning.tag:speed&motion.layers3.attn_win32.classes8.time20211013-204741_v0.4741_ep16.pth.tar'
    # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/crime/VideoLongformer_Crime/crime.crime_det.classification.layers3.attn_win32.taskclassification.time20211016-012731/val/top1/VideoLongformer_Crime__crime.crime_det.classification.layers3.attn_win32.taskclassification.time20211016-012731_v0.7466_ep1.pth.tar'
    # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/crime/VideoLongformer_Crime/crime.crime_det.classification.layers3.attn_win32.taskclassification.time20211018-104547/val/top1/VideoLongformer_Crime__crime.crime_det.classification.layers3.attn_win32.taskclassification.time20211018-104547_v0.7594_ep2.pth.tar'
    # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/oops/VideoLongformer_MLP/oops.unint_act.classification.layers3.attn_win32.time20210713-093910_cntr_loss/lr:0.000100~ep:40~bs:128~win:32~b_lr:1.000000~ptr:True_cntr_loss/val/top1/VideoLongformer_MLP__oops.unint_act.classification.layers3.attn_win32.time20210713-093910_cntr_loss_v0.6444_ep37.pth.tar'
    opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/crime/VideoLongformer_Crime_I3D/crime.crime_det_i3d.classification.layers3.attn_win32.taskclassification.time20211022-191409/val/top1/VideoLongformer_Crime_I3D__crime.crime_det_i3d.classification.layers3.attn_win32.taskclassification.time20211022-191409_v0.8302_ep255.pth.tar'
    opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/crime/VideoLongformer_Crime_I3D/crime.crime_det_i3d.classification.layers1.attn_win8.taskclassification.time20211025-133621/val/top1/VideoLongformer_Crime_I3D__crime.crime_det_i3d.classification.layers1.attn_win8.taskclassification.time20211025-133621_v0.8365_ep165.pth.tar'




    opt.model_name = 'VideoLongformer_Crime_I3D'
    opt.viz = False
    opt.test = True
    opt.num_workers = 32
    opt.batch_size = 64
    opt.sfx = str('%s.crime_det_i3d.%s.layers%d.attn_win%d.task%s.time%s' % (
        opt.dataset, opt.task, opt.num_hidden_layers, opt.attention_window[0], opt.task,
        datetime.now().strftime('%Y%m%d-%H%M%S')))
    opt.save_model = 1000
    opt.test_val = True
    opt.epochs = 50
    opt.gpu_parallel = False
    opt.use_tqdm = True
    opt.spat_temp = False
    opt.use_memory = False
    opt.pretrained = False
    opt.mlp_dropout = 0.8

    opt.optim = 'adam'
    opt.momentum = 0.9
    opt.lr = 1e-4
    opt.backbone_lr_factor = 0.01
    opt.cos_decay_lr_factor = 1e-2
    opt.weight_decay = 1e-2
    opt.test_freq = 1
    opt.save_model = 1
    opt.viz_env = '%s.%s%s_%s.' % (opt.model_name, opt.temp_learning_dataset_name, opt.env_pref, opt.sfx)

    opt.batch_size = 5
    opt.sample_videos = True
    opt.workers = 32
    opt.balance_fails_only = True
    opt.all_fail_videos = False
    # opt.val = True
    # opt.fails_path = '/BS/unintentional_actions/nobackup/oops/oops_dataset/oops_video'
    # val_loader = get_video_loader(opt)

    setup_logger_path()

    args = option.parser.parse_args()
    config = Config(args)

    train_nloader = DataLoader(Dataset(args, test_mode=False, is_normal=True),
                               batch_size=args.batch_size, shuffle=True,
                               num_workers=0, pin_memory=False, drop_last=True)
    train_aloader = DataLoader(Dataset(args, test_mode=False, is_normal=False),
                               batch_size=args.batch_size, shuffle=True,
                               num_workers=0, pin_memory=False, drop_last=True)
    test_loader = DataLoader(Dataset(args, test_mode=True),
                              batch_size=1, shuffle=False,
                              num_workers=0, pin_memory=False)

    model = Model(args.feature_size, args.batch_size)

    for name, value in model.named_parameters():
        print(name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model = nn.DataParallel(model)

    # if opt.pretrained:
    model_dict = torch.load(opt.vtn_ptr_path)['state_dict']
    model.load_state_dict(model_dict, strict=False)

    model_saver = ModelSaver(path=os.path.join(opt.storage, 'models', opt.dataset, opt.model_name, opt.sfx, opt.log_name, 'val'))


    optimizer = optim.Adam(model.parameters(),
                            lr=opt.lr, weight_decay=0.001)

    test_info = {"epoch": [], "test_AUC": []}
    best_AUC = -1
    # auc = test(test_loader, model)
    # print('Auc : %f' % auc)

    loss_meter = Meter('train')

    for step in tqdm(
            range(1, args.max_epoch + 1),
            total=args.max_epoch,
            dynamic_ncols=True
    ):
        if step > 1 and config.lr[step - 1] != config.lr[step - 2]:
            for param_group in optimizer.param_groups:
                param_group["lr"] = config.lr[step - 1]

        if (step - 1) % len(train_nloader) == 0:
            loadern_iter = iter(train_nloader)

        if (step - 1) % len(train_aloader) == 0:
            loadera_iter = iter(train_aloader)

        loss_val = train(loadern_iter, loadera_iter, model, args.batch_size, optimizer, device)
        loss_meter.update(loss_val, args.batch_size*2)

        if step % 5 == 0 and step > 0:

            auc = test(test_loader, model)
            test_info["epoch"].append(step)
            test_info["test_AUC"].append(auc)
            logger.debug("AUC: %f" % auc)
            loss_meter.log()
            loss_meter.reset()

            # new_lr = lr_func_cosine(opt.lr, opt.lr * opt.cos_decay_lr_factor, 1500, step)
            # logger.debug("New LR: %f" % new_lr)
            # optimizer.param_groups[0]['lr'] = new_lr
            # if len(optimizer.param_groups) > 1:
            #     optimizer.param_groups[1]['lr'] = new_lr * opt.backbone_lr_factor
            # for param_group in optimizer.param_groups:
            #     param_group['lr'] = new_lr



            # if auc > best_AUC:
            #     best_AUC = auc
            #     save_dict = {'epoch': step,
            #                  'state_dict': copy.deepcopy(model.state_dict()),
            #                  'optimizer': copy.deepcopy(optimizer.state_dict().copy())}
            #     model_saver.update({'top1': auc}, save_dict, step)
            #
            #     model_saver.save()

            if model_saver.check({'top1':auc}):
                save_dict = {'epoch': step,
                             'state_dict': copy.deepcopy(model.state_dict()),
                             'optimizer': copy.deepcopy(optimizer.state_dict().copy())}
                model_saver.update({'top1':auc}, save_dict, step)

            model_saver.save()

    #         if test_info["test_AUC"][-1] > best_AUC:
    #             best_AUC = test_info["test_AUC"][-1]
    #             torch.save(model.state_dict(), './ckpt/' + args.model_name + '{}-i3d.pkl'.format(step))
    #             save_best_record(test_info, os.path.join(output_path, '{}-step-AUC.txt'.format(step)))
    # torch.save(model.state_dict(), './ckpt/' + args.model_name + 'final.pkl')