
# @Author: Enea Duka
# @Date: 9/20/21
import sys

sys.path.append('/BS/unintentional_actions/work/unintentional_actions')
import os
from dataloaders.my_oops_loader import get_video_loader
from dataloaders.ucf_crime_loader import get_crime_video_loader
from torch.utils.data import DataLoader
from models.pm_vtn import create_model, MILRegLoss
from utils.logging_setup import setup_logger_path
from utils.arg_parse import opt
from dataloaders.ucf_crime_mil_loader import CrimeFeatDataset, CrimeFeatDatasetVal
from crime_detection.train import train
import torch
from datetime import datetime


if __name__ == '__main__':

    # os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    opt.dataset = 'crime'
    opt.rep_data_level = 'features'
    opt.backbone = 'resnet'
    opt.rep_backbone = 'resnet'
    opt.lr_scheduler = 'step'

    opt.embed_dim = 768
    opt.intermediate_size = 3072
    opt.hidden_dim = 768
    opt.num_classes = 8

    opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/ucf_crime/VideoLongformer/ucf_crime.rep_learning.tag:speed&motion.layers3.attn_win32.classes3.time20211013-072446/val/top1/VideoLongformer__ucf_crime.rep_learning.tag:speed&motion.layers3.attn_win32.classes3.time20211013-072446_v0.4517_ep11.pth.tar'
    opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/crime/VideoLongformer_Crime/crime.crime_det.classification.layers3.attn_win32.taskclassification.time20211013-213552/val/top1/VideoLongformer_Crime__crime.crime_det.classification.layers3.attn_win32.taskclassification.time20211013-213552_v0.7236_ep1.pth.tar'
    opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/ucf_crime/VideoLongformer/ucf_crime.rep_learning.tag:speed&motion.layers3.attn_win32.classes8.time20211013-204741/val/top1/VideoLongformer__ucf_crime.rep_learning.tag:speed&motion.layers3.attn_win32.classes8.time20211013-204741_v0.4741_ep16.pth.tar'
    # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/crime/VideoLongformer_Crime/crime.crime_det.classification.layers3.attn_win32.taskclassification.time20211016-012731/val/top1/VideoLongformer_Crime__crime.crime_det.classification.layers3.attn_win32.taskclassification.time20211016-012731_v0.7466_ep1.pth.tar'
    # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/crime/VideoLongformer_Crime/crime.crime_det.classification.layers3.attn_win32.taskclassification.time20211018-104547/val/top1/VideoLongformer_Crime__crime.crime_det.classification.layers3.attn_win32.taskclassification.time20211018-104547_v0.7594_ep2.pth.tar'
    # opt.vtn_ptr_path = '/BS/unintentional_actions/work/storage/models/oops/VideoLongformer_MLP/oops.unint_act.classification.layers3.attn_win32.time20210713-093910_cntr_loss/lr:0.000100~ep:40~bs:128~win:32~b_lr:1.000000~ptr:True_cntr_loss/val/top1/VideoLongformer_MLP__oops.unint_act.classification.layers3.attn_win32.time20210713-093910_cntr_loss_v0.6444_ep37.pth.tar'


    opt.model_name = 'VideoLongformer_Crime'
    opt.viz = False
    opt.test = True
    opt.num_workers = 32
    opt.batch_size = 64
    opt.sfx = str('%s.crime_det.%s.layers%d.attn_win%d.task%s.time%s' % (
        opt.dataset, opt.task, opt.num_hidden_layers, opt.attention_window[0], opt.task, datetime.now().strftime('%Y%m%d-%H%M%S')))
    opt.save_model = 1
    opt.test_val = True
    opt.epochs = 50
    opt.gpu_parallel = True
    opt.use_tqdm = True
    opt.spat_temp = False
    opt.use_memory = False
    opt.pretrained = False
    opt.mlp_dropout = 0.6

    opt.optim = 'adam'
    opt.momentum = 0.9
    opt.lr = 1e-4
    opt.backbone_lr_factor = 0.01
    opt.cos_decay_lr_factor = 1e-2
    opt.weight_decay = 1e-2
    opt.test_freq = 1
    opt.save_model = 1
    opt.viz_env = '%s.%s%s_%s.' % (opt.model_name, opt.temp_learning_dataset_name, opt.env_pref, opt.sfx)

    opt.batch_size = 4
    opt.sample_videos = True
    opt.workers = 32
    opt.balance_fails_only = True
    opt.all_fail_videos = False
    oops_train_loader = get_video_loader(opt)
    # opt.val = True
    # opt.fails_path = '/BS/unintentional_actions/nobackup/oops/oops_dataset/oops_video'
    # val_loader = get_video_loader(opt)

    setup_logger_path()

    # opt.dataset_path = '/BS/unintentional_actions/nobackup/ucf_crime'
    # opt.val = True
    # opt.frames_per_clip = 4
    # opt.step_between_clips_sec = 1 / 30
    # opt.fps_list = [30]
    # opt.workers = 32
    # opt.batch_size = 16
    # crime_val_loader = get_crime_video_loader(opt)

    train_set = CrimeFeatDataset(iterations=800, clip_stride=1, frames_per_clip=16, segments_per_video=32)
    train_loader = DataLoader(train_set,
                              num_workers=opt.num_workers,
                              batch_size=opt.batch_size,
                              shuffle=False,
                              drop_last=True,
                              collate_fn=train_set._load_videos_collate_fn)

    # opt.dataset_path = '/BS/unintentional_actions/nobackup/ucf_crime'
    # opt.val = True
    # opt.frames_per_clip = 16
    # opt.step_between_clips_sec = 2
    # opt.fps_list = [30]
    # opt.workers = 32
    # opt.batch_size = 64
    # val_loader = get_crime_video_loader(opt)

    val_set = CrimeFeatDatasetVal(16)
    val_loader = DataLoader(val_set,
                            num_workers=32,
                            batch_size=1)



    model, optimizer, loss = create_model(num_classes=opt.num_classes, pretrained=opt.pretrained)
    # oops_model, _, _ = create_model(pretrained=True)
    loss = MILRegLoss(model)

    epoch = 0

    # if opt.pretrained:
    #     saved_model = torch.load(opt.vtn_ptr_path)
    #     optimizer.load_state_dict(saved_model['optimizer'])
    #     epoch = saved_model['epoch'] + 1


    train(model=model,
          train_loader=train_loader,
          val_loader=val_loader,
          optimizer=optimizer,
          loss=loss,
          test_freq=1,
          epochs=25,
          train_set=train_set,
          epoch=epoch)


    # loss_meter = Meter(mode='train', name='loss')
    # total = 0
    # correct = 0
    # correct_nrm = 0
    # correct_anrm = 0
    # total_nrm = 0
    # total_anrm = 0
    #
    #
    # fpc = 16
    # logger.debug('Temporal extent: %d' % fpc)
    # clip_train_loader = get_anomaly_loader('avenue', fpc, 1 / 25, 25, load_videos=False, load_frames=False)
    # clip_val_loader = get_anomaly_loader('avenue', fpc, 1 / 25, 25, load_videos=False, val=True, load_frames=False)
    #
    # for epoch in range(opt.epochs):
    #     oops_model.eval()
    #     crime_model.train()
    #     oops_iter = iter(oops_train_loader)
    #     for idx, data in enumerate(tqdm(clip_train_loader)):
    #         crime_videos = data['features']
    #         crime_pnf = data['pure_nr_frames'].squeeze()
    #         # crime_labels = data['label'].squeeze()
    #         # print(crime_labels)
    #
    #         crime_position_ids = torch.tensor(list(range(0, crime_videos.shape[1]))) \
    #             .expand(1, crime_videos.shape[1]) \
    #             .repeat(crime_videos.shape[0], 1)
    #         out_crime = crime_model(crime_videos, crime_position_ids, None, crime_pnf, return_features=True)
    #         # out_crime[crime_labels != 2]
    #         normal_crime = torch.nn.functional.normalize(out_crime)
    #         if normal_crime.shape[0] == 0:
    #             continue
    #
    #         try:
    #             oops_data = next(oops_iter)
    #         except StopIteration:
    #             oops_iter = iter(oops_train_loader)
    #             oops_data = next(oops_iter)
    #
    #         oops_videos = oops_data['features']
    #         oops_pnf = oops_data['pure_nr_frames'].squeeze()
    #         oops_labels = oops_data['label'].squeeze()
    #
    #         oops_videos = oops_videos[oops_labels != 2]
    #         oops_pnf = oops_pnf[oops_labels != 2]
    #         oops_labels = oops_labels[oops_labels != 2]
    #
    #         oops_position_ids = torch.tensor(list(range(0, oops_videos.shape[1]))) \
    #             .expand(1, oops_videos.shape[1]) \
    #             .repeat(oops_videos.shape[0], 1)
    #         out_oops = oops_model(oops_videos, oops_position_ids, None, oops_pnf, return_features=True)
    #         out_oops = torch.nn.functional.normalize(out_oops)
    #
    #         # crime_class = oops_model(out_crime, crime_position_ids, None, crime_pnf, classifier_only=True)
    #         #
    #         # for i, cl in enumerate(crime_class):
    #         #     cl = torch.softmax(cl, dim=0)
    #         #     total += 1
    #         #
    #         #     if crime_labels[i] == 0:
    #         #         total_nrm += 1
    #         #     else:
    #         #         total_anrm += 1
    #         #
    #         #     if crime_labels[i] == 0 and torch.argmax(cl) == 0:
    #         #         correct += 1
    #         #         correct_nrm += 1
    #         #
    #         #     if crime_labels[i] != 0 and torch.argmax(cl) != 0:
    #         #         correct += 1
    #         #         correct_anrm += 1
    #
    #         # if indicator is 0 it means that the pair is positive else it is negative
    #         indicators = torch.ones((out_oops.shape[0])).cuda()
    #         indicators[oops_labels != 0] = 0
    #         # rep_crime_out = torch.repeat_interleave(normal_crime, out_oops.shape[0], dim=0)
    #         # rep_oops_out = out_oops.repeat(normal_crime.shape[0], 1)
    #         indicators = indicators.repeat(normal_crime.shape[0], 1).flatten()
    #
    #         distances = torch.cdist(normal_crime.unsqueeze(0), out_oops.unsqueeze(0), p=2)
    #         distances = distances.squeeze().flatten()
    #         # cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    #         # distances = 1 - cos(rep_crime_out, rep_oops_out)
    #         p_samples = (indicators == 0).sum()
    #         n_samples = (indicators == 1).sum()
    #         loss_positive = (1 - indicators) * distances ** 2
    #         if n_samples > 0:
    #             loss_nevative =  indicators * torch.max(torch.zeros_like(indicators), 4 - distances) ** 2
    #             loss = (loss_positive + loss_nevative).mean()
    #         else:
    #             loss = loss_positive.mean()
    #
    #         if idx % 20 == 0:
    #             logger.debug('Loss: %f' % loss_meter.avg)
    #
    #         loss_meter.update(loss.item(), normal_crime.shape[0])
    #         loss.backward()
    #         torch.nn.utils.clip_grad_norm_(crime_model.parameters(), 1)
    #         optimizer.step()
    #     logger.debug('Train Loss: %f' % loss_meter.avg)
    #     loss_meter.reset()
    #
    #     crime_model.eval()
    #     val_loss_metter = Meter(mode='val', name='loss')
    #     for i, d in enumerate(tqdm(clip_val_loader)):
    #
    #         vids = d['features']
    #         pnf = d['pure_nr_frames']
    #         labs = d['label']
    #
    #         crime_pids = torch.tensor(list(range(0, vids.shape[1]))) \
    #             .expand(1, vids.shape[1]) \
    #             .repeat(vids.shape[0], 1)
    #         out_crime = crime_model(vids, crime_pids, None, pnf, return_features=True)
    #
    #         n_crime = torch.nn.functional.normalize(out_crime[labs == 0])
    #
    #         if n_crime.shape[0] != 0:
    #             oops_d = next(oops_iter)
    #             oops_v = oops_d['features']
    #             oops_pnf = oops_d['pure_nr_frames']
    #             oops_l = oops_d['label']
    #
    #             oops_position_ids = torch.tensor(list(range(0, oops_v.shape[1]))) \
    #                 .expand(1, oops_v.shape[1]) \
    #                 .repeat(oops_v.shape[0], 1)
    #             out_oops = oops_model(oops_v, oops_position_ids, None, oops_pnf, return_features=True)
    #             out_oops = torch.nn.functional.normalize(out_oops)
    #
    #             indicators = torch.ones((out_oops.shape[0])).cuda()
    #             indicators[oops_l != 0] = 0
    #             # rep_crime_out = torch.repeat_interleave(normal_crime, out_oops.shape[0], dim=0)
    #             # rep_oops_out = out_oops.repeat(normal_crime.shape[0], 1)
    #             indicators = indicators.repeat(n_crime.shape[0], 1).flatten()
    #
    #             distances = torch.cdist(n_crime, out_oops, p=2)
    #             distances = distances.flatten()
    #             # cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    #             # distances = 1 - cos(rep_crime_out, rep_oops_out)
    #             p_samples = (indicators == 0).sum()
    #             n_samples = (indicators == 1).sum()
    #             loss_positive = (1 - indicators) * distances ** 2
    #             if n_samples > 0:
    #                 loss_nevative = indicators * torch.max(torch.zeros_like(indicators), 2 - distances) ** 2
    #                 loss = (loss_positive + loss_nevative).mean()
    #             else:
    #                 loss = loss_positive.mean()
    #
    #             # if idx % 20 == 0:
    #             #     logger.debug('Loss: %f' % loss_meter.avg)
    #
    #             val_loss_metter.update(loss.item(), n_crime.shape[0])
    #             if math.isnan(val_loss_metter.avg):
    #                 print('here')
    #
    #         crime_class = oops_model(out_crime, crime_pids, None, pnf, classifier_only=True)
    #
    #         for i, cl in enumerate(crime_class):
    #             cl = torch.softmax(cl, dim=0)
    #             total += 1
    #
    #             if labs[i] == 0:
    #                 total_nrm += 1
    #             else:
    #                 total_anrm += 1
    #
    #             if labs[i] == 0 and torch.argmax(cl) == 0:
    #                 correct += 1
    #                 correct_nrm += 1
    #
    #             if labs[i] != 0 and torch.argmax(cl) != 0:
    #                 correct += 1
    #                 correct_anrm += 1
    #             # logger.debug("Label %d, Predicton %d" % (crime_labels[idx], torch.argmax(cl)))
    #     # if idx % 100 == 0:
    #     logger.debug('Val Loss: %f' % val_loss_metter.avg)
    #     logger.debug("Accuracy: %f" % (correct / total))
    #     logger.debug("Normal accuracy: %f" % (correct_nrm / total_nrm))
    #     logger.debug("Anomalous accuracy: %f" % (correct_anrm / total_anrm))