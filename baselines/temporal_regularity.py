# @Author: Enea Duka
# @Date: 4/29/21
import sys

sys.path.append('/BS/unintentional_actions/work/unintentional_actions')
import os
import copy
import torch
import numpy as np
from datetime import datetime
from dataloaders.pedestrian_loader import PedestrianDataset
from dataloaders.avenue_loader import AvenueDataset, Label_loader
from torch.utils.data import DataLoader
from models.ConvVAE import create_model
from utils.util_functions import Meter
from tqdm import tqdm
from matplotlib import pyplot as plt
from persistance1d_impl.persistence1d import RunPersistence
from utils.logging_setup import logger
from utils.arg_parse import opt
from utils.model_saver import ModelSaver
from utils.plotting_utils import visdom_plot_losses
from utils.logging_setup import setup_logger_path
from utils.logging_setup import viz


def train(**kwargs):
    model = kwargs['model']
    optimizer = kwargs['optimizer']
    loss = kwargs['loss']
    dataloader = kwargs['train_dataloader']
    tst = datetime.now().strftime('%Y%m%d-%H%M%S')

    loss_meter = Meter(mode='train')
    model_saver = ModelSaver(
        path=os.path.join(opt.storage, 'models', opt.dataset, opt.model_name, opt.sfx, opt.log_name, 'val'))

    logger.debug('Starting training for %d epochs:' % opt.epochs)
    for epoch in range(opt.epochs):
        epoch_loss = 0.0
        model.train()
        for i, data in enumerate(tqdm(dataloader)):
            videos = data['video'].cuda()
            optimizer.zero_grad()

            outputs = model(videos)
            _loss = loss(outputs, videos) / (2 * opt.batch_size)
            loss_meter.update(_loss.item(), videos.shape[0])
            _loss.backward()
            optimizer.step()
        logger.debug('Loss: %f' % loss_meter.avg)
        # visdom_plot_losses(viz.env, opt.log_name + '-train-loss-' + str(tst), epoch,
        #                    xylabel=('epoch', 'loss'), **loss_meter.viz_dict())
        loss_meter.reset()
        check_val = None
        if opt.test_freq and epoch % opt.test_freq == 0:
            check_val = test(mode='val',
                             p1d_threshold=0.2,
                             model=model,
                             dataloader=kwargs['test_dataloader'],
                             epoch=epoch,
                             time_id=tst)

        if opt.save_model and epoch % opt.save_model == 0 and check_val:
            if model_saver.check(check_val):
                save_dict = {'epoch': epoch,
                             'state_dict': copy.deepcopy(model.state_dict()),
                             'optimizer': copy.deepcopy(optimizer.state_dict().copy())}
                model_saver.update(check_val, save_dict, epoch)

            model_saver.save()


def plot_reg_scores(reg_scores, anom_segments=None, scatter=False):
    x_inx = np.arange(0, reg_scores.shape[0])
    y_vals = reg_scores.detach().cpu().numpy()
    if scatter:
        plt.scatter(x_inx, y_vals)
    else:
        plt.plot(x_inx, y_vals)
    if anom_segments is not None:
        for segment in anom_segments:
            plt.axvspan(segment[0], segment[1], color='orange', alpha=0.5)
    plt.show()


def comp_extrema_and_persistance(reg_scores, threshold, window):
    reg_scores = reg_scores.detach().cpu().numpy()
    ext_per = RunPersistence(reg_scores)
    ext_per = sorted(ext_per, key=lambda ExtremumAndPersistence: ext_per[0])
    filtered_ext_per = [t for t in ext_per if t[1] > threshold]
    anom_segments = merge_anomalous_segments([t[0] for t in filtered_ext_per], reg_scores.shape[0], window)
    return anom_segments


def get_gt_segments(labels):
    gt_segments = []
    gn_segments = []
    start = -1
    for idx, label in enumerate(labels):
        if label == 0 and start == -1:
            continue
        if start == -1:
            start = idx
            continue
        if (label == 0 or idx == len(labels) - 1) and start != -1:
            gt_segments.append([start, idx - 1])
            start = -1

    for idx, g_seg in enumerate(gt_segments):
        if idx == 0 and g_seg[0] > 0:
            gn_segments.append([0, g_seg[0]-1])
        if idx == len(gt_segments) -1 and g_seg[1] < len(labels) -1:
            gn_segments.append([g_seg[1]+1, len(labels) - 1])
            break
        gn_segments.append([gt_segments[idx-1][1]+1, g_seg[0]-1])


    return gt_segments, gn_segments


def claculate_tp_fp(labels, anom_segments):
    gt_segments, gn_segments = get_gt_segments(labels)
    tp = 0
    fp = 0
    for a_seg in anom_segments:
        # go over all the gt and test
        for t_seg in gt_segments:
            # a_seg includes t_seg in it
            if a_seg[0] <= t_seg[0] < t_seg[1] <= a_seg[1]:
                # check if t_seg covers 50% of a_seg
                if (t_seg[1] - t_seg[0]) >= (a_seg[1] - a_seg[0]) // 2:
                    tp += 1
                else:
                    fp += 1
            # t_seg includes a_seg
            elif t_seg[0] <= a_seg[0] < a_seg[1] <= t_seg[1]:
                # check if a_seg covers 50% of t_seg
                if (a_seg[1] - a_seg[0]) >= (t_seg[1] - t_seg[0]) // 2:
                    tp += 1
                else:
                    fp += 1
            # intersection on the left of a_seg and right of t_seg:
            elif a_seg[0] < t_seg[1] < a_seg[0]:
                coverage = t_seg[1] - a_seg[0]
                if coverage >= (a_seg[1] - a_seg[0]) // 2 and coverage >= (t_seg[1] - t_seg[0]) // 2:
                    tp += 1
                else:
                    fp += 1
            # intersection on the right of a_seg and left of t_seg:
            elif a_seg[0] < t_seg[1] < a_seg[1]:
                coverage = t_seg[1] - a_seg[0]
                if coverage >= (a_seg[1] - a_seg[0]) // 2 and coverage >= (t_seg[1] - t_seg[0]) // 2:
                    tp += 1
                else:
                    fp += 1
    return tp, fp, tp/len(gt_segments)


def calculate_acc(labels, anom_segments):
    gt_segments, _ = get_gt_segments(labels)
    correct = 0
    tp = 0
    fp = 0
    for gt_segment in gt_segments:
        covered_frames = 0
        for anom_segment in anom_segments:
            # they intersect on the left of the gt
            if anom_segment[1] > gt_segment[0] > anom_segment[0]:
                covered_frames += anom_segment[1] - gt_segment[0]
            # they intersect on the right of the gt
            if anom_segment[0] < gt_segment[1] < anom_segment[1]:
                covered_frames += gt_segment[1] - anom_segment[0]
            # anom is inside gt
            if anom_segment[0] >= gt_segment[0] and anom_segment[1] <= gt_segment[1]:
                covered_frames += anom_segment[1] - anom_segment[0]
            # gt is inside anom
            if anom_segment[0] <= gt_segment[0] and anom_segment[1] >= gt_segment[1]:
                covered_frames += gt_segment[1] - gt_segment[0]
        if covered_frames >= (gt_segment[1] - gt_segment[0]) // 2:
            tp += 1
        else:
            fp += 1
    return tp, fp, tp/len(gt_segments)


def merge_anomalous_segments(center_points, max_len, seg_width):
    segments = []
    center_points = sorted(center_points)
    for point in center_points:
        seg_start = max(0, point - seg_width // 2)
        seg_end = min(max_len, point + seg_width // 2)
        if len(segments) == 0:
            segments.append([seg_start, seg_end])
        else:
            last_segment = segments[-1]
            if last_segment[1] >= seg_start:
                segments[-1] = [last_segment[0], seg_end]
            else:
                segments.append([seg_start, seg_end])
    return segments


def test(load_pretrain=False, p1d_threshold=0.2, **kwargs):
    model = kwargs['model']
    loss = torch.nn.MSELoss(reduction='none')
    dataloader = kwargs['dataloader']
    mode = kwargs['mode']
    epoch = kwargs['epoch']
    time_id = kwargs['time_id']
    meter = Meter(mode=mode, name='loss')

    # if load_pretrain:
    #     model_path = opt.ptr_tmpreg_model_path
    #     model_dict = torch.load(model_path)
    #     model.load_state_dict(model_dict)
    model.eval()
    total_acc = 0.0
    total = 0
    tps = 0
    fps = 0
    with torch.no_grad():
        for idx, data in enumerate(tqdm(dataloader)):
            video = data['video'].squeeze()
            # in this case the video is in chunks of 10 frames
            #  we feed the whole video to the model to get the reconstruction
            rec = model(video.cuda())
            _loss = loss(rec, video.cuda())
            # print(_loss.shape)
            meter.update(_loss.sum(3).sum(2).sum(1).mean(0).item(), 1)
            # get avg over spatial axes to get the error for each frame
            frame_loss = _loss.sum(3).sum(2)
            frame_loss = frame_loss.flatten()
            min_frame_loss = frame_loss.min()
            max_frame_loss = frame_loss.max()
            reg_scores = 1 - ((frame_loss - min_frame_loss) / max_frame_loss)
            anom_segments = comp_extrema_and_persistance(reg_scores, p1d_threshold, 50)
            # if p1d_threshold == 0.0:
            # plot_reg_scores(reg_scores, anom_segments)
            # plot_reg_scores(data['label'].squeeze())
            tp, fp, acc = claculate_tp_fp(data['label'].squeeze(), anom_segments)
            tps += tp
            fps += fp
            total_acc += acc
            total += 1
        meter.log()
        print('%d === %d' % (tps, fps))
        print("Acc: %f" % (total_acc / total))
        #
        # if opt.viz and epoch % opt.viz_freq == 0:
        #     visdom_plot_losses(viz.env, opt.log_name + '-loss-' + str(time_id), epoch,
        #                        xylabel=('epoch', 'loss'), **meter.viz_dict())
        #     visdom_plot_losses(viz.env, opt.log_name + '-prec-' + str(time_id), epoch,
        #                        xylabel=('epoch', 'prec@1'), **{'pr@1/%s' % mode.upper(): (total_acc / total)})
        #
        return {'top1': (total_acc / total)}


def setup_model_storage(storage_path, dataset_name):
    project_storage_path = storage_path + 'temporal_regularity/' + dataset_name
    subdirs = os.listdir(project_storage_path)
    if len(subdirs) > 0:
        subdirs = sorted(subdirs)
        last_dir = subdirs[-1]
        dir_name = last_dir.split('/')[-1][1:]
        next_dir = project_storage_path + '/v%d/' % (int(dir_name) + 1)
    else:
        next_dir = project_storage_path + '/v%d/' % 0

    project_storage_path = next_dir

    if not os.path.isdir(project_storage_path):
        os.makedirs(project_storage_path)
    return project_storage_path


def calc_roc(**kwargs):
    ptr_model = kwargs['model']
    dataloader = kwargs['dataloader']
    steps = kwargs['steps']
    time_id = datetime.now().strftime('%Y%m%d-%H%M%S')

    # calculate the tpr and fpr for each step
    for i in range(steps):
        p1d_threshold = (1 / steps) * i
        print(p1d_threshold)
        tp = test(load_pretrain=True,
                  p1d_threshold=p1d_threshold,
                  mode='val',
                  model=ptr_model,
                  dataloader=dataloader,
                  epoch=-1,
                  time_id=''
                  )
        print('tp')
        # tpr = tp / (tp + fn)
        # fpr = fp / (fp + tn)
        #
        # visdom_plot_losses(viz.env, opt.log_name + '-roc-' + str(time_id), fpr,
        #                    xylabel=('fpr', 'tpr'), **{'tpr': tpr})


if __name__ == '__main__':

    opt.temp_learning_dataset_name = 'avenue'
    opt.num_in_channels = 20
    opt.spatial_size = 215
    opt.batch_size = 1
    if opt.temp_learning_dataset_name == 'ped':
        opt.ptr_tmpreg_model_path = "/BS/unintentional_actions/work/storage/models/kt/ConvVAE/ped.conv_vae.num_in_frames20/val/top1/ConvVAE__ped.conv_vae.num_in_frames20_v0.8021_ep1105.pth.tar"

    opt.model_name = 'ConvVAE'
    opt.viz = False
    opt.save_model = False
    opt.test = True
    opt.num_workers = 16
    opt.sfx = str('%s.conv_vae.num_in_frames%d' % (opt.temp_learning_dataset_name, opt.num_in_channels))
    opt.save_model = 1
    opt.test_val = True
    opt.epochs = 3000

    opt.optim = 'adam'
    opt.momentum = 0.9
    opt.lr = 1e-5
    opt.weight_decay = 1e-5
    opt.test_freq = 20
    opt.save_model = 1
    opt.viz_env = '%s.%s%s_%s.' % (opt.model_name, opt.temp_learning_dataset_name, opt.env_pref, opt.sfx)
    setup_logger_path()

    # project_storage_path = setup_model_storage(storage_path, opt.temp_learning_dataset_name)

    if opt.temp_learning_dataset_name == 'ped':
        train_set = PedestrianDataset('Train', gray_scale=True, in_channels=opt.num_in_channels)
        test_set = PedestrianDataset('Test', gray_scale=True, in_channels=opt.num_in_channels)
    elif opt.temp_learning_dataset_name == 'avenue':
        train_set = AvenueDataset('train',
                                  (opt.spatial_size, opt.spatial_size),
                                  '/BS/unintentional_actions/nobackup/avenue/avenue/training',
                                  in_channels=opt.num_in_channels,
                                  gray_scale=True)
        test_set = AvenueDataset('test',
                                 (opt.spatial_size, opt.spatial_size),
                                 '/BS/unintentional_actions/nobackup/avenue/avenue/testing',
                                 in_channels=opt.num_in_channels,
                                 gray_scale=True)
        label_loader = Label_loader(
            sorted([x[0] for x in os.walk('/BS/unintentional_actions/nobackup/avenue/avenue/testing')][1:]))

    train_dataloader = DataLoader(
        train_set,
        num_workers=16,
        batch_size=32,
        shuffle=True,
        pin_memory=True,
        collate_fn=train_set.tem_reg_collate_fn
    )

    test_dataloader = DataLoader(
        test_set,
        num_workers=0,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        collate_fn=test_set.tem_reg_inference_collate_fn
    )
    # print(len(train_dataloader))
    # for idx, data in enumerate(test_dataloader):
    #     videos = data['video']

    # train_stats = compute_mean_and_std_single_channel(train_dataloader)
    # test_stats = compute_mean_and_std_single_channel(test_dataloader)
    #
    # all_stats = train_stats + test_stats
    # mean = all_stats[0] / all_stats[2]
    # std = torch.sqrt((all_stats[1] / all_stats[2]) - mean.pow(2))
    #
    # print(mean, std)

    model, optimizer, loss = create_model(opt.num_in_channels, opt.lr, opt.optim, opt.weight_decay, pretrained=False)
    # calc_roc(model=model,
    #          dataloader=test_dataloader,
    #          steps=10)
    train(model=model,
          optimizer=optimizer,
          loss=loss,
          train_dataloader=train_dataloader,
          test_dataloader=test_dataloader)

    # test(model=model,
    #      dataloader=test_dataloader,
    #      model_path='/BS/unintentional_actions/work/storage/temporal_regularity/v8/1619960231.258286_62754.21875.pth')
