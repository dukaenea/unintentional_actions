# @Author: Enea Duka
# @Date: 4/16/21
import torch
import math
from tqdm import tqdm
from utils.logging_setup import logger
import torch.nn.functional as F
import torch.nn as nn
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, auc
# from torchmetrics import AUROC

class Labels:
    def __init__(self, path, nums=False):
        self.label2idx = {}
        self.idx2label = {}
        with open(path, 'r') as f:
            for line in f:
                idx, label = line.strip().split()
                idx = int(idx)
                if nums:
                    label = int(label)
                self.label2idx[label] = idx
                self.idx2label[idx] = label
        self._length = len(self.label2idx)

    def __getitem__(self, item):
        if isinstance(item, str):
            return self.label2idx[item]
        if isinstance(item, int):
            return self.idx2label[item]

    def __len__(self):
        return self._length


class Meter(object):
    def __init__(self, mode='', name=''):
        self.mode = mode
        self.name = name
        self.val, self.avg, self.sum, self.count = 0, 0, 0, 0

    def log(self):
        logger.debug('%s %s: %f' % (self.mode.upper(), self.name, self.avg))

    def viz_dict(self):
        return {
            '%s/%s' % (self.name, self.mode.upper()): self.avg
        }

    def reset(self):
        self.val, self.avg, self.sum, self.count = 0, 0, 0, 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class AIAYNScheduler:
    def __init__(self, d_model, warmup_steps=4000):
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.lr_base = 1e-8
    def step(self, optimizer, step):
        arg1 = (step + 1) ** -0.5
        arg2 = (step + 1) * (self.warmup_steps ** -1.5)
        new_lr = self.d_model ** -0.5 * min(arg1, arg2)
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.lr_base + new_lr
        return self.lr_base + new_lr

def compute_mean_and_std(dataloader):
    mean = torch.zeros((1, 3))
    mean_squared = torch.zeros((1, 3))
    num_videos = 0
    print(mean)
    for idx, data in enumerate(tqdm(dataloader)):
        features = data['features'].squeeze()
        try:
            features = features.permute(1, 0, 2, 3)
        except RuntimeError:
            print(features.shape)
            continue
        features = features.reshape(features.shape[0], -1)
        num_pixels = features.shape[1]
        num_videos += num_pixels
        sum_vid = features.sum(dim=1)
        sum_sq = features.pow(2).sum(dim=1)
        mean += sum_vid
        mean_squared += sum_sq

    # mean = mean / num_videos
    # std = torch.sqrt(mean_squared / num_videos - mean.pow(2))

    return mean, mean_squared, num_videos


def compute_mean_and_std_single_channel(dataloader):
    mean = 0
    mean_squared = 0
    num_videos = 0
    print(mean)
    for idx, data in enumerate(tqdm(dataloader)):
        features = data['video'].squeeze()
        # try:
        #     features = features.permute(1, 0, 2, 3)
        # except RuntimeError:
        #     print(features.shape)
        #     continue
        if len(features.shape) > 3:
            features = features.reshape(features.shape[0], -1)
            num_pixels = features.shape[1]
            num_videos += num_pixels
            sum_vid = features.sum(dim=1)
            sum_sq = features.pow(2).sum(dim=1)
        else:
            features = features.flatten()
            num_pixels = features.shape[0]
            num_videos += num_pixels
            sum_vid = features.sum(dim=0)
            sum_sq = features.pow(2).sum(dim=0)
        mean += sum_vid
        mean_squared += sum_sq

    # mean = mean / num_videos
    # std = torch.sqrt(mean_squared / num_videos - mean.pow(2))

    return mean, mean_squared, num_videos


def adjust_lr(optimizer, new_lr, decay_factor):
    """Decrease learning rate during training"""
    new_lr *= decay_factor
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
    return new_lr


def trn_label2idx(label_list):
    lab2idx = {'rotate': 1, 'shift': 2, 'scale': 3,
               'spacial_cutout': 4, 'random_crop': 5, 'flip': 6,
               'temporal_cutout': 7, 'fast_forward': 8, 'time_warp': 9,
               'temp_transform': -1}

    if len(label_list) == 0:
        return [0] + [-2] * (len(lab2idx.keys()) - 1)
    return [lab2idx[key] for key in label_list] + [-2] * (len(lab2idx.keys()) - len(label_list))


def label_idx_to_one_hot(label_idxs):
    tensor_labels = torch.stack(label_idxs).transpose(0, 1)
    batch_size = tensor_labels.shape[0]
    one_hot = torch.zeros((batch_size, 10))
    for idx, sample_label in enumerate(tensor_labels):
        filtered_label = [l.item() for l in sample_label if l != -2]
        one_hot[idx][filtered_label] = 1

    return one_hot


class Precision(object):
    def __init__(self, mode=''):
        self.mode = mode
        self._top1 = 0
        self._total = 0
        self.lab_class = None
        # self.auroc = AUROC(pos_label=1)

    def calculate_aucroc(self, outs, labels):
        # outs = torch.sigmoid(outs)
        # return self.auroc(outs, labels)
        outs = outs.cpu().numpy()
        labels = labels.cpu().numpy()

        fpr, tpr, thresholds = roc_curve(y_true=labels, y_score=outs, pos_label=1)
        return auc(fpr, tpr), fpr, tpr


    def update_probs_loc_class(self, outputs, labels, trn_times, clip_boundries):
        outs = torch.softmax(outputs, dim=1)
        best_trn_pred_idx = 0
        best_trn_pred = 0
        self._total += 1
        for idx, out in enumerate(outs):
            if out.argmax() != 1:
                continue
            if out[1] > best_trn_pred:
                best_trn_pred = out[1]
                best_trn_pred_idx = idx

        best_trn_bound = clip_boundries[best_trn_pred_idx]
        if best_trn_bound[0] <= trn_times <= best_trn_bound[1]:
            self._top1 += 1


    def update_probs_sig(self, outputs, labels):
        self._total += outputs.shape[0]
        for idx, label in enumerate(labels):
            ones_idx = ((label > 0).nonzero(as_tuple=True)[0])
            out_top_idx = torch.sort(torch.topk(outputs[idx], ones_idx.shape[0])[1])[0]
            if ones_idx.shape == out_top_idx.shape:
                if torch.all(ones_idx.eq(out_top_idx)):
                    self._top1 += 1

    def update_probs_reg_rel(self, outputs, lTrueabels):
        self._total += outputs.shape[0]
        for idx, label in enumerate(labels):
            if torch.abs(label - outputs[0]) < 0.05:
                self._top1 += 1


    def update_probs_reg(self, outputs, labels, lens):
        self._total += outputs.shape[0]
        for idx, label in enumerate(labels):
            # boundry_percentage = (1 / lens[idx]) * 1.5
            abs_time = outputs[idx] * lens[idx]
            if torch.abs(abs_time - label) <= 1.5:
                self._top1 += 1

    def update_probs_sfx(self, outputs, labels, report_pca=False, num_classes=0):
        self._total += outputs.shape[0]
        if report_pca and self.lab_class is None:
            self.lab_class = [None] * num_classes
        for idx, out in enumerate(outputs):
            out = F.softmax(out)
            max_idx = torch.argmax(out)
            if report_pca:
                if self.lab_class[labels[idx]] is None:
                    self.lab_class[labels[idx]] = {'total': 1, 'correct': 0}
                else:
                    self.lab_class[labels[idx]]['total'] += 1
            if max_idx == labels[idx]:
                self._top1 += 1
                if report_pca:
                    self.lab_class[labels[idx]]['correct'] += 1
        # outputs = F.softmax(outputs, dim=1)
        # max_idxs = torch.argmax(outputs, dim=1)
        # self._top1 += torch.sum((max_idxs==labels)).item()

    def update_probs_bc(self, output, labels):
        output = torch.sigmoid(output)
        output = torch.round(output)
        self._total += output.shape[0]
        self._top1 += (output == labels).sum()

    def calculate_auc(self, output, labels):
        fpr, tpr, threshold = roc_curve(labels, output)
        auc = roc_auc_score(labels, output)
        fnr = 1 - tpr
        eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
        plt.plot(fpr, tpr, label="data 1, auc=" + str(auc))
        plt.show()
        return auc, eer

    def update_probs_crf(self, out, labels, report_pca=False, num_classes=0):
        if report_pca and self.lab_class is None:
            self.lab_class = [None] * num_classes
        self._total += out.shape[0]
        self._top1 += (out == labels).sum()
        for idx, o in enumerate(out):
            max_idx = o
            if report_pca:
                if self.lab_class[labels[idx]] is None:
                    self.lab_class[labels[idx]] = {'total': 1, 'correct': 0}
                else:
                    self.lab_class[labels[idx]]['total'] += 1
            if max_idx == labels[idx]:
                if report_pca:
                    self.lab_class[labels[idx]]['correct'] += 1


    def top1(self, report_pca=False):
        if report_pca:
            class_prec = []
            for cl in self.lab_class:
                class_prec.append(cl['correct']/cl['total'])
            logger.debug(str(class_prec))
        return self._top1 / self._total


def lr_func_cosine(base_lr, end_lr, max_epochs, cur_epoch):
    """
    Retrieve the learning rate to specified values at specified epoch with the
    cosine learning rate schedule. Details can be found in:
    Ilya Loshchilov, and  Frank Hutter
    SGDR: Stochastic Gradient Descent With Warm Restarts.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        cur_epoch (float): the number of epoch of the current training stage.
    """
    assert end_lr < base_lr
    return (
        end_lr
        + (base_lr - end_lr)
        * (math.cos(math.pi * cur_epoch / max_epochs) + 1.0)
        * 0.5
    )

def consist_loss(out, org_out, org_idx):
    nr_out = out.shape[0]
    nr_org_out = org_out.shape[0]

    out = out.repeat_interleave(nr_org_out, dim=0)
    org_out = org_out.repeat(nr_out, 1)

    cos_dist = nn.CosineSimilarity(dim=1)
    distances = torch.stack(list((1 - cos_dist(out, org_out)).split(nr_org_out)))

    indicators = torch.zeros((nr_out, nr_org_out)).to(distances.device)
    for i in range(distances.shape[0]):
        for j in range(distances.shape[1]):
            if org_idx[i] == j:
                indicators[i, j] = 1
                break

    indicators.flatten()
    distances.flatten()
    loss = (indicators * distances ** 2 + (1-indicators) * torch.max(torch.zeros_like(indicators), 0.2 - distances) ** 2).sum()
    return loss

def contrastive_loss(out, dsets):
    out_c = torch.clone(out)
    nr_out = out.shape[0]

    out = out.repeat_interleave(nr_out, dim=0)
    out_c = out_c.repeat(nr_out, 1)

    cos_dist = nn.CosineSimilarity(dim=1)
    distances = torch.stack(list((1 - cos_dist(out, out_c)).split(nr_out)))

    indicators = torch.zeros((nr_out, nr_out)).to(distances.device)
    for i in range(distances.shape[0]):
        row_dset = dsets[i]
        for j in range(distances.shape[1]):
            if dsets[j] == row_dset:
                indicators[i][j] = 1

    indicators.flatten()
    distances.flatten()
    loss = (indicators * distances ** 2 + (1 - indicators) * torch.max(torch.zeros_like(indicators),
                                                                       0.2 - distances) ** 2).mean()
    return loss




class RegressionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.sl1 = nn.SmoothL1Loss(beta=0.5)

    def forward(self, outs, labels, lens):
        total_loss = torch.zeros(1).cuda()
        for idx, label in enumerate(labels):
            boundry_percentage = 1 / lens[idx]
            if torch.abs(outs[idx] - label) < boundry_percentage:
                continue
            else:
                total_loss += self.sl1(outs[idx], label)
        return total_loss

class DistributionPlotter(object):
    def __init__(self):
        self.bins = [0] * 11

    def update_bins(self, outs):
        for idx, out in enumerate(outs):
            idx = math.floor(out * 10)
            try:
                if self.bins[idx] is None:
                    self.bins[idx] = 1
                else:
                    self.bins[idx] += 1
            except Exception as e:
                print(e)

    def plot_out_dist(self):
        self.bins = [tb / sum(self.bins) for tb in self.bins]
        x = range(11)
        plt.bar(x, self.bins)
        plt.title('Train split relative action start time distribution')
        plt.xlabel('Relative video time')
        plt.ylabel('% Videos/Bin')
        plt.show()


def plot_valid_add_and_loss(acc, loss):
    # acc *= 100
    x = range(len(acc))
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    major_ticks = np.arange(0, max(max(loss), max(acc)), 1)
    minor_ticks = np.arange(0, max(max(loss), max(acc)), 0.01)

    ax.set_yticks(major_ticks)
    ax.set_yticks(minor_ticks, minor=True)

    ax.grid(which='both')

    ax.plot(x, acc, label='VAL Acc')
    ax.plot(x, loss, label='VAL Loss')

    plt.show()


def mil_objective(y_pred, y_true):
    labmdas = 8e-5

    normal_vid_idx = torch.where(y_true == 0)
    anomal_vid_idx = torch.where(y_true == 1)

    normal_vid_sc = y_pred[normal_vid_idx]
    anomal_vid_sc = y_pred[anomal_vid_idx]

    normal_segs_max_scores = normal_vid_sc.max(dim=-1)[0]
    anomal_segs_max_scores = anomal_vid_sc.max(dim=-1)[0]

    hinge_loss = 1 - anomal_segs_max_scores + normal_segs_max_scores
    hinge_loss = torch.max(hinge_loss, torch.zeros_like(hinge_loss))

    smoothed_scores = anomal_vid_sc[:, 1:] - anomal_vid_sc[:, :-1]
    smoothed_scores_ss = smoothed_scores.pow(2).sum(dim=-1)

    sparsity_loss = anomal_vid_sc.sum(dim=-1)

    final_loss = (hinge_loss + labmdas*smoothed_scores_ss + labmdas*sparsity_loss).mean()

    return final_loss


if __name__ == '__main__':
    acc = [0.0020, 0.0702, 0.9863, 0.3462, 0.3255, 0.3489, 0.5820, 0.3020, 0.8320, 0.2020]
    loss = [0.3255, 0.3489, 0.5820, 0.3020, 0.8320, 0.2020, 0.0020, 0.0702, 0.9863, 0.3462]

    out = torch.rand((5, 768))
    labels = torch.randint(0, 3, (5, ))
    mmargin_contrastive_loss(out.cuda(), labels.cuda())

    # # some data
    # orig = torch.rand(2, 5)
    # # where we want to put the data
    # # notice: idx.size() is equal orig.size()
    # # idx will be dimension zero index of the target))
    # idx = torch.LongTensor([[0, 1, 2, 0, 0], [2, 0, 0, 1, 2]])
    # # notice: t1.size(1) is equal orig.size(1)
    # t1 = torch.zeros(3, 5).scatter_(0, idx, orig)
    # print(t1)

    plot_valid_add_and_loss(acc, loss)
