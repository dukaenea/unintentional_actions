# @Author: Enea Duka
# @Date: 8/14/21

import sys
import numpy as np
import torch
from models.pm_vtn import create_model
from tqdm import tqdm
from models.gaussian_classifier import GaussianClassifier
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import pandas as pd
import seaborn as sb
from visualization.tsne_visualizer import do_tsne
from models import mlp_vae
from utils.arg_parse import opt
from utils.util_functions import Meter
from utils.logging_setup import logger
from sklearn.svm import OneClassSVM
from sklearn.metrics import f1_score, roc_curve, auc
from models.vit import create_vit_model


class AnomalyFeatureExtractor(object):
    def __init__(self, feat_save_path):
        self.feat_save_path = feat_save_path
        # self.feat_extractor = create_vit_model(pretrain=True)
        # self.feat_extractor.eval()
        self.model, _, _ = create_model(num_classes=None, pretrained=True)
        self.model.eval()
        self.head, self.optimizer, self.loss = mlp_vae.create_model(768, 128)
        self.train_features = []
        self.gaussian_classifier = None

    def train_ae(self, dataloader):
        loss_meter = Meter('train')
        for epoch in range(opt.epochs):
            for idx, data in enumerate(tqdm(dataloader)):
                self.optimizer.zero_grad()

                out = self._pass_through_model(data)
                out_ae = self.head(out.cuda())
                loss = self.loss(out_ae, out)
                loss_meter.update(loss.item(), out.shape[0])

                loss.backward()
                # torch.nn.utils.clip_grad_norm_(selfmodel.parameters(), 1)
                self.optimizer.step()
            logger.debug('Train Loss: %f' % loss_meter.avg)
            loss_meter.reset()

    def extract_feats(self, dataloader):
        with torch.no_grad():
            for idx, data in enumerate(tqdm(dataloader)):
                out = self._pass_through_model(data)
                # out = self.head(out, return_latent=True)
                for i, o in enumerate(out):
                    self.train_features.append(o)
                # self.train_features.append(out)
                # if idx == 10:
                #     break
        self.train_features = torch.stack(self.train_features).squeeze()
        self.gaussian_classifier = GaussianClassifier(self.train_features, 'mahalanobis')

    def train_svm(self, frame_dataloader, clip_dataloader):
        train_features = []
        with torch.no_grad():
            for idx, data in enumerate(tqdm(frame_dataloader)):
                # out = self._pass_through_model(data)
                # out = data['features'].squeeze()
                out = self._pass_through_model(data, backbone_only=True)
                # out = torch.tanh(out)
                for i, o in enumerate(out):
                    train_features.append(o)

        train_features = torch.stack(train_features).squeeze()

        # min = torch.min(train_features)
        # max = torch.max(train_features)
        # train_features = (train_features - min) / (max-min)

        train_features = train_features.cpu().numpy()
        self.f_oc_svm = OneClassSVM(kernel='rbf', gamma='auto', nu=0.5)
        self.f_oc_svm.fit(train_features)
        # self.max = np.max(self.oc_svm.decision_function(self.train_features))

        train_features = []

        with torch.no_grad():
            for idx, data in enumerate(tqdm(clip_dataloader)):
                # data['features'] = self.pass_clips_through_backbone(data['features'])
                out = self._pass_through_model(data)
                # out = data['features'].squeeze()
                # out = torch.tanh(out)
                for i, o in enumerate(out):
                    train_features.append(o)

        train_features = torch.stack(train_features).squeeze()

        # min = torch.min(train_features)
        # max = torch.max(train_features)
        # train_features = (train_features - min) / (max - min)

        train_features = train_features.cpu().numpy()
        self.c_oc_svm = OneClassSVM(kernel='rbf', gamma='auto', nu=0.5)
        self.c_oc_svm.fit(train_features)
        # # self.max = np.max(self.oc_svm.decision_function(self.train_features))

        return

    def eval_svm(self, frame_dataloader, clip_dataloader):
        labs = []
        outs = []
        np.set_printoptions(threshold=sys.maxsize)
        with torch.no_grad():
            for idx, data in enumerate(tqdm(frame_dataloader)):
                label = data['label']
                # out = self._pass_through_model(data)
                # out = data['features'].squeeze()
                out = self._pass_through_model(data, backbone_only=True)
                # out = torch.tanh(out)
                for i, o in enumerate(out):
                    outs.append(o)
                    labs.append(label[i])
        outs = torch.stack(outs).squeeze()

        # min = torch.min(outs)
        # max = torch.max(outs)
        # outs = (outs - min) / (max - min)

        outs = outs.cpu().numpy()
        f_labels = torch.stack(labs)
        f_labels = f_labels.numpy()
        f_labels[f_labels == 1] = -1
        f_labels[f_labels == 0] = 1
        # print(labels)
        frame_preds = self.f_oc_svm.predict(outs)
        frames_cm = confusion_matrix(f_labels, frame_preds, labels=[-1, 1])
        logger.debug(str(frames_cm))
        frame_y_scores = self.f_oc_svm.decision_function(outs)
        y_min = np.min(frame_y_scores)
        y_max = np.max(frame_y_scores)
        frame_n_score = (frame_y_scores - y_min) / (y_max - y_min)

        labs = []
        outs = []
        with torch.no_grad():
            for idx, data in enumerate(tqdm(clip_dataloader)):
                label = data['label']
                # data['features'] = self.pass_clips_through_backbone(data['features'])
                out = self._pass_through_model(data)
                # out = data['features'].squeeze()
                # out = torch.tanh(out)
                for i, o in enumerate(out):
                    outs.append(o)
                    labs.append(label[i])
        outs = torch.stack(outs).squeeze()

        # min = torch.min(outs)
        # max = torch.max(outs)
        # outs = (outs - min) / (max - min)

        outs = outs.cpu().numpy()
        c_labels = torch.stack(labs)
        c_labels = c_labels.numpy()
        c_labels[c_labels == 1] = -1
        c_labels[c_labels == 0] = 1
        # # print(labels)
        #
        clip_preds = self.c_oc_svm.predict(outs)
        clip_cm = confusion_matrix(c_labels, clip_preds, labels=[-1, 1])
        logger.debug(str(clip_cm))
        #
        # score = f1_score(frame_preds, clip_preds, pos_label=-1)
        #
        clip_y_scores = self.c_oc_svm.decision_function(outs)
        y_min = np.min(clip_y_scores)
        y_max = np.max(clip_y_scores)
        clip_n_score = (clip_y_scores - y_min) / (y_max - y_min)

        scores = self.fuse_scores(frame_preds, clip_preds, frame_n_score, clip_n_score)
        # scores = (clip_n_score + frame_n_score) / 2
        # eq = np.array_equal(f_labels, c_labels)
        #   print(eq)
        # print("F1 Score: %f" % score)
        # labels = torch.from_numpy(labels)
        # labels[labels == -1] = 0
        # labels = labels.numpy()
        # y_score = self.oc_svm.decision_function(outs)
        # y_min = np.min(y_score)
        # y_max = np.max(y_score)
        # n_score = (y_score - y_min) / (y_max - y_min)
        fpr, tpr, thresholds = roc_curve(f_labels, frame_n_score)
        auc_score = auc(fpr, tpr)

        logger.debug("AUC: %f" % auc_score)

        fpr, tpr, thresholds = roc_curve(c_labels, clip_n_score)
        auc_score = auc(fpr, tpr)

        logger.debug("AUC: %f" % auc_score)

        fpr, tpr, thresholds = roc_curve(f_labels, scores)
        auc_score = auc(fpr, tpr)

        logger.debug("AUC: %f" % auc_score)


    def fuse_scores(self, f_preds, c_preds, f_scores, c_scores):
        for i in range(len(f_preds)):
            # if f_preds[i] == 1 and c_preds[i] == -1:
            #     f_scores[i] = c_scores[i]
            if c_scores[i] > 0:
                f_scores[i] = c_scores[i]

        return f_scores


    def update_threshold(self, t):
        self.gaussian_classifier.update_threshold(t)

    def do_outlier_detection(self, dataloader):
        total = 0
        correct = 0
        gt = []
        pred = []
        data_vector = []
        with torch.no_grad():
            for idx, data in enumerate(tqdm(dataloader)):
                label = data['label']
                # if label == 2:
                #     continue
                out = self._pass_through_model(data)
                # out = self.head(out, return_latent=True)
                for i, o in enumerate(out):
                    c = self.gaussian_classifier(o)
                    total += 1
                    if label[i] == c:
                        correct += 1
                    gt.append(label[i].item())
                    pred.append(c)
                    data_vector.append(o)
                # if idx == 10:
                #     break
        # conf_mtx = confusion_matrix(gt, pred)
        print("Acc: %f" % (correct / total))
        # print(conf_mtx)
        # f1 = f1_score(gt, pred, average='binary')
        # print(f1)
        # self._plot_confussion_matrix(conf_mtx)
        my_conf_mtx = self._calcl_conf_mtrx(gt, pred)
        print(my_conf_mtx)
        # my_f1 = self._calc_f1_score(my_conf_mtx)
        # print(my_f1)
        tpr, fpr = self._calc_tpr_fpr(my_conf_mtx)
        # do_tsne(torch.stack(data_vector).cpu(), torch.tensor(gt).cpu(), "")
        return tpr, fpr

    def _pass_through_model(self, sample, backbone_only=False):
        videos = sample['features']
        pure_nr_frames = sample['pure_nr_frames']
        # video_names = sample['video_name']

        position_ids = torch.tensor(list(range(0, videos.shape[1]))) \
            .expand(1, videos.shape[1]) \
            .repeat(videos.shape[0], 1)
        out = self.model(videos, position_ids, None, pure_nr_frames, True, backbone_feats_only=backbone_only)

        return out.squeeze()

    def _calcl_conf_mtrx(self, gts, preds):
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        zeros = len([x for x in gts if x == 0])
        ones = len([x for x in gts if x == 1])
        for gt, pred in zip(gts, preds):
            if pred == 1 and gt == 1:
                tp += 1
            if pred == 0 and gt == 0:
                tn += 1
            if pred == 1 and gt == 0:
                fp += 1
            if pred == 0 and gt == 1:
                fn += 1
        return [[tp / ones, fn / ones], [fp / zeros, tn / zeros]]

    def _calc_f1_score(self, conf_mtx):
        tp = conf_mtx[0][0]
        fn = conf_mtx[0][1]
        fp = conf_mtx[1][0]
        tn = conf_mtx[1][1]

        precission = tp / (tp + fp)
        recall = tp / (tp + fn)

        return 2 * (precission * recall) / (precission + recall)

    def _calc_tpr_fpr(self, conf_mtx):
        tp = conf_mtx[0][0]
        fn = conf_mtx[0][1]
        fp = conf_mtx[1][0]
        tn = conf_mtx[1][1]

        tpr = tp / (tp + fn)
        print('TPR: %f' % tpr)
        fpr = fp / (fp + tn)
        print('FPR: %f' % fpr)

        return tpr, fpr

    def _plot_confussion_matrix(self, matrix):
        df_cm = pd.DataFrame(matrix, index=[i for i in ['Normal', 'Anomalous']],
                             columns=[i for i in ['Normal', 'Anomalous']])
        plt.figure(figsize=(10, 7))
        sb.heatmap(df_cm, annot=True)
        plt.show()

    def pass_frames_through_backbone(self, frames):
        return self.feat_extractor(frames)

    def pass_clips_through_backbone(self, clips):
        c, f, ch, w, h = clips.shape
        clips = clips.reshape(c*f, ch, w, h)

        clips = self.feat_extractor(clips)

        return torch.stack(list(torch.split(clips, 32)))