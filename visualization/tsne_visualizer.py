# @Author: Enea Duka
# @Date: 8/23/21

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import numpy as np
import torch
import random
import pandas as pd
from utils.util_functions import Meter
from collections import OrderedDict
from matplotlib import interactive

def do_tsne(data, labels, title):
    label_set = [0, 1, 2]
    interactive(True)
    tsne = TSNE(n_components=2, random_state=0, n_iter=1000)
    pca = PCA(n_components=50)
    features_2d = pca.fit_transform(data)
    features_2d = tsne.fit_transform(features_2d)

    colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
    colors_keys = list(colors.keys())
    colors = ['g', 'b', 'r']
    plt.figure()
    plt.axis('off')
    # plt.title(title)
    for i, label in enumerate(label_set):
        label_idx = labels == label
        lab_feats = features_2d[label_idx]
        num_points = 2000 if label == 1 else 10000
        plt.scatter(lab_feats[:num_points, 0],
                    lab_feats[:num_points, 1],
                    c=colors[i],
                    label=str(label),
                    alpha=0.08)

    # plt.grid()
    # plt.show()
    plt.savefig('/BS/unintentional_actions/work/unintentional_actions/tsne.pdf')


def vis_tsne_base():
    base_split_path = '/BS/feat_augm/work/data/kinetics/data_splits'
    base_feats_paths = '/BS/feat_augm/work/data/kinetics/c3d_cnn_nn'
    labels_path = '/BS/feat_augm/work/data/kinetics/data_splits/base_classes.txt'
    labels_emb_path = '/BS/feat_augm/work/data/kinetics/label_embeddings'
    model_path = '/BS/feat_augm/work/storage/models/kt/proto_framework_attention/c3d.kt.proto_framework/bs1024.dp0.2.ep1200.video.!hf.lr0.001.nh2.nl1.snip0.adam.max.!ft0.sa-.cr:central.!ss.ta-.wd1e-05./val/top1/proto_framework_attention__c3d.kt.proto_framework_v0.5832_ep229.pth.tar'
    # model_path = '/BS/feat_augm/work/storage/models/kt/proto_framework_euc/c3d.kt.proto_framework/bs512.dp0.2.ep1200.video.!hf.lr0.001.snip0.adam.max.!ft0.sa-.cr:central.!ss.ta-.wd1e-05./val/top1/proto_framework_euc__c3d.kt.proto_framework_v0.6349_ep259.pth.tar'
    test_labels_path = '/BS/feat_augm/work/data/kinetics/data_splits/test_classes.txt'

    framework, optimizer, loss = create_framework(512, 2048, 256, [1e-3], 'cos', False)
    state_dict = torch.load(model_path)
    state_dict = state_dict['state_dict']
    framework.load_state_dict(state_dict)


    dataset = ProtoFeatureLoader(
        base_split_path,
        base_feats_paths,
        test_labels_path,
        labels_emb_path,
        'test',
        'pool',
        5,
        4
    )

    features, labels = dataset.get_all_features()
    _, features, _, _ = framework(None,
                                  features.to('cuda'),
                                  None,
                                  5)
    np_features = features.detach().cpu().numpy()
    labels = np.array(labels)

    do_tsne(np_features, labels, "Embedded Kinetics novel features")


def vis_tsne_novel():
    base_split_path = '/BS/feat_augm/work/data/kinetics/data_splits'
    base_feats_paths = '/BS/feat_augm/work/data/kinetics/c3d_cnn_nn'
    labels_path = '/BS/feat_augm/work/data/kinetics/data_splits/base_classes.txt'
    labels_emb_path = '/BS/feat_augm/work/data/kinetics/label_embeddings'
    episode_split_path = '/BS/feat_augm/work/data/kinetics/data_splits/test_episodes_5shot_5way.json'
    test_labels_path = '/BS/feat_augm/work/data/kinetics/data_splits/test_classes.txt'

    proto_episode_loader = ProtoEpisodicFeatureLoader(
        base_split_path,
        base_feats_paths,
        episode_split_path,
        test_labels_path,
        labels_emb_path,
        'test',
        'pool',
        5,
        5
    )

    model_path = '/BS/feat_augm/work/storage/models/kt/proto_framework_cos_all/c3d.kt.proto_framework/bs512.dp0.2.ep1200.video.!hf.lr0.001.snip0.adam.max.!ft0.sa-.cr:central.!ss.ta-.wd1e-05./val/top1/proto_framework_cos_all__c3d.kt.proto_framework_v0.6283_ep203.pth.tar'

    num_ep = 0
    for episode in proto_episode_loader.get_episodes():
        num_ep += 1
        if num_ep == 1 or num_ep == 0:
            continue
        if num_ep == 3:
            break

        framework, optimizer, loss = create_framework(512, 2048, 256, [1e-3], 'cos', False)
        state_dict = torch.load(model_path)
        state_dict = state_dict['state_dict']
        framework.load_state_dict(state_dict)

        train_dataset = episode['train']
        test_dataset = episode['test']
        train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                       batch_size=25,
                                                       shuffle=True,
                                                       num_workers=20,
                                                       drop_last=True,
                                                       pin_memory=True,
                                                       collate_fn=episode['train'].proto_collate_fn)

        train_features, train_labels, train_lex = train_dataset.get_all_features()
        test_features, test_labels, _ = test_dataset.get_all_features()
        features = torch.cat((train_features, test_features), dim=0).numpy()
        labels = np.array(train_labels + test_labels)

        # do_tsne(features, labels, 'Raw novel classes for episode # %s' % episode['key'])

        # pass the data through the model without fine tunning
        # framework.eval()
        # _, feats, _, _ = framework(
        #     None,
        #     torch.from_numpy(features).to('cuda'),
        #     None,
        #     5
        # )
        # feats = feats.detach().cpu().numpy()
        # do_tsne(feats, labels, 'Embedded novel classes for episode # %s' % episode['key'])

        # fine tune the network for some epochs
        for epoch in range(50):
            framework.train()
            loss_meter = Meter()
            for i, data in enumerate(train_dataloader):
                train_f = data['train_features'].to('cuda')
                query_features = data['query_features'].to('cuda')
                # train_f = torch.cat([elm[:3] for elm in torch.split(data['train_features'], 5)], dim=0).to(
                #     'cuda')
                # query_features = torch.cat([elm[2:] for elm in torch.split(data['query_features'], 5)], dim=0).to(
                #     'cuda')
                idxs = torch.randperm(5)[:5]
                # query_features = torch.cat([elm[idxs] for elm in torch.split(data['query_features'], 5)], dim=0).to(
                #     'cuda')
                labs = data['labels'].to('cuda')
                # labels = torch.cat([elm[4:] for elm in torch.split(data['labels'], 5)], dim=0).to(opt.device)
                label_embs = torch.stack([elm[0] for elm in torch.split(data['label_embs'], 5)], dim=0).to('cuda')

                prototypes, queries, _, _ = framework(train_f, query_features, label_embs, 5)
                batch_loss, dists, _ = loss(prototypes, queries, labs)
                loss_meter.update(batch_loss.item(), queries.shape[0])
                # prec.update_probs_proto(distances=dists, gt=data['labels'], mode='train')

                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

            framework.eval()
            _, feats, _, _ = framework(
                None,
                torch.from_numpy(features).to('cuda'),
                None,
                5
            )
            print('train feats shape')
            print(train_features.shape)
            split_feats = torch.split(train_features, 5)
            print('nr_splits feats shape')
            print(len(split_feats))
            trunc_feats = torch.cat([elm[:3] for elm in split_feats], dim=0).to('cuda')
            print('trunc feats shape')
            print(trunc_feats.shape)
            protos, _, _, _ = framework(
                train_features.to('cuda'),
                None,
                train_lex.to('cuda'),
                5
            )
            print('proto shape')
            print(protos.shape)
            # the protos will be sorted according to the labels
            proto_labels = list(OrderedDict.fromkeys(train_labels))
            proto_labels = np.array(proto_labels)
            proto_labels = np.array(proto_labels)

            feats = torch.cat((feats, protos), dim=0)
            feats = feats.detach().cpu().numpy()
            do_tsne(feats, labels, 'AFT embedded novel classes for episode # %s' % episode['key'], proto_labels, vis_protos=True)


if __name__ == '__main__':
    vis_tsne_novel()

    # a = [0, 0, 0, 0, 1, 1, 1, 1, 3, 3, 3, 4, 4, 4]
    # b = list(set(a))
