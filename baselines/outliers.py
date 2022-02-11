
# @Author: Enea Duka
# @Date: 4/25/21
import sys
sys.path.append('/BS/unintentional_actions/work/unintentional_actions')
from dataloaders.pedestrian_loader import PedestrianDataset
from dataloaders.avenue_loader import AvenueDataset, test_dataset, Label_loader
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error
from kmeans_pytorch import kmeans
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import torch
import time
import os


def normalize_data(data):
    t_data = torch.stack(data)
    mean = t_data.mean()
    std = t_data.std()

    return (t_data - mean) / std

def get_eigen_vectors(training_data):
    pca = PCA(0.99)
    training_data = StandardScaler().fit_transform(training_data)
    pca_features = pca.fit_transform(training_data)
    return pca.components_

def get_train_reconstruction_error(training_data):
    eigen_vectors = torch.from_numpy(get_eigen_vectors(training_data))
    if eigen_vectors.shape[0] != training_data.shape[1]:
        eigen_vectors_t = torch.transpose(eigen_vectors, 0, 1)
    else:
        eigen_vectors_t = eigen_vectors
    pro_data = torch.mm(training_data, eigen_vectors_t)
    rec_data = torch.mm(pro_data, eigen_vectors)
    mse = mean_squared_error(training_data.detach().numpy(), rec_data.detach().numpy(), multioutput='raw_values')
    print(min(mse))
    print(max(mse))

def get_test_reconstruction_error(test_data, eigen_vectors):
    if eigen_vectors.shape[0] != test_data.shape[1]:
        eigen_vectors_t = torch.transpose(eigen_vectors, 0, 1)
    else:
        eigen_vectors_t = eigen_vectors
    pro_data = torch.mm(test_data, eigen_vectors_t)
    if eigen_vectors.shape[0] != pro_data.shape[1]:
        eigen_vectors = torch.transpose(eigen_vectors, 0, 1)

    rec_data = torch.mm(pro_data, eigen_vectors)
    mse = mean_squared_error(test_data.detach().numpy(), rec_data.detach().numpy(), multioutput='raw_values')
    print(min(mse))
    print(max(mse))

def find_outliers(dataset_name):
    if dataset_name == 'ped':
        # use the ImageNet stats
        train_set = PedestrianDataset('Train', return_hog=True)
        test_set = PedestrianDataset('Test', return_hog=True)
    elif dataset_name == 'avenue':
        train_set = AvenueDataset((256, 256), '/BS/unintentional_actions/nobackup/avenue/avenue/training')
        test_set = AvenueDataset((256, 256), '/BS/unintentional_actions/nobackup/avenue/avenue/testing')
        label_loader = Label_loader(sorted([x[0] for x in os.walk('/BS/unintentional_actions/nobackup/avenue/avenue/testing')][1:]))

    train_dataloader = DataLoader(
        train_set,
        num_workers=32,
        batch_size=1,
        shuffle=False
    )

    test_dataloader = DataLoader(
        test_set,
        num_workers=32,
        batch_size=1,
        shuffle=False
    )

    frames = []
    labels = []

    for _, data in enumerate(tqdm(test_dataloader)):
        videos = data['video']
        # merge all frames and flatten them
        videos = videos.reshape(videos.shape[0] * videos.shape[1], -1)
        frames.extend(videos)
        if dataset_name != 'avenue':
            labels.extend(data['label'][0].tolist())
    if dataset_name == 'avenue':
        labels.extend(label_loader.load_ucsd_avenue())

    frames = normalize_data(frames)
    principal_components = get_eigen_vectors(frames)
    get_train_reconstruction_error(frames)
    frames = []
    labels = []

    for _, data in enumerate(tqdm(train_dataloader)):
        videos = data['video']
        # merge all frames and flatten them
        videos = videos.reshape(videos.shape[0]*videos.shape[1], -1)
        frames.extend(videos)
        labels.extend([0]*videos.shape[0])


    principal_components = torch.from_numpy(principal_components)
    frames = normalize_data(frames)

    frames = torch.matmul(frames, torch.transpose(principal_components, 0, 1))

    get_test_reconstruction_error(frames, principal_components)

    plt.scatter(frames[:-1, 0],
                frames[:-1, 1],
                alpha=0.01)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.show()
    colors_gt = list(map(lambda x: 'blue' if x == 0 else 'red', labels))
    plt.scatter(frames[:-1, 0],
                frames[:-1, 1],
                color=colors_gt[:-1],
                alpha=0.01)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.show()

    components = []
    times = []
    vars_explained = []
    accs = []
    # features_2d = None

    # for i in range(500, 520):
    #     print('Performing PCA...')
    #     pca = PCA(i)
    #     std_data = StandardScaler().fit_transform(frames)
    #     start = time.time()
    #     pca_features = pca.fit_transform(std_data)
    #     end = time.time()
    #     pca_features = torch.from_numpy(pca_features)
    #     components.append(i)
    #     times.append(end-start)
    #     vars_explained.append(pca.explained_variance_ratio_.cumsum()[-1])
    #
    #     print('Performing KMeans clustering...')
    #     cluster_ids_x, cluster_centers = kmeans(
    #         X=pca_features, num_clusters=2, distance='euclidean', device=torch.device('cuda:0')
    #     )
    #     num_zeros = cluster_ids_x.tolist().count(0)
    #     num_ones = cluster_ids_x.tolist().count(1)
    #     nrm_color_indicator = 1 if num_ones > num_zeros else 0
    #     anm_color_indicator = 0 if num_ones > num_zeros else 1
    #     colors = list(map(lambda x: 'blue' if x == nrm_color_indicator else 'red', cluster_ids_x.tolist()))
    #     colors_gt = list(map(lambda x: 'blue' if x == 0 else 'red', labels))
    #     print(len(labels))
    #     print(len(pca_features))
    #     print(pca_features.shape)
    #     if i == 1:
    #         plt.scatter(pca_features,
    #                     np.zeros(len(pca_features)),
    #                     color=colors,
    #                     alpha=0.01)
    #         plt.xlabel('PC1')
    #         plt.show()
    #         plt.scatter(pca_features,
    #                     np.zeros(len(pca_features)),
    #                     color=colors_gt,
    #                     alpha=0.01)
    #         plt.xlabel('PC1')
    #         plt.show()
    #     if i == 2:
    #         plt.scatter(pca_features[:-1, 0],
    #                     pca_features[:-1, 1],
    #                     color=colors[:-1],
    #                     alpha=0.01)
    #         plt.xlabel('PC1')
    #         plt.ylabel('PC2')
    #         plt.show()
    #         plt.scatter(pca_features[:-1, 0],
    #                     pca_features[:-1, 1],
    #                     color=colors_gt[:-1],
    #                     alpha=0.01)
    #         plt.xlabel('PC1')
    #         plt.ylabel('PC2')
    #         plt.show()
    #
    #
    #
    #     total = 0
    #     correct = 0
    #     c_ids = cluster_ids_x.tolist()
    #     print(len(labels))
    #     print(pca_features.shape)
    #     for i in range(len(labels)):
    #         total += 1
    #         if num_zeros < num_ones:
    #             if (labels[i] == 1 and c_ids[i] == 0) or (labels[i] == 0 and c_ids[i] == 1):
    #                 correct += 1
    #         else:
    #             if (labels[i] == 0 and c_ids[i] == 0) or (labels[i] == 1 and c_ids[i] == 1):
    #                 correct += 1
    #
    #     print(correct/total)
    #     accs.append(correct/total)
    #
    #
    # plt.scatter(
    #     components,
    #     times
    # )
    # plt.xlabel('Num. Components')
    # plt.ylabel('Execution time.')
    # plt.show()
    #
    # plt.scatter(
    #     components,
    #     vars_explained
    # )
    # plt.xlabel('Num. Components')
    # plt.ylabel('% of variation explained')
    # plt.show()
    # plt.scatter(
    #     components,
    #     accs
    # )
    # plt.xlabel('Num. Components')
    # plt.ylabel('Classification accuracy')
    # plt.show()

if __name__ == '__main__':
    np.random.seed(123)
    find_outliers('ped')