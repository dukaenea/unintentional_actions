
# @Author: Enea Duka
# @Date: 8/12/21

import torch
import torch.nn as nn
from scipy.stats import multivariate_normal
import numpy as np
from tqdm import tqdm
from scipy.spatial import distance

class GaussianClassifier(nn.Module):
    def __init__(self, training_data_points, distance_type='mahalanobis'):
        super().__init__()
        # to calculate the mahalanobis distance later we need
        # the mean vector and the covariance matrix of the
        # training data
        # each row in the training_data_points is a data point
        self.distance_type=distance_type
        training_data_points = training_data_points.cpu().numpy()
        self.mean = torch.from_numpy(np.average(training_data_points, axis=0)).type(torch.float32).cuda()
        cov = np.cov(training_data_points, rowvar=False)
        # self.prob = multivariate_normal(self.mean.cpu(), cov)
        print(cov)
        self.det = np.linalg.det(cov)
        if not self._is_pos_def(cov):
            print('Error in cov matrix')

        self.cov = torch.from_numpy(np.linalg.pinv(cov)).type(torch.float32).cuda()

        # set the classification threshold as three times the greatest variance
        m_distances = []
        for training_point in tqdm(training_data_points):
            if distance_type == 'mahalanobis':
                smm = (torch.tensor(training_point).cuda()-self.mean)
                if len(smm.shape) == 1:
                    smm = smm.unsqueeze(0)
                left_term = torch.mm(smm, self.cov)

                m_distances.append(torch.sqrt(torch.dot(left_term.squeeze(), smm.squeeze())).item())
            elif distance_type == 'euclidean':
                # calculate the distance of the point from the mean
                dist = torch.sqrt(((torch.tensor(training_point).cuda()-self.mean)**2).sum())
                m_distances.append(dist.item())
        m_distances = np.array(m_distances)
        min_distance = np.min(m_distances)
        max_distance = np.max(m_distances)
        self.mean_m_distance = np.mean(m_distances)
        self.std_m_distance = np.std(m_distances)
        k = .1 * self.std_m_distance

        self.up_t = self.mean_m_distance + k
        self.low_t = self.mean_m_distance - k

        self.max = max_distance
        self.min = min_distance

        print('Determinant of the cov matrix: %f' % self.det)
        print('Threshold segment: %f -> %f' % (self.low_t, self.up_t))
        print('Minimal distance: %f' % min_distance)
        print('Maximal distance: %f' % max_distance)

    def update_threshold(self, t):
        k = t * self.std_m_distance
        self.up_t = self.mean_m_distance + k
        self.low_t = self.mean_m_distance - k
        print('Threshold segment: %f -> %f' % (self.low_t, self.up_t))
        # self.up_t = t

    def forward(self, x):
        # calculate the mahalanobis distance here between the
        # inference data point and the training datapoints
        if self.distance_type == 'mahalanobis':
            # smm = (x-self.mean)
            # if len(smm.shape) == 1:
            #     smm = smm.unsqueeze(0)
            # left_term = torch.mm(smm, self.cov)
            # m_distance = torch.mm(left_term, torch.t(smm))
            # m_distance = torch.sqrt(m_distance)
            m_distance = distance.mahalanobis(x.cpu(), self.mean.cpu(), self.cov.cpu())
            # m_distance = self.prob.pdf(x.cpu().numpy())
        elif self.distance_type == 'euclidean':
            m_distance = torch.sqrt(((x-self.mean)**2).sum())
        if self.low_t <= m_distance <= self.up_t:
            return 0
        else:
            return 1

    def _is_pos_def(self, A):
        if np.allclose(A, A.T):
            try:
                np.linalg.cholesky(A)
                return True
            except np.linalg.LinAlgError:
                return False
        return False



if __name__ == '__main__':
    mean = torch.rand((768, ))
    std  = torch.rand((768, ))

    mean = torch.tensor(mean)
    std  = torch.tensor(std)

    train_samples = []
    for i in range(1000):
        train_samples.append(torch.normal(mean, std))

    train_samples = torch.stack(train_samples)

    gc = GaussianClassifier(0.2, train_samples)
    id_inference_point = torch.normal(mean, std)

    distance = gc(id_inference_point)
    print('Distance of in distribution point: %f' % distance)

    # sampled from a uniform dist U(5, 10)
    od_inference_point = (torch.rand((1, 768)) + 5) * 2
    distance = gc(od_inference_point)
    print('Distance of out of distribution point %f' % distance)

    s1_inderence_point = mean + std
    distance = gc(s1_inderence_point)
    print('Distance of point 1 std away %f' % distance)

    s1_inderence_point = mean + (2 * std)
    distance = gc(s1_inderence_point)
    print('Distance of point 2 std away %f' % distance)

    s1_inderence_point = mean + (3 * std)
    distance = gc(s1_inderence_point)
    print('Distance of point 3 std away %f' % distance)
