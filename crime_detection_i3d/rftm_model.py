
# @Author: Enea Duka
# @Date: 10/20/21

import torch
import torch.nn as nn
import torch.nn.init as torch_init
torch.set_default_tensor_type('torch.cuda.FloatTensor')
from models.pm_vtn import create_model
from transformer2x.crf import CRF
from itertools import chain


def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        torch_init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)

class _NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True, bn_layer=True):
        super(_NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x, return_nl_map=False):
        """
        :param x: (b, c, t, h, w)
        :param return_nl_map: if True return z, nl_map, else only return z.
        :return:
        """

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)

        f = torch.matmul(theta_x, phi_x)
        N = f.size(-1)
        f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        if return_nl_map:
            return z, f_div_C
        return z


class NONLocalBlock1D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock1D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=1, sub_sample=sub_sample,
                                              bn_layer=bn_layer)

class Aggregate(nn.Module):
    def __init__(self, len_feature):
        super(Aggregate, self).__init__()
        bn = nn.BatchNorm1d
        self.len_feature = len_feature
        self.conv_1 = nn.Sequential(
            nn.Conv1d(in_channels=len_feature, out_channels=512, kernel_size=3,
                      stride=1,dilation=1, padding=1),
            nn.ReLU(),
            bn(512)
            # nn.dropout(0.7)
        )
        self.conv_2 = nn.Sequential(
            nn.Conv1d(in_channels=len_feature, out_channels=512, kernel_size=3,
                      stride=1, dilation=2, padding=2),
            nn.ReLU(),
            bn(512)
            # nn.dropout(0.7)
        )
        self.conv_3 = nn.Sequential(
            nn.Conv1d(in_channels=len_feature, out_channels=512, kernel_size=3,
                      stride=1, dilation=4, padding=4),
            nn.ReLU(),
            bn(512)
            # nn.dropout(0.7),
        )
        self.conv_4 = nn.Sequential(
            nn.Conv1d(in_channels=2048, out_channels=512, kernel_size=1,
                      stride=1, padding=0, bias = False),
            nn.ReLU(),
            # nn.dropout(0.7),
        )
        self.conv_5 = nn.Sequential(
            nn.Conv1d(in_channels=2048, out_channels=2048, kernel_size=3,
                      stride=1, padding=1, bias=False), # should we keep the bias?
            nn.ReLU(),
            nn.BatchNorm1d(2048),
            # nn.dropout(0.7)
        )

        self.non_local = NONLocalBlock1D(512, sub_sample=False, bn_layer=True)


    def forward(self, x):
            # x: (B, T, F)
            out = x.permute(0, 2, 1)
            residual = out

            out1 = self.conv_1(out)
            out2 = self.conv_2(out)

            out3 = self.conv_3(out)
            out_d = torch.cat((out1, out2, out3), dim = 1)
            out = self.conv_4(out)
            out = self.non_local(out)
            out = torch.cat((out_d, out), dim=1)
            out = self.conv_5(out)   # fuse all the features together
            out = out + residual
            out = out.permute(0, 2, 1)
            # out: (B, T, 1)

            return out

class Model(nn.Module):
    def __init__(self, n_features, batch_size):
        super(Model, self).__init__()
        self.batch_size = batch_size
        self.num_segments = 32
        self.k_abn = self.num_segments // 10
        self.k_nor = self.num_segments // 10

        self.Aggregate = Aggregate(len_feature=2048)
        self.crf = CRF(num_tags=2, batch_first=True)
        self.Transformer, _, _ = create_model(None, False)
        self.fc1 = nn.Linear(n_features, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)
        self.ln = nn.LayerNorm(2048)

        self.fc1_v = nn.Linear(n_features, 512)
        self.fc3_v = nn.Linear(512, 1)
        self.ln_v = nn.LayerNorm(2048)

        self.drop_out = nn.Dropout(0.7)
        self.relu = nn.GELU()
        self.sigmoid = nn.Sigmoid()
        self.apply(weight_init)

    def forward_crf(self, scores, labels):
        return -self.crf.forward(scores, labels)

    def val_crf(self, scores):
        mls = self.crf.decode(scores)
        return torch.tensor(list(chain.from_iterable(mls))).to(scores.decice)

    def _get_max_idxs(self, inputs):
        # the inputs will be of shape B x CR x SEGS x FEATS
        B, CR, SEGS, FEATS = inputs.shape
        inputs = inputs.mean(1)  # B x SEGS X FEATS
        video_centers = inputs.mean(1).unsqueeze(1)  # B x FEATS
        # calculate the distances of all the clips from the respective video center
        video_centers = torch.repeat_interleave(video_centers, 32, dim=1)

        inputs = inputs.reshape(B * SEGS, FEATS)
        video_centers = video_centers.reshape(B * SEGS, FEATS)

        # distances = torch.cdist(inputs, video_centers, p=2)
        distances_fn = torch.nn.PairwiseDistance(p=2)
        distances = distances_fn(inputs, video_centers)
        distances = torch.stack(list(torch.split(distances, SEGS)), dim=0)

        max_idxs = distances.max(dim=-1)[1]

        return max_idxs

    def forward(self, inputs, use_crf=False):

        k_abn = self.k_abn
        k_nor = self.k_nor

        out = inputs
        bs, ncrops, t, f = out.size()

        out = out.view(-1, t, f)

        # out = self.Aggregate(out)

        position_ids = torch.tensor(list(range(0, out.shape[1]))) \
            .expand(1, out.shape[1]) \
            .repeat(out.shape[0], 1)
        pure_nr_frames = torch.tensor([out.shape[1]] * out.shape[0]).to(out.device)

        out = self.Transformer(out, position_ids, None, pure_nr_frames, high_order_temporal_features=True, return_features=True)

        # out = self.drop_out(out)

        features = out
        scores = self.relu(self.fc1(self.ln(features)))
        scores = self.drop_out(scores)
        scores = self.relu(self.fc2(scores))
        scores = self.drop_out(scores)
        scores = self.sigmoid(self.fc3(scores))
        scores = scores.view(bs, ncrops, -1).mean(1)
        # one_scores = scores.unsqueeze(-1)
        # zero_scores = 1 - scores.unsqueeze(-1)
        #
        # crf_scores = torch.cat((zero_scores, one_scores), dim=-1)
        # if self.training:
        #     n_labels = torch.zeros(8, 32)
        #     an_scores = scores[8:]
        #     an_scores_max = torch.topk(an_scores, 2, dim=-1)[1]# an_scores.max(dim=-1)[1]
        #     # an_scores_max = self._get_max_idxs(inputs)[8:]
        #     an_labels = torch.zeros(8, 32)
        #     an_labels[torch.arange(0, 8), an_scores_max[:, 0]] = 1
        #     an_labels[torch.arange(0, 8), an_scores_max[:, 1]] = 1
        #     crf_labels = torch.cat((n_labels, an_labels), dim=0).to(crf_scores.device).type(torch.long)
        #
        #     crf_loss = -self.crf.forward(crf_scores, crf_labels)
        #     return scores, crf_loss
        # else:
        #     crf_mask = torch.ones_like(scores)
        #     crf_probs = self.crf.compute_marginal_probabilities(crf_scores.permute(0, 1, 2), crf_mask.permute(0, 1))
        return scores

