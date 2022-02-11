# @Author: Enea Duka
# @Date: 7/8/21


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Memory(nn.Module):
    def __init__(self, memory_size, feature_dim, key_dim):
        super(Memory, self).__init__()

        self.memory_size = memory_size
        self.feature_dim = feature_dim
        self.key_dim = key_dim

    def get_score(self, query, memory):
        # print(self.keys.device)
        # print(query.device)
        # self.keys = self.keys.to(query.device)

        # bs, t, d = query.size()
        # m, d = self.keys.size()
        m, d = memory.size()
        score = torch.matmul(query, torch.t(memory))
        # score = score.view(bs*t, m)

        score_query = F.softmax(score, dim=0)
        score_memory = F.softmax(score, dim=1)

        return score_query, score_memory

    def gather_loss(self, query, memory, sfx_score_memory):
        # bs, t, d = query.size()

        loss_mse = nn.MSELoss()
        # query_reshape = query.contiguous().view(bs*t, d)

        _, gathering_idxs = torch.topk(sfx_score_memory, 1, dim=1)
        gathering_loss = loss_mse(query, memory[gathering_idxs].squeeze(1).detach())

        return gathering_loss

    def spread_loss(self, query, memory, sfx_score_memory):
        # bs, t, d = query.size()

        loss = torch.nn.TripletMarginLoss(margin=1.0)

        _, gathering_idxs = torch.topk(sfx_score_memory, 2, dim=1)
        # query_reshape = query.contiguous().view(bs * t, d)

        # 1st and 2nd closest memories
        pos = memory[gathering_idxs[:, 0]]
        neg = memory[gathering_idxs[:, 1]]

        spreading_loss = loss(query, pos.detach(), neg.detach())

        return spreading_loss

    def forward(self, query, memory, train=True):
        # the memory will be of the size num_classes x mem_size x key_size
        # bs, t, d = query.size()
        query = F.normalize(query, dim=1)
        sfx_score_query, sfx_score_memory = self.get_score(query, memory)
        if train:
            # gathering loss
            # gathering_loss = self.gather_loss(query, memory, sfx_score_memory)
            # spreading_loss
            # spreading_loss = self.spread_loss(query, memory, sfx_score_memory)
            # read
            update_query = self.read(query, memory, sfx_score_memory)
            # update
            updated_memory = self.update(query, memory, sfx_score_query, sfx_score_query, train)

            return update_query, updated_memory, sfx_score_query, sfx_score_memory, gathering_loss, spreading_loss

        else:
            # gathering loss
            gathering_loss = self.gather_loss(query, memory, sfx_score_memory)

            # read
            update_query = self.read(query, memory, sfx_score_memory)

            return update_query, sfx_score_query, sfx_score_memory, gathering_loss

    def update(self, query, memory, sfx_score_memory, sfx_score_query, train):

        # bs, t, d = query.size()

        # query_reshape = query.contiguous().view(bs*t, d)

        _, gathering_idxs = torch.topk(sfx_score_memory, 1, dim=1)
        # _, updating_idxs = torch.topk(sfx_score_query, 1, dim=0)

        query_update = self.get_update_query(memory, gathering_idxs, sfx_score_query, query)
        updated_memory = F.normalize(query_update + memory, dim=1)

        return updated_memory.detach()

    def get_update_query(self, memory, max_idx, score, query):

        m, d = memory.size()

        query_update = torch.zeros((m, d)).cuda()
        for i in range(m):
            idx = torch.nonzero(max_idx.squeeze(1) == i)
            a, _ = idx.size()
            if a != 0:
                query_update[i] = torch.sum(((score[idx, i] / torch.max(score[:, i])) * query[idx].squeeze(1)), dim=0)
            else:
                query_update[i] = 0
        return query_update

    def read(self, query, updated_memory, sfx_score_memory):

        bs, d = query.size()

        # query_reshape = query.contiguous().view(bs*t, d)

        concat_memory = torch.matmul(sfx_score_memory.detach(), updated_memory)
        update_query = torch.cat((query, concat_memory), dim=1)
        update_query = update_query.view(bs, 2 * d)

        return update_query



if __name__ == '__main__':
    memory = Memory(16, 4, 4)
    memory.cuda()
    query = torch.rand((8, 3,  4)).cuda()
    mem = torch.rand((5, 4)).cuda()
    # optimizer = torch.optim.SGD(lr=0.1, params=memory.parameters())
    for i in range(10):
        # optimizer.zero_grad()
        update_query, sfx_score_query, sfx_score_memory, gathering_loss, spreading_loss = memory(query, mem)
        print('GT Loss: %f  |  SP Loss: %f' % (gathering_loss, spreading_loss))
        # loss = gathering_loss + spreading_loss
        # loss.backward()
        # optimizer.step()
