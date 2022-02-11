# @Author: Enea Duka
# @Date: 7/9/21

import torch
import torch.nn as nn
import torch.nn.functional as F




class CategoricalMemory(nn.Module):
    def __init__(self):
        super(CategoricalMemory, self).__init__()

    def _get_scores(self, query, memory, query_idx=None):
        # calculates the normalized similarities between the memory and the query
        if query_idx is not None:
            q = query[query_idx]
        else:
            q = query
        score = torch.matmul(q, torch.t(memory))

        score_over_queries = F.softmax(score, dim=0)
        score_over_memory_cells = F.softmax(score, dim=1)

        return score_over_queries, score_over_memory_cells



    def _update_memory(self, query, labels, memory):
        # the query is of the shape bs x feat_dim
        # the memory is of the shape num_classes x mem_cells x mem_size

        # first split the query so that each split only contains samples from the same class
        q0_idx = labels == 0
        q1_idx = labels == 1
        q2_idx = labels == 2

        # calculate the normalized similarities between each split of the query and its respective memory entry
        q0_scores = self._get_scores(query, memory[0], q0_idx)
        q1_scores = self._get_scores(query, memory[1], q1_idx)
        q2_scores = self._get_scores(query, memory[2], q2_idx)

        # update each of the memory compartments
        mem_comp, mem_cells, mem_size = memory.size()
        new_mem = torch.zeros((mem_comp, mem_cells, mem_size)).cuda()
        new_mem[0] = F.normalize(memory[0] + torch.t(torch.matmul(torch.t(query[q0_idx]), q0_scores[0])))
        new_mem[1] = F.normalize(memory[1] + torch.t(torch.matmul(torch.t(query[q1_idx]), q1_scores[0])))
        new_mem[2] = F.normalize(memory[2] + torch.t(torch.matmul(torch.t(query[q2_idx]), q2_scores[0])))

        return new_mem.detach_()

    def _read_memory(self, query, memory, labels):
        # first flatten the memory so that we can compute similarity between all of it and each query
        n_class, n_cells, mem_size = memory.size()
        memory = memory.view(n_class*n_cells, mem_size)

        scores = self._get_scores(query, memory)[1]
        # max_idxs = torch.argmax(scores, dim=1) // 16
        # same_elements = (max_idxs == labels).sum()
        # print(same_elements)
        read_mem = torch.zeros(query.size()).to(query.device)

        for idx, q_row in enumerate(scores):
            read_mem[idx] = (q_row.unsqueeze(1) * memory).sum(0)
        # print(read_mem)
        # print((scores.unsqueeze(2) * memory).sum(0))
        # query = torch.cat((query, read_mem), dim=1)

        return read_mem

    def forward(self, query, labels, memory):
        # first we update the memory (we update only during training)

        q = self._read_memory(query, memory, labels)
        if self.training:
            memory = self._update_memory(query, labels, memory)
        # print(q.size())
        return q, memory



if __name__ == '__main__':
    memory = CategoricalMemory()
    memory.train()
    queries = torch.tensor([[1, 1],
                           [2, 2],
                           [3, 3],
                            [1, 1],
                            [2, 2],
                            [3, 3]
                            ], dtype=torch.float32)

    labels = torch.tensor([0, 2, 2, 2, 1, 0])

    memories = torch.rand((3, 3, 2))

    query, mems = memory(queries, labels, memories)

    # a = torch.tensor([[1, 2], [3, 4]])
    # b = torch.tensor([[1, 2], [1, 2]])
    #
    # c = torch.einsum('')