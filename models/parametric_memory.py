
# @Author: Enea Duka
# @Date: 8/2/21


import torch
import torch.nn as nn

class IncrementalParametricMemory(nn.Module):

    def __init__(self, memory_size, feature_dim):
        super(IncrementalParametricMemory, self).__init__()

        self.memory_size = memory_size
        self.feature_dim = feature_dim

        # define the keys and the values for the memory
        self.keys = nn.Parameter(torch.randn((self.memory_size, self.feature_dim)), requires_grad=True)
        self.values = nn.Parameter(torch.randn((self.memory_size, self.feature_dim)), requires_grad=True)

        self.extended_keys = None # torch.randn((256, self.feature_dim))
        self.extended_values = None # torch.randn((256, self.feature_dim))

    def forward(self, queries):
        # do a matrix multiplication between the queries and the mem keys
        # this way we get the dot prod for each query with each key
        # each row i of the result is the result of dot product of query i and all the keys
        # self.extended_keys = self.extended_keys.to(self.keys.device)
        if self.extended_keys is not None:
            keys = torch.cat([self.keys, self.extended_keys], dim=0)
        else:
            keys = self.keys
        qk = torch.matmul(queries, torch.t(keys)) # b_size x feat_size X feat_dim x mem_size = b_size x mem_size
        attn = torch.softmax(qk, dim=1)

        # use the attn values to get the values from the memory
        # self.extended_values = self.extended_values.to(self.keys.device)
        if self.extended_keys is not None:
            values = torch.cat([self.values, self.extended_values], dim=0)
        else:
            values = self.keys
        read_vals = torch.matmul(attn, values)  # b_size x mem_size X mem_size x feat_size = b_size x feat_size

        return read_vals

    def expand_memory(self, expand_size):
        keys = self.keys.data
        exp_keys = torch.randn((expand_size, self.feature_dim)).cuda()
        self.keys = nn.Parameter(keys, requires_grad=False)
        self.extended_keys = nn.Parameter(exp_keys, requires_grad=True)

        values = self.values.data
        exp_vals = torch.randn((expand_size, self.feature_dim)).cuda()
        self.values = nn.Parameter(values, requires_grad=False)
        self.extended_values = nn.Parameter(exp_vals, requires_grad=True)