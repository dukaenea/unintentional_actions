# @Author: Enea Duka
# @Date: 6/24/21

import torch
import torch.nn as nn
from utils.arg_parse import opt


class PrototypicalMemory(nn.Module):
    def __init__(self, memory_size):
        super(PrototypicalMemory, self).__init__()

        self.memory_size = memory_size
        self.memory = torch.zeros((3, memory_size)).cuda()
        self.nl_mlp = nn.Sequential(
            nn.LayerNorm(memory_size),
            nn.ReLU(),
            nn.Linear(memory_size, memory_size),
            nn.Dropout(0.2)
        )

        self.rareacts_weights = {}
        self.oops_weights = {}
        self.kinetics_weights = {}
        self.weights_count = []

        self.temperature = 0.07

    def forward(self, features):
        fm = torch.mm(features, torch.transpose(self.memory, 0, 1).to(features.device))
        fm /= self.memory.shape[1]
        att = torch.softmax(fm, dim=1)

        mem_values = torch.mm(att, self.memory.to(features.device))
        mem_values = self.nl_mlp(mem_values)

        return features + mem_values

    def reset_memory(self):
        self.memory = torch.zeros((3, self.memory_size))
        self.rareacts_weights = {}
        self.oops_weights = {}
        self.kinetics_weights = {}
        self.weights_count = []

    def update_memory_weights(self, weight, dataset, filename):
        weight = 1 / weight
        if dataset == 0:
            self.rareacts_weights[filename] = weight
        elif dataset == 1:
            self.oops_weights[filename] = weight
        elif dataset == 2:
            self.kinetics_weights[filename] = weight

    def sharpen_normalize_weights(self):
        new_rareacts_weights = torch.softmax(
            torch.pow(
                torch.tensor(list(self.rareacts_weights.values())), 1 / self.temperature
            ), dim=0
        ).tolist()
        for idx, key in enumerate(list(self.rareacts_weights.keys())):
            self.rareacts_weights[key] = new_rareacts_weights[idx]
        self.weights_count.append(len(self.rareacts_weights))

        new_oops_weights = torch.softmax(
            torch.pow(
                torch.tensor(list(self.oops_weights.values())), 1 / self.temperature
            ), dim=0
        ).tolist()
        for idx, key in enumerate(list(self.oops_weights.keys())):
            self.oops_weights[key] = new_oops_weights[idx]
        self.weights_count.append(len(self.oops_weights))


        new_kinetics_weights = torch.softmax(
            torch.pow(
                torch.tensor(list(self.kinetics_weights.values())), 1 / self.temperature
            ), dim=0
        ).tolist()
        for idx, key in enumerate(list(self.kinetics_weights.keys())):
            self.kinetics_weights[key] = new_kinetics_weights[idx]
        self.weights_count.append(len(self.kinetics_weights))

    def update_memory(self, features, dataset, filename, normalize=False):
        try:
            if dataset == 0:
                weight = self.rareacts_weights[filename]
                self.memory[0] += features.cuda() * weight

            elif dataset == 1:
                weight = self.oops_weights[filename]
                self.memory[1] += features.cuda() * weight

            elif dataset == 2:
                weight = self.kinetics_weights[filename]
                self.memory[2] += features.cuda() * weight
        except Exception as e:
            pass

        if normalize:
            self.memory[0] /= self.weights_count[0]
            self.memory[1] /= self.weights_count[1]
            self.memory[2] /= self.weights_count[2]