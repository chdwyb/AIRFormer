import torch
import math
import torch.nn as nn
import numpy as np


def build_filter(pos, freq, POS):
    result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS)
    if freq == 0:
        return result
    else:
        return result * math.sqrt(2)


def generate_dct_matrix(h=8, w=8):
    matrix = np.zeros((h, w, h, w))

    us = list(range(h))
    vs = list(range(w))

    for u in range(h):
        for v in range(w):
            for i in range(h):
                for j in range(w):
                    matrix[u, v, i, j] = build_filter(i, u, h) * build_filter(j, v, w)

    matrix = matrix.reshape(-1, h, w)

    return matrix


class dct_layer(nn.Module):
    def __init__(self, in_c=3, h=8, w=8):
        super(dct_layer, self).__init__()
        assert h == w

        self.dct_conv = nn.Conv2d(in_c, in_c * h * w, h, h, bias=False, groups=in_c)
        matrix = generate_dct_matrix(h=h, w=w)
        self.weight = torch.from_numpy(matrix).float().unsqueeze(1)  # 64,1,8,8

        self.dct_conv.weight.data = torch.cat([self.weight] * in_c, dim=0)  # 192,1,8,8
        self.dct_conv.weight.requires_grad = False

    def forward(self, x):
        dct = self.dct_conv(x)

        return dct




