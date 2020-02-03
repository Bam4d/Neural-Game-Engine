import logging

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class HardSigmoid(nn.Module):

    def __init__(self, slope=0.5):
        super().__init__()
        self._slope = slope

    def forward(self, input):
        x = (input * self._slope) + 0.5
        x = torch.threshold(-x, -1, -1)
        x = torch.threshold(-x, 0, 0)
        return x


class SelectiveGates2D(nn.Module):

    def __init__(self, in_channels):
        super().__init__()

        self._in_channels = in_channels

        self._conv_g = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=True,
                      padding_mode='zeros'),
            nn.ReLU(),
            nn.Conv2d(in_channels, 5 * in_channels, kernel_size=1, stride=1, bias=True,
                      padding_mode='zeros'),
        )

        up = np.array([
            [0, 1, 0],
            [0, 0, 0],
            [0, 0, 0]
        ])

        right = np.array([
            [0, 0, 0],
            [0, 0, 1],
            [0, 0, 0]
        ])

        down = np.array([
            [0, 0, 0],
            [0, 0, 0],
            [0, 1, 0]
        ])

        left = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 0, 0]
        ])

        up = np.tile(up, (in_channels, 1)).reshape(self._in_channels, 1, 3, 3)
        right = np.tile(right, (in_channels, 1)).reshape(self._in_channels, 1, 3, 3)
        down = np.tile(down, (in_channels, 1)).reshape(self._in_channels, 1, 3, 3)
        left = np.tile(left, (in_channels, 1)).reshape(self._in_channels, 1, 3, 3)

        self._up = nn.Parameter(torch.FloatTensor(up), requires_grad=False)
        self._right = nn.Parameter(torch.FloatTensor(right), requires_grad=False)
        self._down = nn.Parameter(torch.FloatTensor(down), requires_grad=False)
        self._left = nn.Parameter(torch.FloatTensor(left), requires_grad=False)

    def forward(self, state):
        up_conv = torch.conv2d(state, self._up, bias=None, padding=1, groups=self._in_channels)
        down_conv = torch.conv2d(state, self._right, bias=None, padding=1, groups=self._in_channels)
        left_conv = torch.conv2d(state, self._down, bias=None, padding=1, groups=self._in_channels)
        right_conv = torch.conv2d(state, self._left, bias=None, padding=1, groups=self._in_channels)

        udlrc = torch.stack([
            up_conv,
            down_conv,
            left_conv,
            right_conv,
            state
        ])

        c = state.shape[1]
        h = state.shape[2]
        w = state.shape[3]

        select = self._conv_g(state)

        select = select.reshape(-1, 5, c, h, w)

        select = torch.softmax(select, dim=1).permute(1, 0, 2, 3, 4)

        # Reshape this to the right format
        return torch.mul(select, udlrc).sum(0)


class MaskedKernelConv(nn.Conv2d):
    """
    Use kernels that have the mask:

    0 1 0
    1 1 1
    0 1 0
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1,
                 bias=True, padding_mode='zeros'):
        super().__init__(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                         padding=padding, dilation=dilation,
                         bias=bias, padding_mode=padding_mode)

        kernel_mask = [
                          [
                              [0, 1, 0],
                              [1, 1, 1],
                              [0, 1, 0]
                          ]
                      ] * in_channels

        kernel_mask = np.expand_dims(kernel_mask, 1)

        self._mask = nn.Parameter(torch.FloatTensor(kernel_mask), requires_grad=False)

    def forward(self, input):
        masked_weight = self.weight * self._mask

        return F.conv2d(input, masked_weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class CGRUCell(nn.Module):

    def __init__(self, channels, saturation_limit=0.9, gate_mechanism=None, kernel_masking=False):
        super().__init__()
        self._channels = channels
        self._saturation_limit = saturation_limit
        self._gate_mechanism = gate_mechanism

        if kernel_masking:
            conv_kernel = MaskedKernelConv
        else:
            conv_kernel = nn.Conv2d

        self._conv_u = conv_kernel(channels, channels, kernel_size=3, padding=1, stride=1, bias=True)
        self._conv_r = conv_kernel(channels, channels, kernel_size=3, padding=1, stride=1, bias=True)
        self._conv_c = conv_kernel(channels, channels, kernel_size=3, padding=1, stride=1, bias=True)

        self._hard_sigmoid = HardSigmoid()

    def saturation_cost(self, x):
        return torch.relu(torch.abs(x) - self._saturation_limit)

    def forward(self, state):
        u = self._conv_u(state)
        r = self._conv_r(state)
        hsig_r = self._hard_sigmoid(r)

        c = self._conv_c(torch.mul(hsig_r, state))

        hsig_u = self._hard_sigmoid(u)
        htan_c = F.hardtanh(c)

        htan_c = F.dropout(htan_c, 0.1, training=self.training)

        if self._gate_mechanism is not None:
            state = self._gate_mechanism(state)

        out = torch.mul(hsig_u, state) + torch.mul((1.0 - hsig_u), htan_c)

        # we want to punish saturation
        if self._saturation_limit < 1.0:
            cost_u = self.saturation_cost(u)
            cost_r = self.saturation_cost(r)
            cost_c = self.saturation_cost(c)
            saturation_cost = torch.mean(cost_u) + torch.mean(cost_r) + torch.mean(cost_c)

            return out, saturation_cost
        else:
            zero = torch.scalar_tensor(0).to(self._device)
            return out, zero
