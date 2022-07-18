#!/usr/bin/env python3
import megengine.module as M


class SELayer(M.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = M.AdaptiveAvgPool2d(1)
        self.fc = M.Sequential(
            M.Linear(channel, channel // reduction, bias=False),
            M.ReLU(),
            M.Linear(channel // reduction, channel, bias=False),
            M.Sigmoid()
        )

    def forward(self, x):
        # print(x.shape)
        b, c, _, _ = x.shape
        y = self.avg_pool(x).reshape(b, c)
        y = self.fc(y).reshape(b, c, 1, 1)
        y = y.reshape(x.shape)
        return x * y, x * (1-y)

# vim: ts=4 sw=4 sts=4 expandtab
