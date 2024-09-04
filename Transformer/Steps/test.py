import pkuseg
import numpy as np
import torch
import torch.nn as nn


# Define a simple neural network model
class Test(nn.Module):
    def forward(self, i, j, k, x):
        print(i, j, k, x)
        i += 1
        j += 1
        k += 1
        x += 1
        print(i, j, k, x)
        return j, k


class SequentialModule(nn.Sequential):
    def forward(self, *inputs):
        i, j, k, x = inputs
        for module in self._modules.values():
            j, k = module(i, j, k, x)
        return j, k


class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = SequentialModule(*[Test() for _ in range(5)])

    def forward(self, i, j, k, x):
        j, k = self.layers(i, j, k, x)
        # only j is changed
        return j, k


snn = SimpleNN()
print(snn(1, 1, 1, 1))

seg = pkuseg.pkuseg()
text = "我喜欢自然语言处理。"
tokens = seg.cut(text)
print(tokens)
