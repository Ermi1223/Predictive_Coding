import torch
import torch.nn as nn
from .predictive_layer import PredictiveCodingLayer

class PredictiveCodingNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = PredictiveCodingLayer(28*28, 128)
        self.layer2 = PredictiveCodingLayer(128, 64)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        p1, e1, h1 = self.layer1(x)
        p2, e2, h2 = self.layer2(h1)
        return p1, e1, p2, h2

    def reset_states(self):
        device = next(self.parameters()).device
        self.layer1.state = torch.zeros(1, 128).to(device)
        self.layer2.state = torch.zeros(1, 64).to(device)
