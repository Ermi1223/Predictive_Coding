import torch
import torch.nn as nn
import torch.nn.functional as F

class PredictiveCodingLayer(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.W = nn.Linear(input_size, hidden_size)
        self.U = nn.Linear(hidden_size, hidden_size)
        self.V = nn.Linear(hidden_size, input_size)
        self.state = torch.zeros(1, hidden_size)

    def forward(self, x):
        prediction = self.V(self.state)
        error = x - prediction
        self.state = self.state + F.relu(self.W(error) + self.U(self.state)) * 0.1
        return prediction, error, self.state
