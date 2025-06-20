import torch
import torch.nn as nn
import torch.nn.functional as F

class PredictiveCodingLayer(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.W = nn.Linear(input_size, hidden_size)    # feedforward weights for error → hidden update
        self.U = nn.Linear(hidden_size, hidden_size)   # lateral hidden → hidden connections
        self.V = nn.Linear(hidden_size, input_size)    # feedback prediction from hidden → input space

    def forward(self, error, h):
        """
        Args:
            error: Tensor of shape (batch_size, input_size) — prediction error from lower layer or input
            h: Tensor of shape (batch_size, hidden_size) — current hidden state

        Returns:
            updated hidden state tensor of shape (batch_size, hidden_size)
        """
        # Prediction from hidden state to input
        prediction = self.V(h)

        # Update hidden state with error signal and lateral connections (ReLU or tanh activation)
        h_new = h + F.relu(self.W(error)) + F.relu(self.U(h))

        return h_new, prediction
