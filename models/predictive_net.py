import torch
import torch.nn as nn
import torch.nn.functional as F

class PredictiveCodingLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.W = nn.Linear(input_dim, hidden_dim)  # feedforward from error to hidden update
        self.U = nn.Linear(hidden_dim, hidden_dim) # lateral hidden-to-hidden

    def forward(self, error, h):
        # Update hidden state with error + lateral connections
        return h + torch.tanh(self.W(error)) + torch.tanh(self.U(h))

class PredictiveCodingNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_dim = 28*28
        self.l1_dim = 128
        self.l2_dim = 64
        self.l3_dim = 32

        # Predictive coding layers
        self.layer1 = PredictiveCodingLayer(self.input_dim, self.l1_dim)
        self.layer2 = PredictiveCodingLayer(self.l1_dim, self.l2_dim)
        self.layer3 = PredictiveCodingLayer(self.l2_dim, self.l3_dim)

        # Feedback weights: top-down predictions
        self.feedback3 = nn.Linear(self.l3_dim, self.l2_dim)
        self.feedback2 = nn.Linear(self.l2_dim, self.l1_dim)
        self.feedback1 = nn.Linear(self.l1_dim, self.input_dim)

        # Classifier on top layer
        self.classifier = nn.Linear(self.l3_dim, 10)

    def forward(self, x, steps=5):
        batch_size = x.size(0)
        device = x.device

        x = x.view(batch_size, -1)  # flatten

        # Initialize hidden states as zeros
        h1 = torch.zeros(batch_size, self.l1_dim, device=device)
        h2 = torch.zeros(batch_size, self.l2_dim, device=device)
        h3 = torch.zeros(batch_size, self.l3_dim, device=device)

        for _ in range(steps):
            # Top-down predictions
            pred_h2 = self.feedback3(h3)
            pred_h1 = self.feedback2(h2)
            pred_x = self.feedback1(h1)

            # Prediction errors
            error_h2 = h2 - pred_h2
            error_h1 = h1 - pred_h1
            error_x = x - pred_x

            # Update hidden states using error and lateral connections
            h3 = self.layer3(error_h2, h3)
            h2 = self.layer2(error_h1, h2)
            h1 = self.layer1(error_x, h1)

        # Classification from top layer latent state
        logits = self.classifier(h3)
        return logits
