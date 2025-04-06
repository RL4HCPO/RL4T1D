import torch
import torch.nn as nn
import torch.nn.functional as F

class CostEstimatorMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128, 64]):
        super(CostEstimatorMLP, self).__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))  # Output: predicted cost
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
