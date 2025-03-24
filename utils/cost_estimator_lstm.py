import torch
import torch.nn as nn
import torch.nn.functional as F

class CostEstimatorLSTM(nn.Module):
    def __init__(self, feature_history, n_features, extra_feat_dim, hidden_size=64):
        super(CostEstimatorLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size + extra_feat_dim, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, obs_seq, extra_features):
        """
        obs_seq: Tensor of shape (batch_size, feature_history, n_features)
        extra_features: Tensor of shape (batch_size, extra_feat_dim)
        """
        _, (h_n, _) = self.lstm(obs_seq)  # h_n: (1, batch, hidden_size)
        h_n = h_n.squeeze(0)  # shape: (batch, hidden_size)
        x = torch.cat([h_n, extra_features], dim=1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)  # Scalar cost