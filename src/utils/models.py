from torch import nn
import torch


class WeightingMlp(nn.Module):
    """The neural network for sample weights"""

    def __init__(self, number_of_features, latent_features):
        """Initializes the neural network

        :param number_of_features: Number of input features
        :param latent_features: Number of latent features
        """
        super(WeightingMlp, self).__init__()
        self.weighting = nn.Sequential(
            nn.Linear(number_of_features, latent_features, dtype=torch.float64),
            nn.ReLU(),
            nn.BatchNorm1d(latent_features, dtype=torch.float64),
            nn.Linear(latent_features, 1, dtype=torch.float64),
        )
        self.softmax = torch.nn.Softmax(dim=0)

    def forward(self, x):
        """Forward pass

        :param x: Training data
        :return: Sample weights
        """
        weights = self.weighting(x).flatten()
        return self.softmax(weights)
