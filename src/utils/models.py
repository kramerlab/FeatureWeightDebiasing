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
        self.encoding = nn.Sequential(
            nn.Linear(number_of_features, latent_features),
            nn.ReLU(),
            nn.BatchNorm1d(latent_features),
            nn.Linear(latent_features, latent_features),
            nn.ReLU(),
            nn.BatchNorm1d(latent_features),
        )

        self.classify_target = nn.Linear(latent_features, 1)
        self.weight_sample = nn.Linear(latent_features, 1)
        self.softmax = torch.nn.Softmax(dim=0)

    def forward(self, x):
        """Forward pass

        :param x: Training data
        :return: Sample weights
        """
        encodings = self.encoding(x)
        weights = self.weight_sample(encodings).flatten()
        logits = self.classify_target(encodings)
    
        return self.softmax(weights), logits
