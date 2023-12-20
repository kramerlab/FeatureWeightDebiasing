from torch import nn
from utils.gradient_reversal import GradientReversal


class DANN(nn.Module):
    """The neural network for sample weights"""

    def __init__(self, number_of_features, latent_features):
        """Initializes the neural network

        :param number_of_features: Number of input features
        :param latent_features: Number of latent features
        """
        super(DANN, self).__init__()
        self.encode = nn.Sequential(
            nn.Linear(number_of_features, latent_features),
            nn.ReLU(),
        )
        self.classify_target = nn.Sequential(
            nn.Linear(latent_features, 1),
        )
        self.classify_domain = nn.Sequential(
            GradientReversal(alpha=1.0),
            nn.Linear(latent_features, 1),
        )

    def forward(self, x):
        """Forward pass

        :param x: Training data
        :return: Sample weights
        """
        encoding = self.encode(x)
        class_log_probabilities = self.classify_target(encoding)
        domain_log_probabilities = self.classify_domain(encoding)
        return class_log_probabilities, domain_log_probabilities, encoding


class CWNN(nn.Module):
    """The neural network for sample weights"""

    def __init__(self, number_of_features, latent_features):
        """Initializes the neural network

        :param number_of_features: Number of input features
        :param latent_features: Number of latent features
        """
        super(CWNN, self).__init__()
        self.correction_weights_layer = nn.Sequential(
            nn.Linear(number_of_features, number_of_features),
            nn.ReLU(),
            nn.Linear(number_of_features, number_of_features),
            # nn.ReLU(),
        )

        self.classify_target = nn.Sequential(
            nn.Linear(number_of_features, latent_features),
            nn.ReLU(),
            nn.Linear(latent_features, 1),
        )
        self.classify_domain = nn.Sequential(
            GradientReversal(alpha=1.0),
            nn.Linear(number_of_features, latent_features),
            nn.ReLU(),
            nn.Linear(latent_features, 1),
        )

    def forward(self, x):
        """Forward pass

        :param x: Training data
        :return: Sample weights
        """
        correction_weights = self.correction_weights_layer(x)
        weighted_x = x * correction_weights
        class_log_probabilities = self.classify_target(weighted_x)
        domain_log_probabilities = self.classify_domain(weighted_x)
        return class_log_probabilities, domain_log_probabilities, correction_weights
