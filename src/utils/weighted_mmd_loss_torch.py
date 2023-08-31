import torch
import torch.nn as nn
from sklearn.metrics.pairwise import rbf_kernel


seed = 5
torch.manual_seed(seed)


# Weighted MMD loss
class WeightedMMDLoss(nn.Module):
    """Weighted MMD loss class"""

    def __init__(self, gamma, N, R, device="cpu"):
        """Initializes the loss function

        :param gamma: Gamma for the RBF-Kernel
        :param N: Non-representative data set
        :param R: Representative data set
        :param device: Device to whom the data is copied
        """
        super(WeightedMMDLoss, self).__init__()
        self.weights_R = (torch.ones(len(R), dtype=torch.float64) / len(R)).to(device)
        self.weights_R = self.weights_R / torch.sum(self.weights_R)
        self.n_n_rbf_matrix = torch.FloatTensor(rbf_kernel(N, N, gamma)).to(device)
        self.n_r_rbf_matrix = torch.FloatTensor(rbf_kernel(N, R, gamma)).to(device)
        self.gamma = gamma

        r_r_rbf_matrix = torch.matmul(
            torch.unsqueeze(self.weights_R, 1), torch.unsqueeze(self.weights_R, 0)
        ) * torch.FloatTensor(rbf_kernel(R, R, gamma)).to(device)
        self.r_r_mean = r_r_rbf_matrix.sum()

    def forward(self, weights):
        """Computes the loss

        :param weights: Current weights
        :return: Loss value
        """
        if not torch.is_tensor(weights):
            weights = torch.DoubleTensor(weights)
        n_n_mean = (
            torch.matmul(torch.unsqueeze(weights, 1), torch.unsqueeze(weights, 0))
            * self.n_n_rbf_matrix
        ).sum()
        n_r_mean = (
            torch.matmul(
                torch.unsqueeze(weights, 1), torch.unsqueeze(self.weights_R, 0)
            )
            * self.n_r_rbf_matrix
        ).sum()

        return torch.sqrt(n_n_mean + self.r_r_mean - 2 * n_r_mean)
