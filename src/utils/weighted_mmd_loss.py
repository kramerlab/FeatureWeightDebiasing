from sklearn.metrics.pairwise import rbf_kernel
import numpy as np


# Weighted MMD loss
class WeightedMMDLoss():
    """Weighted MMD loss class"""

    def __init__(self, gamma, N, R):
        """Initializes the loss function

        :param gamma: Gamma for the RBF-Kernel
        :param N: Non-representative data set
        :param R: Representative data set
        :param device: Device to whom the data is copied
        """
        super(WeightedMMDLoss, self).__init__()
        self.weights_R = np.ones(len(R), dtype=np.float64) / len(R)
        self.weights_R = self.weights_R / np.sum(self.weights_R)
        self.n_n_rbf_matrix = rbf_kernel(N, N, gamma)
        self.n_r_rbf_matrix = rbf_kernel(N, R, gamma)
        self.gamma = gamma

        r_r_rbf_matrix = np.matmul(
            np.expand_dims(self.weights_R, 1), np.expand_dims(self.weights_R, 0)
        ) * rbf_kernel(R, R, gamma)
        self.r_r_mean = r_r_rbf_matrix.sum()

    def __call__(self, weights):
        """Computes the loss

        :param weights: Current weights
        :return: Loss value
        """
        n_n_mean = (
            np.matmul(np.expand_dims(weights, 1), np.expand_dims(weights, 0))
            * self.n_n_rbf_matrix
        ).sum()
        n_r_mean = (
            np.matmul(np.expand_dims(weights, 1), np.expand_dims(self.weights_R, 0))
            * self.n_r_rbf_matrix
        ).sum()

        return np.sqrt(n_n_mean + self.r_r_mean - 2 * n_r_mean)
