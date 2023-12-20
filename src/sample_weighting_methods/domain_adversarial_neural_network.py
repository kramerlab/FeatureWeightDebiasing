import torch
import numpy as np
from pathlib import Path
from torch import nn
from sklearn.metrics import roc_auc_score, average_precision_score
from utils.domain_adaptation_models import DANN
from torch.utils.data import DataLoader, TensorDataset
from torch.linalg import svdvals

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

lambda_weight = 1


def train_domain_adversarial_network(
    N, R, columns, target, auroc_list, auprc_list, *args, **attributes
):
    """Trains a domain adversarial neural network

    :param N: Non-representative data set
    :param R: Representative data set
    :param columns: Training columns
    :return: Samples weights
    """
    N_copy = N.copy().reset_index()
    tensor_N = torch.FloatTensor(N_copy[columns].values)
    tensor_R = torch.FloatTensor(R[columns].values)
    tensor_targets_N = torch.FloatTensor(N_copy[target].values)
    tensor_targets_R = torch.FloatTensor(R[target].values)

    all_weights = np.zeros(len(N))
    model = train_domain_adversarial_neural_network(
        tensor_N, tensor_R, tensor_targets_N, tensor_targets_R
    )

    with torch.no_grad():
        model.eval()
        r_logits, _, _ = model(tensor_R.to(device))
        probabilities = nn.functional.sigmoid(r_logits.cpu()).numpy()
        auroc_r = roc_auc_score(tensor_targets_R.numpy(), probabilities)
        auroc_list.append(auroc_r)

        auprc_r = average_precision_score(tensor_targets_R.numpy(), probabilities)
        auprc_list.append(auprc_r)

    all_weights = np.ones(len(N))
    return all_weights / sum(all_weights)


def train_domain_adversarial_neural_network(
    tensor_N, tensor_R, tensor_targets_N, tensor_targets_R
):
    """Trains the model

    :param tensor_N: Non-representative data set
    :param tensor_R: Representative data set
    :return: Trained model
    """
    iterations = 1000
    batch_size = 32
    latent_features = tensor_N.shape[1] // 2
    Path("models").mkdir(exist_ok=True, parents=True)
    learning_rate = 0.001
    best_validation_metric = 0.5

    loss_function = nn.BCEWithLogitsLoss()
    tensor_N = tensor_N.to(device)
    tensor_R = tensor_R.to(device)
    tensor_targets_N = tensor_targets_N.to(device)

    dataset_n = TensorDataset(tensor_N, tensor_targets_N)
    dataset_r = TensorDataset(tensor_R)
    data_loader_n = DataLoader(dataset_n, shuffle=True, batch_size=batch_size)
    data_loader_r = DataLoader(dataset_r, shuffle=True, batch_size=batch_size)
    r_list = []
    dann_model = DANN(tensor_N.shape[1], latent_features).to(device)
    optimizer = torch.optim.Adam(dann_model.parameters(), lr=learning_rate)

    for _ in range(iterations):
        n_train, targets_n = next(iter(data_loader_n))
        r_train = next(iter(data_loader_r))[0]
        train = torch.concat([n_train, r_train])
        domain_targets_batch = torch.concat(
            [
                torch.ones(len(n_train)),
                torch.zeros(len(r_train)),
            ]
        ).to(device)

        optimizer.zero_grad()
        dann_model.train()
        class_logits, domain_logit, encodings = dann_model(train)
        singular_values_n = svdvals(encodings[:batch_size])
        singular_values_r = svdvals(encodings[batch_size:])
        max_n = torch.max(singular_values_n)
        max_r = torch.max(singular_values_r)
        batch_spectral_penalization = max_n.square() + max_r.square()

        class_logits_n = class_logits[: len(n_train)].squeeze()
        class_loss = loss_function(class_logits_n, targets_n)
        domain_loss = loss_function(domain_logit.squeeze(), domain_targets_batch)

        loss = class_loss + (lambda_weight * domain_loss) + 0.0 * batch_spectral_penalization
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            dann_model.eval()
            r_logits, _, _ = dann_model(tensor_R)
            probabilities = nn.functional.sigmoid(r_logits.cpu()).numpy()
            auroc_r = roc_auc_score(tensor_targets_R.cpu().numpy(), probabilities)
            r_list.append(auroc_r)

            n_logits, _, _ = dann_model(tensor_N)
            probabilities = nn.functional.sigmoid(n_logits.cpu()).numpy()
            auroc_class_n = roc_auc_score(tensor_targets_N.cpu().numpy(), probabilities)

        if auroc_class_n > best_validation_metric:
            best_validation_metric = auroc_class_n
            best_model = dann_model

    return best_model


def validate_model(tensor_N, tensor_R, targets, dann_model):
    """Validates the neural network

    :param tensor_N: Non-representative data set
    :param tensor_R: Representative data set
    :param mmd_loss_function: Validation function
    :param mmd_model: Neural network
    :return: Validation value
    """
    domain_targets = torch.concat(
        [
            torch.ones(len(tensor_N)),
            torch.zeros(len(tensor_R)),
        ]
    ).to(tensor_N)

    dann_model.eval()
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            class_logits_n, domain_logits_n, _ = dann_model(tensor_N)
            _, domain_logits_r, _ = dann_model(tensor_R)
        domain_logits = torch.concat([domain_logits_n, domain_logits_r]).to(tensor_N)
        class_cross_entropy = nn.functional.binary_cross_entropy_with_logits(
            class_logits_n.squeeze(), targets
        )
        domain_cross_entropy = nn.functional.binary_cross_entropy_with_logits(
            domain_logits.squeeze(), domain_targets
        )
    return class_cross_entropy, domain_cross_entropy
