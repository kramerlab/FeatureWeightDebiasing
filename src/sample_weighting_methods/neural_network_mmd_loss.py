import torch
import numpy as np
from pathlib import Path

from utils.metrics import calculate_rbf_gamma
from utils.models import WeightingMlp
from utils.weighted_mmd_loss_torch import WeightedMMDLoss
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lambda_value = 0


def neural_network_mmd_loss_weighting(N, R, columns, target, *args, **attributes):
    """Trains a neural network with weighted MMD Loss

    :param N: Non-representative data set
    :param R: Representative data set
    :param columns: Training columns
    :return: Samples weights
    """
    N_copy = N.copy().reset_index()
    N_dropped = N_copy.drop_duplicates(subset=columns)
    indices = N_dropped.index
    tensor_N = torch.FloatTensor(N_dropped[columns].values)
    tensor_R = torch.FloatTensor(R[columns].values)
    tensor_targets_N = torch.FloatTensor(N_dropped[target].values)

    all_weights = np.zeros(len(N))
    model, mmd_list = train_weighted_mmd_model(tensor_N, tensor_R, tensor_targets_N)

    if attributes["mmd_list"] is not None:
        attributes["mmd_list"].append(mmd_list)

    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            model.eval()
            weights, _ = model(tensor_N.to(device))
        weights = weights.cpu().squeeze().numpy().astype(np.float64)
    all_weights[indices] = weights
    return all_weights / sum(all_weights)


def train_weighted_mmd_model(tensor_N, tensor_R, targets_n):
    """Trains the model

    :param tensor_N: Non-representative data set
    :param tensor_R: Representative data set
    :return: Trained model
    """
    iterations = 100
    gamma = calculate_rbf_gamma(np.append(tensor_N, tensor_R, axis=0))
    latent_features = tensor_N.shape[1] // 2
    Path("models").mkdir(exist_ok=True, parents=True)
    model_path = Path(f"models/best_model_mmd_loss.pt")
    mmd_list = []
    learning_rate = 0.001
    best_validation_metric = torch.inf

    loss_function = WeightedMMDLoss(gamma, tensor_N, tensor_R, device)
    classification_loss_function = nn.BCEWithLogitsLoss()
    tensor_N = tensor_N.to(device)
    tensor_R = tensor_R.to(device)
    targets_n = targets_n.to(device)
    # data_loader_n = DataLoader(dataset_n, shuffle=True, batch_size=64)

    mmd_model = WeightingMlp(tensor_N.shape[1], latent_features).to(device)
    optimizer = torch.optim.Adam(mmd_model.parameters(), lr=learning_rate)

    for _ in range(iterations):
        indices_n = np.random.choice(len(tensor_N), len(tensor_N), replace=True)
        mmd_model.train()
        optimizer.zero_grad()
        # n_train, targets_n_batch = next(iter(data_loader_n))
        n_train = tensor_N[indices_n]
        targets_n_batch = targets_n[indices_n]
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            train_weights, logits = mmd_model(n_train)
        classification_loss = classification_loss_function(logits.squeeze(), targets_n_batch)
        mmd_loss = loss_function(train_weights, n_train.cpu())
        loss =  mmd_loss + (lambda_value * classification_loss)
        loss.backward()
        optimizer.step()

        mmd = validate_model(
            tensor_N,
            loss_function,
            mmd_model,
        )
        mmd_list.append(mmd)

        if mmd < best_validation_metric:
            best_validation_metric = mmd
            torch.save(mmd_model.state_dict(), model_path)

    mmd_model.load_state_dict(torch.load(model_path))

    return mmd_model, mmd_list


def validate_model(tensor_N, mmd_loss_function, mmd_model):
    """Validates the neural network

    :param tensor_N: Non-representative data set
    :param tensor_R: Representative data set
    :param mmd_loss_function: Validation function
    :param mmd_model: Neural network
    :return: Validation value
    """
    mmd_model.eval()
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            validation_weights, _ = mmd_model(tensor_N)
            return (
                mmd_loss_function(
                    validation_weights,
                    tensor_N.cpu(),
                )
                .cpu()
                .numpy()
            )
